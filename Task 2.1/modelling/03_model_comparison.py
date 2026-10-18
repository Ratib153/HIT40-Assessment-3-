import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from settings import (
    CV_FOLDS, PREDICTORS, RANDOM_STATE, RESPONSE, corrected_resampled_ttest, cv_splitter,
    holdout_split, load_data, metrics, output_dir, summarise_folds,
)


OUTPUT_DIR = output_dir("model_comparison")

df = load_data()
train_index, _ = holdout_split(df)
train = df.iloc[train_index].reset_index(drop=True)
X, y = train[PREDICTORS], train[RESPONSE]


def scaled(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


# Every model uses its default settings here; tuning happens in 04_model_optimisation.py.
# "family" groups the models for the report.
models = {
    "Mean Baseline": ("Baseline", DummyRegressor(strategy="mean")),
    "Simple Linear Regression (market value only)": ("Baseline", LinearRegression()),
    "Multiple Linear Regression": ("Linear", LinearRegression()),
    "Ridge Regression": ("Linear, regularised", scaled(Ridge())),
    "Lasso Regression": ("Linear, regularised", scaled(Lasso(max_iter=10000))),
    "Elastic Net": ("Linear, regularised", scaled(ElasticNet(max_iter=10000))),
    "Huber Regression": ("Linear, robust to outliers", scaled(HuberRegressor(max_iter=1000))),
    "Polynomial (degree 2) + Ridge": ("Non-linear", Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", Ridge()),
    ])),
    "K-Nearest Neighbours": ("Non-linear", scaled(KNeighborsRegressor())),
    "Support Vector Regression (RBF)": ("Non-linear", scaled(SVR())),
    "Decision Tree": ("Tree", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    "Random Forest": ("Ensemble", RandomForestRegressor(random_state=RANDOM_STATE)),
    "Extra Trees": ("Ensemble", ExtraTreesRegressor(random_state=RANDOM_STATE)),
    "Gradient Boosting": ("Ensemble", GradientBoostingRegressor(random_state=RANDOM_STATE)),
}

single_predictor = ["market_value_log_ratio"]


# Repeated 5-fold cross-validation on the training matches.
# Every model sees exactly the same 50 train/validation splits.
splits = list(cv_splitter().split(X))

fold_rows = []
oof_rows = []

for model_name, (family, model) in models.items():
    columns = single_predictor if model_name.startswith("Simple") else PREDICTORS
    for split_number, (fit_index, val_index) in enumerate(splits):
        fitted = clone(model).fit(X.iloc[fit_index][columns], y.iloc[fit_index])
        predicted = fitted.predict(X.iloc[val_index][columns])

        fold_rows.append({
            "model": model_name,
            "family": family,
            "repeat": split_number // CV_FOLDS + 1,
            "fold": split_number % CV_FOLDS + 1,
            **metrics(y.iloc[val_index], predicted),
        })
        oof_rows.append(pd.DataFrame({
            "model": model_name,
            "repeat": split_number // CV_FOLDS + 1,
            "row": val_index,
            "actual": y.iloc[val_index].values,
            "predicted": predicted,
        }))
    print("Cross-validated:", model_name)

fold_results = pd.DataFrame(fold_rows)
oof = pd.concat(oof_rows, ignore_index=True)


# Summary of the 50 validation folds
summary = summarise_folds(fold_results)
summary.insert(1, "family", summary["model"].map({name: family for name, (family, _) in models.items()}))

# Pooled out-of-fold R2: each repeat predicts every training match once,
# which is more stable than averaging R2 over small folds
pooled = (
    oof.groupby(["model", "repeat"])
    .apply(lambda g: metrics(g["actual"], g["predicted"])["R2"], include_groups=False)
    .groupby("model").mean()
)
summary["R2_pooled_out_of_fold"] = summary["model"].map(pooled)


# Is each model better than multiple linear regression?
# Corrected resampled t-test on the 50 paired fold RMSEs (negative = lower RMSE than MLR)
n_validation = len(X) // CV_FOLDS
n_fit = len(X) - n_validation
mlr_rmse = fold_results[fold_results["model"] == "Multiple Linear Regression"].sort_values(["repeat", "fold"])["RMSE"].values

test_rows = []
for model_name in models:
    model_rmse = fold_results[fold_results["model"] == model_name].sort_values(["repeat", "fold"])["RMSE"].values
    difference, t_statistic, p_value = corrected_resampled_ttest(model_rmse, mlr_rmse, n_fit, n_validation)
    test_rows.append({
        "model": model_name,
        "RMSE_difference_vs_MLR": difference,
        "t_statistic": t_statistic,
        "p_value": p_value,
    })
significance = pd.DataFrame(test_rows)
summary = summary.merge(significance, on="model", how="left")


# Save results
fold_results.to_csv(OUTPUT_DIR / "cross_validation_folds.csv", index=False)
summary.to_csv(OUTPUT_DIR / "cross_validation_summary.csv", index=False)
oof.to_csv(OUTPUT_DIR / "cross_validation_predictions.csv", index=False)


# Box plot of the 50 fold RMSEs for each model
order = summary["model"].tolist()
plt.figure(figsize=(11, 6))
sns.boxplot(data=fold_results, x="RMSE", y="model", order=order, color="lightsteelblue")
plt.ylabel("")
plt.axvline(summary.loc[summary["model"] == "Multiple Linear Regression", "RMSE_mean"].iloc[0],
            linestyle="--", color="red", label="MLR mean RMSE")
plt.xlabel("Validation RMSE (goals)")
plt.title("Repeated 5-Fold Cross-Validation (10 repeats, 83 training matches)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_rmse_comparison.png", dpi=300)
plt.close()


print("\nModel comparison complete (default settings)")
print(summary[["model", "RMSE_mean", "RMSE_std", "MAE_mean", "R2_pooled_out_of_fold",
               "RMSE_difference_vs_MLR", "p_value"]].round(4).to_string(index=False))
print("\nOutputs saved to:", OUTPUT_DIR)
