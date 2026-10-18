from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LinearRegression, lasso_path
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from settings import (
    CV_FOLDS, PREDICTORS, RESPONSE, corrected_resampled_ttest, cv_splitter, holdout_split,
    load_complete, load_data, output_dir,
)


# Evidence for how the 8 explanatory variables were chosen and transformed.
# Everything here uses the 83 training matches only.
OUTPUT_DIR = output_dir("variable_selection")

df = load_data()
complete = load_complete()
assert (complete["match_id"] == df["match_id"]).all()

train_index, _ = holdout_split(df)
train = complete.iloc[train_index].reset_index(drop=True)
y = train[RESPONSE]

# Extra candidate versions of some variables, for the specification checks
train["market_value_diff_m"] = train["team_a_market_value_m"] - train["team_b_market_value_m"]
train["rest_days_capped_log_ratio"] = np.log(train["team_a_rest_days"].clip(upper=14) / train["team_b_rest_days"].clip(upper=14))

splits = list(cv_splitter().split(train))
n_validation = len(train) // CV_FOLDS
n_fit = len(train) - n_validation


def cv_rmse(columns, model=None):
    model = model if model is not None else LinearRegression()
    scores = cross_val_score(model, train[columns], y, cv=splits, scoring="neg_root_mean_squared_error")
    return -scores


def swap(old, new):
    return [new if p == old else p for p in PREDICTORS]


# ----------------------------------------------------------------------
# 1. Specification checks: alternative versions of a variable
# ----------------------------------------------------------------------
specifications = {
    "Chosen 8 variables": PREDICTORS,
    "Elo rating instead of FIFA points": swap("fifa_points_diff", "elo_rating_diff"),
    "Raw rest-day difference instead of log ratio": swap("rest_days_log_ratio", "rest_days_diff"),
    "Rest days capped at 14 before the log ratio": swap("rest_days_log_ratio", "rest_days_capped_log_ratio"),
    "Raw market value difference instead of log ratio": swap("market_value_log_ratio", "market_value_diff_m"),
}

chosen_scores = cv_rmse(PREDICTORS)
specification_rows = []
for name, columns in specifications.items():
    scores = cv_rmse(columns)
    difference, t_statistic, p_value = corrected_resampled_ttest(scores, chosen_scores, n_fit, n_validation)
    specification_rows.append({
        "specification": name,
        "model": "Multiple Linear Regression",
        "RMSE_mean": scores.mean(),
        "RMSE_std": scores.std(ddof=1),
        "RMSE_difference_vs_chosen": difference,
        "p_value": p_value,
    })
specification_table = pd.DataFrame(specification_rows)


# Collinearity check that ruled out using FIFA points and Elo together
strength_correlation = pd.DataFrame([{
    "pair": "fifa_points_diff vs elo_rating_diff",
    "pearson_r": train["fifa_points_diff"].corr(train["elo_rating_diff"]),
}, {
    "pair": "fifa_points_diff vs market_value_log_ratio",
    "pearson_r": train["fifa_points_diff"].corr(train["market_value_log_ratio"]),
}, {
    "pair": "elo_rating_diff vs market_value_log_ratio",
    "pearson_r": train["elo_rating_diff"].corr(train["market_value_log_ratio"]),
}])


# ----------------------------------------------------------------------
# 2. Drop-one-variable check
# A positive change means validation RMSE gets worse without the variable.
# ----------------------------------------------------------------------
ablation_rows = []
for predictor in PREDICTORS:
    columns = [p for p in PREDICTORS if p != predictor]
    scores = cv_rmse(columns)
    difference, t_statistic, p_value = corrected_resampled_ttest(scores, chosen_scores, n_fit, n_validation)
    ablation_rows.append({
        "variable_removed": predictor,
        "RMSE_without_variable": scores.mean(),
        "RMSE_change_when_removed": difference,
        "p_value": p_value,
    })
ablation = pd.DataFrame(ablation_rows).sort_values("RMSE_change_when_removed", ascending=False)


# ----------------------------------------------------------------------
# 3. All 255 possible subsets of the 8 variables (best-subset search)
# ----------------------------------------------------------------------
subset_rows = []
for size in range(1, len(PREDICTORS) + 1):
    for subset in combinations(PREDICTORS, size):
        scores = cv_rmse(list(subset))
        subset_rows.append({"size": size, "variables": " + ".join(subset), "RMSE_mean": scores.mean()})
subsets = pd.DataFrame(subset_rows).sort_values("RMSE_mean").reset_index(drop=True)

best_by_size = subsets.loc[subsets.groupby("size")["RMSE_mean"].idxmin()].sort_values("size")
top_20 = subsets.head(20)
inclusion = pd.DataFrame({
    "variable": PREDICTORS,
    "share_of_top_20_subsets": [top_20["variables"].str.contains(p, regex=False).mean() for p in PREDICTORS],
}).sort_values("share_of_top_20_subsets", ascending=False)


# ----------------------------------------------------------------------
# 4. Lasso path: the order in which variables enter as the penalty falls
# ----------------------------------------------------------------------
X_scaled = StandardScaler().fit_transform(train[PREDICTORS])
alphas = np.logspace(-3, 0.3, 60)
path_alphas, path_coefficients, _ = lasso_path(X_scaled, y - y.mean(), alphas=alphas)

alpha_scores = []
for alpha in alphas:
    model = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=20000))])
    alpha_scores.append(cv_rmse(PREDICTORS, model).mean())
alpha_table = pd.DataFrame({"alpha": alphas, "cv_RMSE": alpha_scores})
best_alpha = alpha_table.loc[alpha_table["cv_RMSE"].idxmin(), "alpha"]

entry_rows = []
for index, predictor in enumerate(PREDICTORS):
    non_zero = np.where(np.abs(path_coefficients[index]) > 1e-10)[0]
    entry_rows.append({"variable": predictor,
                       "largest_alpha_with_non_zero_coefficient": path_alphas[non_zero].max() if len(non_zero) else 0})
entry_order = pd.DataFrame(entry_rows).sort_values("largest_alpha_with_non_zero_coefficient", ascending=False)


# Save results
specification_table.to_csv(OUTPUT_DIR / "specification_checks.csv", index=False)
strength_correlation.to_csv(OUTPUT_DIR / "strength_measure_correlations.csv", index=False)
ablation.to_csv(OUTPUT_DIR / "drop_one_variable.csv", index=False)
subsets.to_csv(OUTPUT_DIR / "all_subsets.csv", index=False)
best_by_size.to_csv(OUTPUT_DIR / "best_subset_by_size.csv", index=False)
inclusion.to_csv(OUTPUT_DIR / "top_subset_inclusion.csv", index=False)
alpha_table.to_csv(OUTPUT_DIR / "lasso_alpha_cv.csv", index=False)
entry_order.to_csv(OUTPUT_DIR / "lasso_entry_order.csv", index=False)


# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for index, predictor in enumerate(PREDICTORS):
    axes[0].plot(path_alphas, path_coefficients[index], label=predictor)
axes[0].set_xscale("log")
axes[0].axvline(best_alpha, linestyle="--", color="black", label="lowest CV RMSE")
axes[0].set_xlabel("Lasso alpha (penalty)")
axes[0].set_ylabel("Standardised coefficient")
axes[0].set_title("Lasso Coefficient Path")
axes[0].legend(fontsize=7)
axes[1].plot(alpha_table["alpha"], alpha_table["cv_RMSE"], marker=".")
axes[1].set_xscale("log")
axes[1].axvline(best_alpha, linestyle="--", color="black")
axes[1].set_xlabel("Lasso alpha (penalty)")
axes[1].set_ylabel("Repeated CV RMSE")
axes[1].set_title("Validation Error by Penalty")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lasso_path.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(best_by_size["size"], best_by_size["RMSE_mean"], marker="o")
plt.axhline(chosen_scores.mean(), linestyle="--", color="red", label="All 8 variables")
plt.xlabel("Number of variables in the model")
plt.ylabel("Repeated CV RMSE of the best subset")
plt.title("Best-Subset Search (Multiple Linear Regression)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_subset_by_size.png", dpi=300)
plt.close()


print("Variable selection evidence complete")
print("\nSpecification checks (MLR, repeated CV on training):")
print(specification_table.round(4).to_string(index=False))
print("\nStrength measure correlations:")
print(strength_correlation.round(3).to_string(index=False))
print("\nDrop-one-variable check:")
print(ablation.round(4).to_string(index=False))
print("\nBest subset of each size:")
print(best_by_size.round(4).to_string(index=False))
print("\nShare of the 20 best subsets that include each variable:")
print(inclusion.round(2).to_string(index=False))
print("\nOrder in which variables enter the Lasso:")
print(entry_order.round(4).to_string(index=False))
print("\nOutputs saved to:", OUTPUT_DIR)
