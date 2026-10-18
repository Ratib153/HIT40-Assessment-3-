import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold

from settings import (
    CV_FOLDS, PREDICTORS, RANDOM_STATE, RESPONSE, corrected_resampled_ttest, holdout_split,
    load_data, metrics, output_dir, summarise_folds, tuning_candidates,
)


OUTPUT_DIR = output_dir("model_optimisation")

df = load_data()
train_index, _ = holdout_split(df)
train = df.iloc[train_index].reset_index(drop=True)
X, y = train[PREDICTORS], train[RESPONSE]


candidates = tuning_candidates()


def tuned(model, grid, inner_cv):
    if not grid:
        return clone(model)
    return GridSearchCV(clone(model), grid, scoring="neg_root_mean_squared_error", cv=inner_cv, n_jobs=-1)


# ----------------------------------------------------------------------
# Nested cross-validation on the 83 training matches
# Outer loop: 5 folds x 3 repeats estimate performance (validation sets).
# Inner loop: 5 folds choose the hyperparameters using the outer training part only,
# so no validation fold is ever used to tune the model it scores.
# ----------------------------------------------------------------------
outer_splits = list(RepeatedKFold(n_splits=CV_FOLDS, n_repeats=3, random_state=RANDOM_STATE).split(X))
inner_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_rows = []
parameter_rows = []
for model_name, (model, grid, _) in candidates.items():
    start = time.time()
    for split_number, (fit_index, val_index) in enumerate(outer_splits):
        search = tuned(model, grid, inner_cv)
        search.fit(X.iloc[fit_index], y.iloc[fit_index])
        predicted = search.predict(X.iloc[val_index])
        fold_rows.append({
            "model": model_name,
            "repeat": split_number // CV_FOLDS + 1,
            "fold": split_number % CV_FOLDS + 1,
            **metrics(y.iloc[val_index], predicted),
        })
        if grid:
            parameter_rows.append({"model": model_name, "outer_split": split_number + 1,
                                   "best_parameters": json.dumps(search.best_params_, default=float)})
    print(f"Nested CV: {model_name} ({time.time() - start:.0f} s)")

nested = pd.DataFrame(fold_rows)
fold_parameters = pd.DataFrame(parameter_rows)
summary = summarise_folds(nested)
summary["complexity_rank"] = summary["model"].map({m: c[2] for m, c in candidates.items()})


# Corrected resampled t-test against multiple linear regression
n_validation = len(X) // CV_FOLDS
n_fit = len(X) - n_validation
mlr_rmse = nested[nested["model"] == "Multiple Linear Regression"].sort_values(["repeat", "fold"])["RMSE"].values
tests = []
for model_name in candidates:
    rmse = nested[nested["model"] == model_name].sort_values(["repeat", "fold"])["RMSE"].values
    difference, t_statistic, p_value = corrected_resampled_ttest(rmse, mlr_rmse, n_fit, n_validation)
    tests.append({"model": model_name, "RMSE_difference_vs_MLR": difference,
                  "t_statistic": t_statistic, "p_value": p_value})
summary = summary.merge(pd.DataFrame(tests), on="model")


# ----------------------------------------------------------------------
# Model selection: one-standard-error rule
# Choose the simplest model whose mean RMSE is within one standard error
# of the best model. With only 83 training matches, small differences in
# RMSE are not reliable, so a simpler model is preferred when it is not
# clearly worse.
# ----------------------------------------------------------------------
best = summary.iloc[0]
threshold = best["RMSE_mean"] + best["RMSE_se"]
eligible = summary[summary["RMSE_mean"] <= threshold].sort_values(["complexity_rank", "RMSE_mean"])
selected_model = eligible.iloc[0]["model"]
summary["within_one_se_of_best"] = summary["RMSE_mean"] <= threshold
summary["selected"] = summary["model"] == selected_model


# ----------------------------------------------------------------------
# Final hyperparameters: tune each model once on all 83 training matches
# (repeated 5-fold CV), to be used in 05_final_model.py
# ----------------------------------------------------------------------
final_cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=3, random_state=RANDOM_STATE)
final_rows = []
for model_name, (model, grid, _) in candidates.items():
    if grid:
        search = tuned(model, grid, final_cv).fit(X, y)
        best_parameters, best_rmse = search.best_params_, -search.best_score_
    else:
        best_parameters, best_rmse = {}, np.nan
    final_rows.append({"model": model_name, "best_parameters": json.dumps(best_parameters, default=float),
                       "tuning_cv_rmse": best_rmse})
final_parameters = pd.DataFrame(final_rows)


# Save results
nested.to_csv(OUTPUT_DIR / "nested_cv_folds.csv", index=False)
summary.to_csv(OUTPUT_DIR / "optimised_model_summary.csv", index=False)
fold_parameters.to_csv(OUTPUT_DIR / "fold_best_parameters.csv", index=False)
final_parameters.to_csv(OUTPUT_DIR / "final_best_parameters.csv", index=False)
with open(OUTPUT_DIR / "selected_model.json", "w", encoding="utf-8") as file:
    json.dump({"selected_model": selected_model, "best_model_by_rmse": best["model"],
               "one_se_threshold": threshold}, file, indent=2)


# Plot nested CV RMSE with one-standard-error bars
plt.figure(figsize=(10, 6))
colours = ["tab:green" if m == selected_model else "tab:blue" for m in summary["model"]]
plt.barh(summary["model"], summary["RMSE_mean"], xerr=summary["RMSE_se"], color=colours, capsize=3)
plt.axvline(threshold, linestyle="--", color="red", label="Best + 1 SE")
plt.gca().invert_yaxis()
plt.xlim(left=1.2)
plt.xlabel("Nested CV RMSE (mean of 15 validation folds)")
plt.title("Tuned Models (selected model in green)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "optimised_model_rmse.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=nested, x="RMSE", y="model", order=summary["model"], color="lightsteelblue")
plt.ylabel("")
plt.xlabel("Validation RMSE (goals)")
plt.title("Nested Cross-Validation Fold RMSE")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "nested_cv_boxplot.png", dpi=300)
plt.close()


print("\nModel optimisation complete")
print(summary[["model", "RMSE_mean", "RMSE_se", "R2_mean", "RMSE_difference_vs_MLR", "p_value",
               "within_one_se_of_best"]].round(4).to_string(index=False))
print("\nLowest RMSE:", best["model"])
print("Selected by the one-standard-error rule:", selected_model)
print("\nFinal parameters (tuned on all training matches):")
print(final_parameters.to_string(index=False))
print("\nOutputs saved to:", OUTPUT_DIR)
