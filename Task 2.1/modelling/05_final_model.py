import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, learning_curve

from settings import (
    CV_FOLDS, PREDICTORS, RANDOM_STATE, RESPONSE, holdout_split, load_complete, load_data,
    metrics, output_dir, tuning_candidates,
)


OUTPUT_DIR = output_dir("final_model")
OPTIMISATION_DIR = output_dir("model_optimisation")

df = load_data()
complete = load_complete()
train_index, test_index = holdout_split(df)

X, y = df[PREDICTORS], df[RESPONSE]
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

candidates = tuning_candidates()
selected_name = json.loads((OPTIMISATION_DIR / "selected_model.json").read_text())["selected_model"]
final_parameters = pd.read_csv(OPTIMISATION_DIR / "final_best_parameters.csv")


def build(model_name):
    # Model with the hyperparameters chosen by cross-validation on the training matches
    model = clone(candidates[model_name][0])
    parameters = json.loads(final_parameters.loc[final_parameters["model"] == model_name, "best_parameters"].iloc[0])
    return model.set_params(**parameters)


def tuned_on(model_name, X_fit, y_fit):
    # Re-tune the hyperparameters with cross-validation on the data given.
    # Returns the fitted model and the chosen parameters.
    model, grid, _ = candidates[model_name]
    if not grid:
        return clone(model).fit(X_fit, y_fit), {}
    cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=3, random_state=RANDOM_STATE)
    search = GridSearchCV(clone(model), grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
    search.fit(X_fit, y_fit)
    return search.best_estimator_, search.best_params_


# ----------------------------------------------------------------------
# 1. One final evaluation on the 21 held-out test matches
# The selected model was chosen in 04_model_optimisation.py using the
# training matches only. Other models are shown for reference; they were
# not used to make the choice.
# ----------------------------------------------------------------------
# Best tree ensemble from the nested CV (complexity rank 4), shown for reference
optimisation_summary = pd.read_csv(OPTIMISATION_DIR / "optimised_model_summary.csv")
best_ensemble = optimisation_summary[optimisation_summary["complexity_rank"] == 4].iloc[0]["model"]
test_models = {
    f"{selected_name} (selected)": build(selected_name),
    "Multiple Linear Regression": LinearRegression(),
    f"{best_ensemble} (reference)": build(best_ensemble),
    "Simple Linear Regression (market value only)": LinearRegression(),
    "Mean Baseline": DummyRegressor(strategy="mean"),
}

test_predictions = {}
test_rows = []
for model_name, model in test_models.items():
    columns = ["market_value_log_ratio"] if model_name.startswith("Simple") else PREDICTORS
    model.fit(X_train[columns], y_train)
    predicted = model.predict(X_test[columns])
    test_predictions[model_name] = predicted
    test_rows.append({"model": model_name, **metrics(y_test, predicted)})
test_performance = pd.DataFrame(test_rows)


# Bootstrap 95% confidence intervals for the test metrics
# (resample the 21 test matches 2000 times)
rng = np.random.default_rng(RANDOM_STATE)
n_boot = 2000
boot_index = rng.integers(0, len(y_test), size=(n_boot, len(y_test)))
actual = y_test.values

interval_rows = []
for model_name, predicted in test_predictions.items():
    boot = [metrics(actual[i], predicted[i]) for i in boot_index]
    row = {"model": model_name}
    for metric in ["RMSE", "MAE", "R2"]:
        values = np.array([b[metric] for b in boot])
        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.percentile(values, [2.5, 97.5])
    interval_rows.append(row)
test_performance = test_performance.merge(pd.DataFrame(interval_rows), on="model")

# Paired bootstrap: is the selected model's test RMSE lower than MLR's?
selected_key = f"{selected_name} (selected)"
selected_errors = (actual - test_predictions[selected_key]) ** 2
mlr_errors = (actual - test_predictions["Multiple Linear Regression"]) ** 2
rmse_difference = np.array([
    np.sqrt(selected_errors[i].mean()) - np.sqrt(mlr_errors[i].mean()) for i in boot_index
])
paired = pd.DataFrame([{
    "comparison": f"{selected_name} minus Multiple Linear Regression",
    "test_RMSE_difference": np.sqrt(selected_errors.mean()) - np.sqrt(mlr_errors.mean()),
    "ci_low": np.percentile(rmse_difference, 2.5),
    "ci_high": np.percentile(rmse_difference, 97.5),
    "share_of_resamples_selected_better": (rmse_difference < 0).mean(),
}])


# ----------------------------------------------------------------------
# 2. Which variables matter on unseen matches? (permutation importance)
# Each variable is shuffled 200 times in the test set; the rise in RMSE
# shows how much the selected model relies on it.
# ----------------------------------------------------------------------
selected_model = test_models[selected_key]
importance = permutation_importance(
    selected_model, X_test, y_test, scoring="neg_root_mean_squared_error",
    n_repeats=200, random_state=RANDOM_STATE,
)
permutation = pd.DataFrame({
    "predictor": PREDICTORS,
    "RMSE_increase_mean": importance.importances_mean,
    "RMSE_increase_std": importance.importances_std,
}).sort_values("RMSE_increase_mean", ascending=False)


# ----------------------------------------------------------------------
# 3. Learning curve: would more matches help?
# ----------------------------------------------------------------------
curve_rows = []
for model_name, model in [("Multiple Linear Regression", LinearRegression()), (selected_name, build(selected_name))]:
    sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, train_sizes=np.linspace(0.3, 1.0, 6),
        cv=RepeatedKFold(n_splits=CV_FOLDS, n_repeats=5, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    for size, tr, va in zip(sizes, train_scores, val_scores):
        curve_rows.append({"model": model_name, "training_matches": size,
                           "train_RMSE": -tr.mean(), "validation_RMSE": -va.mean(), "validation_RMSE_std": va.std()})
curve = pd.DataFrame(curve_rows)


# ----------------------------------------------------------------------
# 4. Time-based check: train on the group stage, test on the knockouts
# A random split mixes early and late matches. This check copies real
# forecasting: only earlier matches are used to predict later ones.
# ----------------------------------------------------------------------
group_stage = (complete["round"] == "Group stage").values
temporal_rows = []
for model_name in [selected_name, "Multiple Linear Regression"]:
    model, _ = tuned_on(model_name, X[group_stage], y[group_stage])
    temporal_rows.append({"model": model_name, **metrics(y[~group_stage], model.predict(X[~group_stage]))})
baseline = DummyRegressor(strategy="mean").fit(X[group_stage], y[group_stage])
temporal_rows.append({"model": "Mean Baseline", **metrics(y[~group_stage], baseline.predict(X[~group_stage]))})
temporal = pd.DataFrame(temporal_rows)
temporal.insert(1, "train_matches", int(group_stage.sum()))
temporal.insert(2, "test_matches", int((~group_stage).sum()))


# ----------------------------------------------------------------------
# 5. Final model for interpretation: re-tuned and refitted on all 104 matches
# ----------------------------------------------------------------------
final_model, final_model_parameters = tuned_on(selected_name, X, y)
coefficient_table = None
if hasattr(final_model, "named_steps") and hasattr(final_model.named_steps["model"], "coef_"):
    scaler = final_model.named_steps["scaler"]
    linear = final_model.named_steps["model"]
    standardised = linear.coef_
    coefficient_table = pd.DataFrame({
        "predictor": PREDICTORS,
        "standardised_coefficient": standardised,
        "original_scale_coefficient": standardised / scaler.scale_,
        "retained": standardised != 0,
    })
    intercept = linear.intercept_ - np.sum(standardised * scaler.mean_ / scaler.scale_)
    coefficient_table = pd.concat([pd.DataFrame([{
        "predictor": "intercept", "standardised_coefficient": linear.intercept_,
        "original_scale_coefficient": intercept, "retained": True,
    }]), coefficient_table], ignore_index=True)

    # Selection stability: refit on 500 bootstrap samples of the 104 matches
    # with the same alpha, and count how often each coefficient is non-zero
    stability_rng = np.random.default_rng(RANDOM_STATE)
    boot_coefficients = []
    for _ in range(500):
        sample = stability_rng.integers(0, len(X), len(X))
        boot_model = clone(final_model).fit(X.iloc[sample], y.iloc[sample])
        boot_coefficients.append(boot_model.named_steps["model"].coef_)
    boot_coefficients = np.array(boot_coefficients)
    stability = pd.DataFrame({
        "predictor": PREDICTORS,
        "share_non_zero": (boot_coefficients != 0).mean(axis=0),
        "share_positive": (boot_coefficients > 0).mean(axis=0),
        "coefficient_ci_low": np.percentile(boot_coefficients, 2.5, axis=0),
        "coefficient_ci_high": np.percentile(boot_coefficients, 97.5, axis=0),
    }).sort_values("share_non_zero", ascending=False)

final_details = pd.DataFrame([{
    "selected_model": selected_name,
    "parameters_from_training_cv": final_parameters.loc[final_parameters["model"] == selected_name, "best_parameters"].iloc[0],
    "parameters_refit_all_104": json.dumps(final_model_parameters, default=float),
}])


# Save results
test_performance.to_csv(OUTPUT_DIR / "test_set_performance.csv", index=False)
paired.to_csv(OUTPUT_DIR / "test_paired_bootstrap.csv", index=False)
permutation.to_csv(OUTPUT_DIR / "permutation_importance_test.csv", index=False)
curve.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)
temporal.to_csv(OUTPUT_DIR / "temporal_validation.csv", index=False)
final_details.to_csv(OUTPUT_DIR / "final_model_details.csv", index=False)

prediction_table = df.iloc[test_index][["match_id", "date", "team_a", "team_b"]].copy()
prediction_table["actual_goal_diff"] = actual
for model_name, predicted in test_predictions.items():
    prediction_table[model_name] = predicted
prediction_table.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

if coefficient_table is not None:
    coefficient_table.to_csv(OUTPUT_DIR / "final_coefficients.csv", index=False)
    stability.to_csv(OUTPUT_DIR / "coefficient_stability_bootstrap.csv", index=False)


# Plots
predicted = test_predictions[selected_key]
plt.figure(figsize=(6, 5))
plt.scatter(actual, predicted, alpha=0.75)
low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
plt.plot([low, high], [low, high])
plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")
plt.title(f"{selected_name}: Held-Out Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_actual_vs_predicted.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.scatter(predicted, actual - predicted, alpha=0.75)
plt.axhline(0)
plt.xlabel("Predicted Goal Difference")
plt.ylabel("Residual")
plt.title(f"{selected_name}: Test Residuals")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_residual_plot.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.barh(permutation["predictor"], permutation["RMSE_increase_mean"], xerr=permutation["RMSE_increase_std"], capsize=3)
plt.gca().invert_yaxis()
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Increase in test RMSE when the variable is shuffled")
plt.title("Permutation Importance (Held-Out Test Set)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "permutation_importance.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
for model_name, group in curve.groupby("model"):
    plt.plot(group["training_matches"], group["validation_RMSE"], marker="o", label=f"{model_name} (validation)")
    plt.plot(group["training_matches"], group["train_RMSE"], marker="o", linestyle="--", label=f"{model_name} (training)")
plt.xlabel("Number of training matches")
plt.ylabel("RMSE (goals)")
plt.title("Learning Curve")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curve.png", dpi=300)
plt.close()

if coefficient_table is not None:
    plot_data = coefficient_table[coefficient_table["predictor"] != "intercept"].sort_values("standardised_coefficient")
    plt.figure(figsize=(8, 5))
    plt.barh(plot_data["predictor"], plot_data["standardised_coefficient"])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardised coefficient (goals per 1 SD change)")
    plt.title(f"Final {selected_name} Coefficients (all 104 matches)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_coefficients.png", dpi=300)
    plt.close()


print("Final model:", selected_name)
print("\nHeld-out test set (21 matches) with bootstrap 95% intervals:")
print(test_performance.round(3).to_string(index=False))
print("\nPaired bootstrap of test RMSE:")
print(paired.round(3).to_string(index=False))
print("\nPermutation importance on the test set:")
print(permutation.round(3).to_string(index=False))
print("\nTime-based check (group stage -> knockouts):")
print(temporal.round(3).to_string(index=False))
print("\nLearning curve:")
print(curve.round(3).to_string(index=False))
if coefficient_table is not None:
    print("\nFinal coefficients (refitted on all 104 matches):")
    print(coefficient_table.round(4).to_string(index=False))
    print("\nBootstrap selection stability:")
    print(stability.round(3).to_string(index=False))
print("\nOutputs saved to:", OUTPUT_DIR)
