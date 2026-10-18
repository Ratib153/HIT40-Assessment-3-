import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

from settings import (
    PREDICTORS, RESPONSE, cv_splitter, holdout_split, load_data, metrics, output_dir,
)


OUTPUT_DIR = output_dir("mlr")

df = load_data()
train_index, test_index = holdout_split(df)

X, y = df[PREDICTORS], df[RESPONSE]
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# ----------------------------------------------------------------------
# 1. Simple train/test split (as in the week 8 lab)
# ----------------------------------------------------------------------
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
baseline_pred = DummyRegressor(strategy="mean").fit(X_train, y_train).predict(X_test)

value_range = y.max() - y.min()
performance = pd.DataFrame([
    {"model": "Multiple Linear Regression", "evaluation": "Test set (21 matches)", **metrics(y_test, y_pred)},
    {"model": "Mean Baseline", "evaluation": "Test set (21 matches)", **metrics(y_test, baseline_pred)},
])


# ----------------------------------------------------------------------
# 2. Repeated k-fold cross-validation on the training matches
# A single 21-match test set gives a noisy estimate, so the same model is
# also validated 50 times (5 folds x 10 repeats) inside the training data.
# ----------------------------------------------------------------------
scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}
for name, estimator in [("Multiple Linear Regression", LinearRegression()),
                        ("Mean Baseline", DummyRegressor(strategy="mean"))]:
    scores = cross_validate(estimator, X_train, y_train, cv=cv_splitter(), scoring=scoring)
    performance = pd.concat([performance, pd.DataFrame([{
        "model": name,
        "evaluation": "Repeated 5-fold CV on training (mean of 50 folds)",
        "MAE": -scores["test_MAE"].mean(),
        "MSE": np.mean(scores["test_RMSE"] ** 2),
        "RMSE": -scores["test_RMSE"].mean(),
        "R2": scores["test_R2"].mean(),
        "RMSE_std": scores["test_RMSE"].std(ddof=1),
    }])], ignore_index=True)

performance["Normalised_RMSE"] = performance["RMSE"] / value_range
performance.to_csv(OUTPUT_DIR / "mlr_performance.csv", index=False)


# Coefficients and test predictions
pd.DataFrame({"predictor": PREDICTORS, "coefficient": model.coef_}).to_csv(
    OUTPUT_DIR / "mlr_coefficients.csv", index=False)

predictions = df.iloc[test_index][["match_id", "date", "team_a", "team_b"]].copy()
predictions["actual_goal_diff"] = y_test.values
predictions["predicted_goal_diff"] = y_pred
predictions["residual"] = predictions["actual_goal_diff"] - predictions["predicted_goal_diff"]
predictions.to_csv(OUTPUT_DIR / "mlr_test_predictions.csv", index=False)


# ----------------------------------------------------------------------
# 3. Statistical inference and assumption checks (statsmodels OLS)
# Fitted on the training matches; an all-104 fit is also saved because it
# gives the most precise coefficient estimates for interpretation.
# ----------------------------------------------------------------------
X_train_const = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_train_const).fit()
ols_robust = ols.get_robustcov_results(cov_type="HC3")
ols_all = sm.OLS(y, sm.add_constant(X)).fit()

with open(OUTPUT_DIR / "mlr_statsmodels_summary.txt", "w", encoding="utf-8") as file:
    file.write("OLS fitted on the 83 training matches\n\n")
    file.write(ols.summary().as_text())
    file.write("\n\n\nSame model with HC3 heteroscedasticity-robust standard errors\n\n")
    file.write(ols_robust.summary().as_text())
    file.write("\n\n\nOLS fitted on all 104 matches\n\n")
    file.write(ols_all.summary().as_text())

ols_results = pd.DataFrame({
    "variable": ols.params.index,
    "coefficient": ols.params.values,
    "std_error": ols.bse.values,
    "p_value": ols.pvalues.values,
    "robust_HC3_p_value": ols_robust.pvalues,
    "confidence_interval_low": ols.conf_int()[0].values,
    "confidence_interval_high": ols.conf_int()[1].values,
    "coefficient_all_104": ols_all.params.values,
    "p_value_all_104": ols_all.pvalues.values,
})
ols_results.to_csv(OUTPUT_DIR / "mlr_ols_results.csv", index=False)


# Assumption tests
residuals = ols.resid
bp_lm, bp_p, _, _ = het_breuschpagan(residuals, X_train_const)
shapiro = stats.shapiro(residuals)
reset = linear_reset(ols, power=2, use_f=True)

# Durbin-Watson needs residuals in time order
order = np.argsort(df.iloc[train_index]["date"].values, kind="stable")

assumptions = pd.DataFrame([
    {"assumption": "Linearity", "test": "Ramsey RESET (squared fitted values)",
     "statistic": float(reset.fvalue), "p_value": float(reset.pvalue)},
    {"assumption": "Constant variance", "test": "Breusch-Pagan",
     "statistic": bp_lm, "p_value": bp_p},
    {"assumption": "Normal residuals", "test": "Shapiro-Wilk",
     "statistic": shapiro.statistic, "p_value": shapiro.pvalue},
    {"assumption": "Independent residuals", "test": "Durbin-Watson (matches in date order, about 2 is good)",
     "statistic": durbin_watson(residuals.values[order]), "p_value": np.nan},
])
assumptions["conclusion_at_5_percent"] = np.where(
    assumptions["p_value"].isna(), "See statistic",
    np.where(assumptions["p_value"] < 0.05, "Assumption questionable", "No evidence against"),
)
assumptions.to_csv(OUTPUT_DIR / "assumption_tests.csv", index=False)


# Influential matches (Cook's distance above 4/n is a common rule of thumb)
influence = ols.get_influence()
influential = df.iloc[train_index][["match_id", "date", "team_a", "team_b", RESPONSE]].copy()
influential["fitted"] = ols.fittedvalues.values
influential["studentised_residual"] = influence.resid_studentized_external
influential["leverage"] = influence.hat_matrix_diag
influential["cooks_distance"] = influence.cooks_distance[0]
influential = influential.sort_values("cooks_distance", ascending=False)
influential.to_csv(OUTPUT_DIR / "influential_matches.csv", index=False)
cooks_threshold = 4 / len(train_index)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
minimum, maximum = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([minimum, maximum], [minimum, maximum])
plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")
plt.title("Multiple Linear Regression: Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.scatter(ols.fittedvalues, residuals, alpha=0.7)
plt.axhline(y=0)
plt.xlabel("Fitted Goal Difference")
plt.ylabel("Residual")
plt.title("Residuals vs Fitted (Training Matches)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(residuals, bins=15, edgecolor="black")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Number of Matches")
axes[0].set_title("Distribution of Residuals")
sm.qqplot(residuals, line="45", fit=True, ax=axes[1])
axes[1].set_title("Normal Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 4.5))
cooks = influence.cooks_distance[0]
plt.stem(np.arange(len(cooks)), cooks, markerfmt=",")
plt.axhline(cooks_threshold, linestyle="--", color="red", label="4/n")
plt.xlabel("Training match (index)")
plt.ylabel("Cook's distance")
plt.title("Influence of Each Training Match")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cooks_distance.png", dpi=300)
plt.close()


# Print main results
print("Multiple linear regression complete")
print("Training matches:", len(train_index), " Test matches:", len(test_index))
print("\nPerformance:")
print(performance.round(4).to_string(index=False))
print("\nOLS (training): R2 = %.4f, adjusted R2 = %.4f" % (ols.rsquared, ols.rsquared_adj))
print("OLS (all 104):  R2 = %.4f, adjusted R2 = %.4f" % (ols_all.rsquared, ols_all.rsquared_adj))
print(ols_results[["variable", "coefficient", "p_value", "robust_HC3_p_value"]].round(4).to_string(index=False))
print("\nAssumption tests:")
print(assumptions.round(4).to_string(index=False))
print("\nMatches above the Cook's distance threshold (%.3f):" % cooks_threshold)
print(influential[influential["cooks_distance"] > cooks_threshold][
    ["match_id", "team_a", "team_b", RESPONSE, "cooks_distance"]].round(3).to_string(index=False))
print("\nOutputs saved to:", OUTPUT_DIR)
