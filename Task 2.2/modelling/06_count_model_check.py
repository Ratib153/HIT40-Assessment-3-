"""
Task 2.2 - Step 6: count-model and assumption checks.

goals_scored is a non-negative count, so this script checks:
1. The linear regression assumptions for the 8-variable MLR
   (linearity, constant variance, normal residuals, independence).
2. Whether coefficient inference changes when standard errors are
   clustered by match (the two rows of a match are not independent).
3. Whether a Poisson regression (log link, scikit-learn PoissonRegressor)
   predicts better than the MLR and the selected Lasso, using exactly the
   same nested grouped cross-validation folds as 05_final_model.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "team_matches_208_model.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "count_model_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

response = "goals_scored"
predictors = [
    "fifa_points_diff",
    "rest_days",
    "venue_altitude_m",
    "attacking_form_5",
    "opponent_defensive_form_5",
    "previous_extra_time",
    "lineup_changes",
    "qualification_pressure",
]

lasso_alphas = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]          # same grid as step 5
poisson_alphas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

df = pd.read_csv(DATA_FILE).sort_values(["date", "match_id", "team"]).reset_index(drop=True)
assert len(df) == 208

# Step 5 reads the file in its stored order; use the same order for identical folds
df_cv = pd.read_csv(DATA_FILE)
X = df_cv[predictors]
y = df_cv[response]
groups = df_cv["match_id"]


# ---------------------------------------------------------------------
# 1. Assumption tests for the 8-variable MLR (all 208 rows)
# ---------------------------------------------------------------------
X_ols = sm.add_constant(df[predictors])
ols = sm.OLS(df[response], X_ols).fit()
resid = ols.resid

reset = linear_reset(ols, power=2, use_f=True)
bp = het_breuschpagan(resid, X_ols)
sw = stats.shapiro(resid)
dw = durbin_watson(resid)

# Correlation of residuals between the two teams in the same match
pair = df.assign(resid=resid.values).sort_values(["match_id", "team"])
first = pair.groupby("match_id")["resid"].first()
second = pair.groupby("match_id")["resid"].last()
within_match_r, within_match_p = stats.pearsonr(first, second)

assumptions = pd.DataFrame([
    ["Linearity", "Ramsey RESET (squared fitted values)", reset.fvalue, reset.pvalue],
    ["Constant variance", "Breusch-Pagan", bp[0], bp[1]],
    ["Normal residuals", "Shapiro-Wilk", sw.statistic, sw.pvalue],
    ["Independent residuals", "Durbin-Watson (date order, about 2 is good)", dw, np.nan],
    ["Independent residuals", "Within-match residual correlation (team vs opponent)",
     within_match_r, within_match_p],
], columns=["assumption", "test", "statistic", "p_value"])
assumptions["conclusion_at_5_percent"] = np.where(
    assumptions["p_value"].isna(), "See statistic",
    np.where(assumptions["p_value"] < 0.05, "Evidence against", "No evidence against"),
)
assumptions.to_csv(OUTPUT_DIR / "mlr_assumption_tests.csv", index=False)

vif = pd.DataFrame({
    "predictor": predictors,
    "VIF": [variance_inflation_factor(X_ols.values, i + 1) for i in range(len(predictors))],
}).sort_values("VIF", ascending=False)
vif.to_csv(OUTPUT_DIR / "variance_inflation_factors.csv", index=False)

# Standard vs match-clustered standard errors
ols_cluster = sm.OLS(df[response], X_ols).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(df["match_id"])[0]}
)
inference = pd.DataFrame({
    "variable": ols.params.index,
    "coefficient": ols.params.values,
    "p_value_standard": ols.pvalues.values,
    "p_value_cluster_by_match": ols_cluster.pvalues.values,
})
inference.to_csv(OUTPUT_DIR / "ols_cluster_robust_inference.csv", index=False)


# ---------------------------------------------------------------------
# 2. Poisson GLM on all rows (interpretation + over-dispersion check)
# ---------------------------------------------------------------------
glm = sm.GLM(df[response], X_ols, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(df["match_id"])[0]}
)
dispersion = float(
    sm.GLM(df[response], X_ols, family=sm.families.Poisson()).fit().pearson_chi2
    / (len(df) - len(predictors) - 1)
)
sd = df[predictors].std()
glm_table = pd.DataFrame({
    "variable": glm.params.index,
    "coefficient_log_scale": glm.params.values,
    "rate_ratio_per_unit": np.exp(glm.params.values),
    "p_value_cluster_by_match": glm.pvalues.values,
})
glm_table["rate_ratio_per_1_sd"] = [
    np.nan if v == "const" else float(np.exp(glm.params[v] * sd[v])) for v in glm.params.index
]
glm_table.to_csv(OUTPUT_DIR / "poisson_glm_coefficients.csv", index=False)
pd.DataFrame([{
    "pearson_dispersion": dispersion,
    "mean_goals": df[response].mean(),
    "variance_goals": df[response].var(),
    "share_zero_goals_observed": (df[response] == 0).mean(),
}]).to_csv(OUTPUT_DIR / "dispersion_check.csv", index=False)


# ---------------------------------------------------------------------
# 3. Nested grouped CV: same outer folds as 05_final_model.py
# ---------------------------------------------------------------------
def scaled(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


fold_rows, pred_rows = [], []
outer_cv = GroupKFold(n_splits=5)

for fold, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    inner = GroupKFold(n_splits=4)

    lasso = GridSearchCV(scaled(Lasso(max_iter=10000)), {"model__alpha": lasso_alphas},
                         scoring="neg_root_mean_squared_error", cv=inner, n_jobs=1)
    lasso.fit(X_tr, y_tr, groups=groups.iloc[tr])

    poisson = GridSearchCV(scaled(PoissonRegressor(max_iter=3000)),
                           {"model__alpha": poisson_alphas},
                           scoring="neg_root_mean_squared_error", cv=inner, n_jobs=1)
    poisson.fit(X_tr, y_tr, groups=groups.iloc[tr])

    models = {
        "Poisson Regression": poisson.best_estimator_,
        "Lasso Regression": lasso.best_estimator_,
        "Multiple Linear Regression": scaled(LinearRegression()).fit(X_tr, y_tr),
        "Mean Baseline": DummyRegressor(strategy="mean").fit(X_tr, y_tr),
    }

    for name, m in models.items():
        p = m.predict(X_te)
        fold_rows.append({
            "fold": fold, "model": name,
            "MAE": mean_absolute_error(y_te, p),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, p))),
            "R2": r2_score(y_te, p),
            # deviance needs positive predictions; clip only for this metric
            "Poisson_deviance": mean_poisson_deviance(y_te, np.clip(p, 1e-6, None)),
            "share_negative_raw_predictions": float((m.predict(X_te) < 0).mean()),
            "alpha": (poisson.best_params_["model__alpha"] if name == "Poisson Regression"
                      else lasso.best_params_["model__alpha"] if name == "Lasso Regression"
                      else np.nan),
        })
        for idx, val in zip(te, p):
            pred_rows.append({"row": idx, "fold": fold, "model": name,
                              "actual": y.iloc[idx], "predicted": val})

folds = pd.DataFrame(fold_rows)
folds.to_csv(OUTPUT_DIR / "nested_cv_folds.csv", index=False)
pd.DataFrame(pred_rows).to_csv(OUTPUT_DIR / "nested_cv_predictions.csv", index=False)

summary = folds.groupby("model").agg(
    MAE_mean=("MAE", "mean"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
    R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    Poisson_deviance_mean=("Poisson_deviance", "mean"),
).sort_values("RMSE_mean").reset_index()

# Paired t-test on fold RMSE vs Poisson
pois = folds[folds.model == "Poisson Regression"].set_index("fold")["RMSE"]
summary["RMSE_difference_vs_Poisson"] = np.nan
summary["paired_p_value"] = np.nan
for i, r in summary.iterrows():
    other = folds[folds.model == r.model].set_index("fold")["RMSE"]
    summary.loc[i, "RMSE_difference_vs_Poisson"] = (other - pois).mean()
    if r.model != "Poisson Regression":
        summary.loc[i, "paired_p_value"] = stats.ttest_rel(other, pois).pvalue
summary.to_csv(OUTPUT_DIR / "nested_cv_summary.csv", index=False)


# ---------------------------------------------------------------------
# 4. Predicted vs observed goal-count distribution (out-of-fold)
# ---------------------------------------------------------------------
preds = pd.DataFrame(pred_rows)
counts = np.arange(0, 6)
observed = [(y == k).mean() if k < 5 else (y >= 5).mean() for k in counts]
pois_mu = preds[preds.model == "Poisson Regression"]["predicted"].values
expected = [stats.poisson.pmf(k, pois_mu).mean() if k < 5
            else stats.poisson.sf(4, pois_mu).mean() for k in counts]
pd.DataFrame({"goals": [str(k) if k < 5 else "5+" for k in counts],
              "observed_share": observed,
              "poisson_expected_share": expected}).to_csv(
    OUTPUT_DIR / "goal_count_fit.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
w = 0.38
ax.bar(counts - w / 2, observed, w, label="Observed")
ax.bar(counts + w / 2, expected, w, label="Poisson model (out-of-fold)")
ax.set_xticks(counts, [str(k) if k < 5 else "5+" for k in counts])
ax.set_xlabel("Goals scored by team")
ax.set_ylabel("Share of team-matches")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "goal_count_fit.png", dpi=200)
plt.close(fig)

print(assumptions.to_string(index=False))
print("\nWithin-match residual r:", round(within_match_r, 3))
print("Pearson dispersion:", round(dispersion, 3))
print("\n", inference.to_string(index=False))
print("\n", glm_table.to_string(index=False))
print("\n", summary.to_string(index=False))
print("\nFold alphas:\n", folds.dropna(subset=["alpha"])[["fold", "model", "alpha"]].to_string(index=False))
