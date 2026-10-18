import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from settings import (
    CV_FOLDS, PREDICTORS, RANDOM_STATE, RESPONSE, corrected_resampled_ttest, cv_splitter,
    holdout_split, load_data, metrics, output_dir, tuning_candidates,
)


# Swapped-teams check
# Which team is "team A" only depends on how FBref lists the match.
# Every explanatory variable is a team A vs team B difference (or log ratio),
# so swapping the two teams flips the sign of every variable and of goal_diff.
# A fair model should therefore give the same prediction with the opposite sign:
#     f(swapped match) = -f(original match)
# This script checks (1) whether being listed first matters, (2) how far each
# model breaks this rule, and (3) whether a model that obeys it predicts better.
# Everything uses the 83 training matches only.
OUTPUT_DIR = output_dir("swapped_teams_check")

df = load_data()
train_index, _ = holdout_split(df)
train = df.iloc[train_index].reset_index(drop=True)
X, y = train[PREDICTORS], train[RESPONSE]

candidates = tuning_candidates()
lasso_grid = candidates["Lasso Regression"][1]
extra_trees_grid = candidates["Extra Trees"][1]
inner_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def swapped(X_part, y_part=None):
    # Swapping the teams negates every difference and log ratio (host_diff included)
    return (-X_part, None) if y_part is None else (-X_part, -y_part)


# ----------------------------------------------------------------------
# 1. Does being listed as team A matter?
# The intercept of the regression is the predicted goal difference for two
# identical teams. If it is not different from 0, listing order does not matter.
# ----------------------------------------------------------------------
ols = sm.OLS(y, sm.add_constant(X)).fit()
listing_effect = pd.DataFrame([{
    "check": "Mean goal difference (team A - team B)",
    "estimate": y.mean(),
    "p_value": stats.ttest_1samp(y, 0).pvalue,
    "note": "Raw difference, not adjusted for team strength",
}, {
    "check": "OLS intercept with all 8 variables",
    "estimate": ols.params["const"],
    "p_value": ols.pvalues["const"],
    "note": "Listing effect after adjusting for team strength",
}, {
    "check": "Mean fifa_points_diff",
    "estimate": X["fifa_points_diff"].mean(),
    "p_value": stats.ttest_1samp(X["fifa_points_diff"], 0).pvalue,
    "note": "Positive means FBref tends to list the stronger team first",
}, {
    "check": "Mean market_value_log_ratio",
    "estimate": X["market_value_log_ratio"].mean(),
    "p_value": stats.ttest_1samp(X["market_value_log_ratio"], 0).pvalue,
    "note": "Positive means FBref tends to list the more valuable team first",
}])


# ----------------------------------------------------------------------
# 2 and 3. Compare standard models with versions that obey the swap rule
# ----------------------------------------------------------------------

class Augmented:
    """Train on every match twice (as listed and swapped), then average
    the prediction with the negated prediction for the swapped match.
    This makes any model obey f(swapped) = -f(original)."""

    def __init__(self, model):
        self.model = model

    def fit(self, X_fit, y_fit):
        X_swap, y_swap = swapped(X_fit, y_fit)
        self.model.fit(pd.concat([X_fit, X_swap]), pd.concat([y_fit, y_swap]))
        return self

    def predict(self, X_new):
        return (self.model.predict(X_new) - self.model.predict(-X_new)) / 2


models = {
    "Multiple Linear Regression": (lambda: LinearRegression(), "standard"),
    "Multiple Linear Regression, no intercept": (lambda: LinearRegression(fit_intercept=False), "symmetric"),
    "Lasso Regression (selected)": (
        lambda: GridSearchCV(Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=20000))]),
                             lasso_grid, scoring="neg_root_mean_squared_error", cv=inner_cv),
        "standard"),
    # Scaling without centring and no intercept keep the Lasso symmetric
    "Lasso Regression, no intercept": (
        lambda: GridSearchCV(Pipeline([("scaler", StandardScaler(with_mean=False)),
                                       ("model", Lasso(fit_intercept=False, max_iter=20000))]),
                             lasso_grid, scoring="neg_root_mean_squared_error", cv=inner_cv),
        "symmetric"),
    "Extra Trees": (
        lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE),
        "standard"),
    "Extra Trees, trained on both orientations": (
        lambda: Augmented(ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE)),
        "symmetric"),
}

splits = list(cv_splitter().split(X))
fold_rows = []
for model_name, (make_model, kind) in models.items():
    for split_number, (fit_index, val_index) in enumerate(splits):
        model = make_model().fit(X.iloc[fit_index], y.iloc[fit_index])
        X_val, y_val = X.iloc[val_index], y.iloc[val_index]
        predicted = model.predict(X_val)
        predicted_swapped = model.predict(-X_val)

        fold_rows.append({
            "model": model_name,
            "type": kind,
            "repeat": split_number // CV_FOLDS + 1,
            "fold": split_number % CV_FOLDS + 1,
            **metrics(y_val, predicted),
            # RMSE when the validation matches are given with the teams swapped
            "RMSE_swapped_orientation": metrics(-y_val, predicted_swapped)["RMSE"],
            # 0 when the model obeys f(swapped) = -f(original)
            "symmetry_violation": np.mean(np.abs(predicted + predicted_swapped)),
        })
    print("Checked:", model_name)

folds = pd.DataFrame(fold_rows)
summary = (
    folds.groupby(["model", "type"], as_index=False, sort=False)
    .agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
         RMSE_swapped_orientation=("RMSE_swapped_orientation", "mean"),
         R2_mean=("R2", "mean"),
         symmetry_violation_goals=("symmetry_violation", "mean"))
)


# Does the symmetric version predict better or worse than its standard version?
n_validation = len(X) // CV_FOLDS
n_fit = len(X) - n_validation
pairs = [
    ("Multiple Linear Regression, no intercept", "Multiple Linear Regression"),
    ("Lasso Regression, no intercept", "Lasso Regression (selected)"),
    ("Extra Trees, trained on both orientations", "Extra Trees"),
]
pair_rows = []
for symmetric_name, standard_name in pairs:
    a = folds[folds["model"] == symmetric_name].sort_values(["repeat", "fold"])["RMSE"].values
    b = folds[folds["model"] == standard_name].sort_values(["repeat", "fold"])["RMSE"].values
    difference, t_statistic, p_value = corrected_resampled_ttest(a, b, n_fit, n_validation)
    pair_rows.append({"symmetric_model": symmetric_name, "standard_model": standard_name,
                      "RMSE_difference": difference, "t_statistic": t_statistic, "p_value": p_value})
paired = pd.DataFrame(pair_rows)


# Save results
listing_effect.to_csv(OUTPUT_DIR / "listing_order_effect.csv", index=False)
folds.to_csv(OUTPUT_DIR / "swapped_teams_folds.csv", index=False)
summary.to_csv(OUTPUT_DIR / "swapped_teams_summary.csv", index=False)
paired.to_csv(OUTPUT_DIR / "symmetric_vs_standard.csv", index=False)


# Plot: symmetry violation and RMSE for each model
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
colours = ["tab:blue" if kind == "standard" else "tab:green" for kind in summary["type"]]
axes[0].barh(summary["model"], summary["symmetry_violation_goals"], color=colours)
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean |f(match) + f(swapped match)| (goals)")
axes[0].set_title("Symmetry Violation (0 = fair to listing order)")
axes[1].barh(summary["model"], summary["RMSE_mean"], xerr=summary["RMSE_std"] / np.sqrt(len(splits)),
             color=colours, capsize=3)
axes[1].invert_yaxis()
axes[1].set_xlim(left=1.3)
axes[1].set_xlabel("Repeated CV RMSE (goals)")
axes[1].set_title("Validation Error (green = symmetric version)")
axes[1].set_yticklabels([])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "swapped_teams_check.png", dpi=300)
plt.close()


print("\nSwapped-teams check complete")
print("\nDoes being listed first matter?")
print(listing_effect.round(4).to_string(index=False))
print("\nModels (repeated 5-fold CV on training matches):")
print(summary.round(4).to_string(index=False))
print("\nSymmetric version minus standard version (negative = symmetric is better):")
print(paired.round(4).to_string(index=False))
print("\nOutputs saved to:", OUTPUT_DIR)
