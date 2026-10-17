from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_model.csv"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "mlr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Variables
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


# Load data
df = pd.read_csv(DATA_FILE)

assert len(df) == 208
assert len(predictors) == 8
assert df[predictors + [response]].notna().all().all()


X = df[predictors]
y = df[response]
groups = df["match_id"]


# Split by match so both rows stay together
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.4,
    random_state=0
)

train_index, test_index = next(
    splitter.split(
        X,
        y,
        groups
    )
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]


# Multiple linear regression
model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)


# Model performance
mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = math.sqrt(mse)

response_range = (
    y.max() - y.min()
)

rmse_normalised = (
    rmse / response_range
)

r2 = r2_score(
    y_test,
    y_pred
)


# Mean baseline
baseline_value = y_train.mean()

baseline_pred = np.full(
    len(y_test),
    baseline_value
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_mse = mean_squared_error(
    y_test,
    baseline_pred
)

baseline_rmse = math.sqrt(
    baseline_mse
)

baseline_rmse_normalised = (
    baseline_rmse
    / response_range
)

baseline_r2 = r2_score(
    y_test,
    baseline_pred
)


# Save model coefficients
coefficients = pd.DataFrame({
    "predictor": predictors,
    "coefficient": model.coef_,
})

coefficients.to_csv(
    OUTPUT_DIR
    / "mlr_coefficients.csv",
    index=False
)


# Save test predictions
predictions = df.iloc[
    test_index
][
    [
        "match_id",
        "date",
        "team",
        "opponent",
    ]
].copy()

predictions["actual_goals"] = (
    y_test.values
)

predictions["predicted_goals"] = (
    y_pred
)

predictions["residual"] = (
    predictions["actual_goals"]
    - predictions["predicted_goals"]
)

predictions.to_csv(
    OUTPUT_DIR
    / "mlr_test_predictions.csv",
    index=False
)


# Save performance comparison
performance = pd.DataFrame([
    {
        "model": "Multiple Linear Regression",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Normalised_RMSE":
            rmse_normalised,
        "R2": r2,
    },
    {
        "model": "Mean Baseline",
        "MAE": baseline_mae,
        "MSE": baseline_mse,
        "RMSE": baseline_rmse,
        "Normalised_RMSE":
            baseline_rmse_normalised,
        "R2": baseline_r2,
    },
])

performance.to_csv(
    OUTPUT_DIR
    / "mlr_performance.csv",
    index=False
)


# Statsmodels model using all observations
X_stats = sm.add_constant(
    X
)

ols_model = sm.OLS(
    y,
    X_stats
).fit()

with open(
    OUTPUT_DIR
    / "mlr_statsmodels_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        ols_model.summary().as_text()
    )


# Save coefficient statistics
ols_results = pd.DataFrame({
    "variable":
        ols_model.params.index,

    "coefficient":
        ols_model.params.values,

    "p_value":
        ols_model.pvalues.values,

    "confidence_interval_low":
        ols_model.conf_int()[0].values,

    "confidence_interval_high":
        ols_model.conf_int()[1].values,
})

ols_results.to_csv(
    OUTPUT_DIR
    / "mlr_ols_results.csv",
    index=False
)


# Actual vs predicted plot
plt.figure(
    figsize=(6, 5)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel(
    "Actual Goals"
)

plt.ylabel(
    "Predicted Goals"
)

plt.title(
    "Multiple Linear Regression: Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# Residual plot
residuals = (
    y_test.values
    - y_pred
)

plt.figure(
    figsize=(6, 5)
)

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0
)

plt.xlabel(
    "Predicted Goals"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residuals vs Predicted Goals"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "residual_plot.png",
    dpi=300
)

plt.close()


# Print main results
print("Multiple linear regression complete")

print(
    "Training rows:",
    len(X_train)
)

print(
    "Test rows:",
    len(X_test)
)

print(
    "Training matches:",
    df.iloc[
        train_index
    ]["match_id"].nunique()
)

print(
    "Test matches:",
    df.iloc[
        test_index
    ]["match_id"].nunique()
)

print(
    "\nMLR performance"
)

print(
    "MAE:",
    round(mae, 4)
)

print(
    "MSE:",
    round(mse, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "Normalised RMSE:",
    round(
        rmse_normalised,
        4
    )
)

print(
    "R2:",
    round(r2, 4)
)

print(
    "\nBaseline performance"
)

print(
    "MAE:",
    round(
        baseline_mae,
        4
    )
)

print(
    "RMSE:",
    round(
        baseline_rmse,
        4
    )
)

print(
    "R2:",
    round(
        baseline_r2,
        4
    )
)

print(
    "\nIntercept:",
    round(
        model.intercept_,
        4
    )
)

print(
    "\nCoefficients:"
)

print(
    coefficients.to_string(
        index=False
    )
)

print(
    "\nOutputs saved to:",
    OUTPUT_DIR
)
