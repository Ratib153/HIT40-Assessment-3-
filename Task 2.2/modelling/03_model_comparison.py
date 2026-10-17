from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


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
    / "model_comparison"
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


# Models for the initial comparison
models = {
    "Mean Baseline":
        DummyRegressor(
            strategy="mean"
        ),

    "Multiple Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge()
            ),
        ]),

    "Lasso Regression":
        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Lasso(
                    max_iter=10000
                )
            ),
        ]),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=0
        ),

    "Random Forest":
        RandomForestRegressor(
            random_state=0
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=0
        ),
}


# Five-fold cross-validation by match
cv = GroupKFold(
    n_splits=5
)

fold_rows = []
prediction_rows = []


for model_name, model in models.items():

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]

        fitted_model = clone(
            model
        )

        fitted_model.fit(
            X_train,
            y_train
        )

        y_pred = fitted_model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = math.sqrt(
            mse
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        fold_rows.append({
            "model": model_name,
            "fold": fold,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })


        fold_data = df.iloc[
            test_index
        ][
            [
                "match_id",
                "date",
                "team",
                "opponent",
            ]
        ].copy()

        fold_data["model"] = (
            model_name
        )

        fold_data["fold"] = fold

        fold_data["actual_goals"] = (
            y_test.values
        )

        fold_data["predicted_goals"] = (
            y_pred
        )

        prediction_rows.append(
            fold_data
        )


# Combine fold results
fold_results = pd.DataFrame(
    fold_rows
)

predictions = pd.concat(
    prediction_rows,
    ignore_index=True
)


# Summarise cross-validation results
summary = (
    fold_results
    .groupby(
        "model",
        as_index=False
    )
    .agg(
        MAE_mean=(
            "MAE",
            "mean"
        ),
        MAE_std=(
            "MAE",
            "std"
        ),
        RMSE_mean=(
            "RMSE",
            "mean"
        ),
        RMSE_std=(
            "RMSE",
            "std"
        ),
        R2_mean=(
            "R2",
            "mean"
        ),
        R2_std=(
            "R2",
            "std"
        ),
    )
)


summary = summary.sort_values(
    "RMSE_mean"
).reset_index(
    drop=True
)


# Calculate overall out-of-fold performance
overall_rows = []

for model_name in models:

    model_predictions = predictions[
        predictions["model"]
        == model_name
    ]

    actual = (
        model_predictions[
            "actual_goals"
        ]
    )

    predicted = (
        model_predictions[
            "predicted_goals"
        ]
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = math.sqrt(
        mse
    )

    r2 = r2_score(
        actual,
        predicted
    )

    overall_rows.append({
        "model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    })


overall = pd.DataFrame(
    overall_rows
)

overall = overall.sort_values(
    "RMSE"
).reset_index(
    drop=True
)


# Save results
fold_results.to_csv(
    OUTPUT_DIR
    / "cross_validation_folds.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR
    / "cross_validation_summary.csv",
    index=False
)

overall.to_csv(
    OUTPUT_DIR
    / "overall_model_comparison.csv",
    index=False
)

predictions.to_csv(
    OUTPUT_DIR
    / "cross_validation_predictions.csv",
    index=False
)


# RMSE comparison chart
plt.figure(
    figsize=(9, 5)
)

plt.bar(
    summary["model"],
    summary["RMSE_mean"]
)

plt.errorbar(
    summary["model"],
    summary["RMSE_mean"],
    yerr=summary["RMSE_std"],
    fmt="none",
    capsize=4
)

plt.ylabel(
    "Mean RMSE"
)

plt.xlabel(
    "Regression Model"
)

plt.title(
    "Five-Fold Cross-Validation Model Comparison"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "model_rmse_comparison.png",
    dpi=300
)

plt.close()


# Print results
print(
    "Model comparison complete"
)

print(
    "\nFive-fold cross-validation results:"
)

print(
    summary[
        [
            "model",
            "MAE_mean",
            "RMSE_mean",
            "R2_mean",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nOverall out-of-fold performance:"
)

print(
    overall.to_string(
        index=False
    )
)

print(
    "\nBest model by mean CV RMSE:"
)

print(
    summary.iloc[0][
        "model"
    ]
)

print(
    "\nOutputs saved to:",
    OUTPUT_DIR
)