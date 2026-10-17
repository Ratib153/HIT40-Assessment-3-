from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    / "final_model"
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

alpha_values = [
    0.001,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
]


# Load data
df = pd.read_csv(DATA_FILE)

assert len(df) == 208
assert len(predictors) == 8
assert df[predictors + [response]].notna().all().all()

X = df[predictors]
y = df[response]
groups = df["match_id"]


# Five outer folds keep both rows of a match together
outer_cv = GroupKFold(
    n_splits=5
)

metric_rows = []
lasso_prediction_rows = []
parameter_rows = []


for fold, (
    train_index,
    test_index
) in enumerate(
    outer_cv.split(
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

    train_groups = groups.iloc[
        train_index
    ]


    # Tune Lasso inside the training data
    lasso_pipeline = Pipeline([
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
    ])

    inner_cv = GroupKFold(
        n_splits=4
    )

    search = GridSearchCV(
        estimator=lasso_pipeline,
        param_grid={
            "model__alpha":
                alpha_values
        },
        scoring=
            "neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=1
    )

    search.fit(
        X_train,
        y_train,
        groups=train_groups
    )

    lasso_model = (
        search.best_estimator_
    )

    lasso_pred = (
        lasso_model.predict(
            X_test
        )
    )


    # Required multiple linear regression
    mlr_model = LinearRegression()

    mlr_model.fit(
        X_train,
        y_train
    )

    mlr_pred = mlr_model.predict(
        X_test
    )


    # Mean baseline
    baseline_model = DummyRegressor(
        strategy="mean"
    )

    baseline_model.fit(
        X_train,
        y_train
    )

    baseline_pred = (
        baseline_model.predict(
            X_test
        )
    )


    predictions = {
        "Lasso Regression":
            lasso_pred,

        "Multiple Linear Regression":
            mlr_pred,

        "Mean Baseline":
            baseline_pred,
    }


    for model_name, prediction in (
        predictions.items()
    ):

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        mse = mean_squared_error(
            y_test,
            prediction
        )

        rmse = math.sqrt(
            mse
        )

        r2 = r2_score(
            y_test,
            prediction
        )

        metric_rows.append({
            "model": model_name,
            "fold": fold,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })


    parameter_rows.append({
        "fold": fold,
        "selected_alpha":
            search.best_params_[
                "model__alpha"
            ],
    })


    fold_predictions = df.iloc[
        test_index
    ][
        [
            "match_id",
            "date",
            "team",
            "opponent",
        ]
    ].copy()

    fold_predictions["fold"] = fold

    fold_predictions[
        "actual_goals"
    ] = y_test.values

    fold_predictions[
        "predicted_goals"
    ] = lasso_pred

    fold_predictions[
        "residual"
    ] = (
        y_test.values
        - lasso_pred
    )

    lasso_prediction_rows.append(
        fold_predictions
    )


metrics = pd.DataFrame(
    metric_rows
)

lasso_predictions = pd.concat(
    lasso_prediction_rows,
    ignore_index=True
)

fold_parameters = pd.DataFrame(
    parameter_rows
)


# Cross-validation summary
summary = (
    metrics
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


# Tune the final Lasso model on all observations
final_cv = GroupKFold(
    n_splits=5
)

final_pipeline = Pipeline([
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
])

final_search = GridSearchCV(
    estimator=final_pipeline,
    param_grid={
        "model__alpha":
            alpha_values
    },
    scoring=
        "neg_root_mean_squared_error",
    cv=final_cv,
    n_jobs=1
)

final_search.fit(
    X,
    y,
    groups=groups
)

final_model = (
    final_search.best_estimator_
)

final_alpha = (
    final_search.best_params_[
        "model__alpha"
    ]
)


# Extract final Lasso coefficients
scaler = final_model.named_steps[
    "scaler"
]

lasso = final_model.named_steps[
    "model"
]

standardised_coefficients = (
    lasso.coef_
)

original_coefficients = (
    standardised_coefficients
    / scaler.scale_
)

original_intercept = (
    lasso.intercept_
    - np.sum(
        standardised_coefficients
        * scaler.mean_
        / scaler.scale_
    )
)

coefficient_table = pd.DataFrame({
    "predictor": predictors,
    "standardised_coefficient":
        standardised_coefficients,
    "original_scale_coefficient":
        original_coefficients,
})

coefficient_table[
    "absolute_standardised_coefficient"
] = (
    coefficient_table[
        "standardised_coefficient"
    ].abs()
)

coefficient_table[
    "retained_by_lasso"
] = (
    coefficient_table[
        "absolute_standardised_coefficient"
    ] > 1e-12
)

coefficient_table = (
    coefficient_table
    .sort_values(
        "absolute_standardised_coefficient",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# Save outputs
metrics.to_csv(
    OUTPUT_DIR
    / "final_model_folds.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR
    / "final_model_comparison.csv",
    index=False
)

lasso_predictions.to_csv(
    OUTPUT_DIR
    / "lasso_out_of_fold_predictions.csv",
    index=False
)

fold_parameters.to_csv(
    OUTPUT_DIR
    / "lasso_fold_parameters.csv",
    index=False
)

coefficient_table.to_csv(
    OUTPUT_DIR
    / "final_lasso_coefficients.csv",
    index=False
)


final_details = pd.DataFrame([
    {
        "final_model":
            "Lasso Regression",
        "alpha":
            final_alpha,
        "intercept_original_scale":
            original_intercept,
        "starting_predictors":
            len(predictors),
        "retained_predictors":
            int(
                coefficient_table[
                    "retained_by_lasso"
                ].sum()
            ),
    }
])

final_details.to_csv(
    OUTPUT_DIR
    / "final_model_details.csv",
    index=False
)


# Actual vs predicted
plt.figure(
    figsize=(6, 5)
)

plt.scatter(
    lasso_predictions[
        "actual_goals"
    ],
    lasso_predictions[
        "predicted_goals"
    ],
    alpha=0.7
)

minimum = min(
    lasso_predictions[
        "actual_goals"
    ].min(),
    lasso_predictions[
        "predicted_goals"
    ].min()
)

maximum = max(
    lasso_predictions[
        "actual_goals"
    ].max(),
    lasso_predictions[
        "predicted_goals"
    ].max()
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
    "Final Lasso Model: Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# Residual plot
plt.figure(
    figsize=(6, 5)
)

plt.scatter(
    lasso_predictions[
        "predicted_goals"
    ],
    lasso_predictions[
        "residual"
    ],
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
    "Final Lasso Model Residuals"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_residual_plot.png",
    dpi=300
)

plt.close()


# Coefficient plot
plot_data = (
    coefficient_table
    .sort_values(
        "standardised_coefficient"
    )
)

plt.figure(
    figsize=(8, 5)
)

plt.barh(
    plot_data[
        "predictor"
    ],
    plot_data[
        "standardised_coefficient"
    ]
)

plt.axvline(
    x=0
)

plt.xlabel(
    "Standardised Lasso Coefficient"
)

plt.title(
    "Final Lasso Predictor Coefficients"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_lasso_coefficients.png",
    dpi=300
)

plt.close()


# Print final results
print(
    "Final model evaluation complete"
)

print(
    "\nCross-validation comparison:"
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
    "\nLasso alpha selected in each outer fold:"
)

print(
    fold_parameters.to_string(
        index=False
    )
)

print(
    "\nFinal alpha using full dataset:",
    final_alpha
)

print(
    "\nFinal Lasso coefficients:"
)

print(
    coefficient_table[
        [
            "predictor",
            "standardised_coefficient",
            "retained_by_lasso",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nNumber of predictors retained:",
    int(
        coefficient_table[
            "retained_by_lasso"
        ].sum()
    )
)

print(
    "\nOutputs saved to:",
    OUTPUT_DIR
)