from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    Ridge,
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
    / "model_optimisation"
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


# Models selected for optimisation
models = {
    "Ridge Regression": (
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
        {
            "model__alpha": [
                0.01,
                0.1,
                0.5,
                1,
                2,
                5,
                10,
                25,
            ]
        }
    ),

    "Lasso Regression": (
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
        {
            "model__alpha": [
                0.001,
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
            ]
        }
    ),

    "Random Forest": (
        RandomForestRegressor(
            random_state=0
        ),
        {
            "n_estimators": [
                100,
                200,
            ],
            "max_depth": [
                None,
                3,
                5,
            ],
            "min_samples_leaf": [
                1,
                2,
                4,
            ],
        }
    ),

    "Gradient Boosting": (
        GradientBoostingRegressor(
            random_state=0
        ),
        {
            "n_estimators": [
                50,
                100,
            ],
            "learning_rate": [
                0.03,
                0.05,
                0.1,
            ],
            "max_depth": [
                1,
                2,
            ],
        }
    ),
}


# Nested grouped cross-validation
outer_cv = GroupKFold(
    n_splits=5
)

result_rows = []
parameter_rows = []


for model_name, (
    model,
    parameter_grid
) in models.items():

    print(
        "\nOptimising:",
        model_name
    )

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

        print(
            " Fold:",
            fold
        )

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

        inner_cv = GroupKFold(
            n_splits=4
        )

        search = GridSearchCV(
            estimator=clone(model),
            param_grid=parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=1
        )

        search.fit(
            X_train,
            y_train,
            groups=train_groups
        )

        best_model = (
            search.best_estimator_
        )

        y_pred = best_model.predict(
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

        result_rows.append({
            "model": model_name,
            "fold": fold,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })

        parameter_rows.append({
            "model": model_name,
            "fold": fold,
            "best_parameters":
                str(
                    search.best_params_
                ),
        })


results = pd.DataFrame(
    result_rows
)

parameters = pd.DataFrame(
    parameter_rows
)


# Summarise nested cross-validation
summary = (
    results
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


# Tune models on the full dataset
final_parameter_rows = []

full_cv = GroupKFold(
    n_splits=5
)

for model_name, (
    model,
    parameter_grid
) in models.items():

    print(
        "\nFinal tuning:",
        model_name
    )

    search = GridSearchCV(
        estimator=clone(model),
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=full_cv,
        n_jobs=1
    )

    search.fit(
        X,
        y,
        groups=groups
    )

    final_parameter_rows.append({
        "model": model_name,
        "best_parameters":
            str(
                search.best_params_
            ),
        "best_cv_rmse":
            -search.best_score_,
    })


final_parameters = pd.DataFrame(
    final_parameter_rows
)

final_parameters = (
    final_parameters
    .sort_values(
        "best_cv_rmse"
    )
    .reset_index(
        drop=True
    )
)


# Save results
results.to_csv(
    OUTPUT_DIR
    / "nested_cv_folds.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR
    / "optimised_model_summary.csv",
    index=False
)

parameters.to_csv(
    OUTPUT_DIR
    / "fold_best_parameters.csv",
    index=False
)

final_parameters.to_csv(
    OUTPUT_DIR
    / "final_best_parameters.csv",
    index=False
)


# Plot model performance
plt.figure(
    figsize=(8, 5)
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
    "Optimised Model Performance"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "optimised_model_rmse.png",
    dpi=300
)

plt.close()


print(
    "\nModel optimisation complete"
)

print(
    "\nNested cross-validation results:"
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
    "\nBest parameters using full dataset:"
)

print(
    final_parameters.to_string(
        index=False
    )
)

print(
    "\nBest optimised model:"
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