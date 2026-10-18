"""Shared settings for the Task 2.1 modelling scripts.

Every modelling script imports this file so that all of them use the same
variables, the same train/test split and the same cross-validation folds.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "matches_104_model.csv"
COMPLETE_FILE = BASE_DIR / "data" / "processed" / "matches_104_complete.csv"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


# Variables
RESPONSE = "goal_diff"

PREDICTORS = [
    "fifa_points_diff",
    "market_value_log_ratio",
    "squad_age_diff",
    "rest_days_log_ratio",
    "host_diff",
    "travel_1000km_diff",
    "tournament_form_diff",
    "pre_wc_form_diff",
]


# Validation design
# 1. A test set of 21 matches (20%) is held out once and not used until
#    the final evaluation in 05_final_model.py.
# 2. All model comparison, tuning and variable checks use repeated
#    k-fold cross-validation on the 83 training matches only
#    (each validation fold plays the role of a validation set).
RANDOM_STATE = 0
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 10


def output_dir(name):
    folder = OUTPUT_ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def load_data():
    df = pd.read_csv(DATA_FILE)
    assert len(df) == 104
    assert len(PREDICTORS) == 8
    assert df[PREDICTORS + [RESPONSE]].notna().all().all()
    return df


def load_complete():
    # All engineered columns, including the alternatives (Elo, raw rest days)
    complete = pd.read_csv(COMPLETE_FILE)
    assert len(complete) == 104
    return complete


def outcome_band(goal_diff):
    # Goal difference grouped into five bands (-2 or less, -1, 0, 1, 2 or more)
    # so the test set has a similar spread of results to the training set
    return np.clip(goal_diff, -2, 2)


def holdout_split(df):
    """Return (train_index, test_index) for the single held-out test set."""
    train_index, test_index = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=outcome_band(df[RESPONSE]),
    )
    return np.sort(train_index), np.sort(test_index)


def cv_splitter(repeats=CV_REPEATS):
    return RepeatedKFold(n_splits=CV_FOLDS, n_repeats=repeats, random_state=RANDOM_STATE)


def metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }


def summarise_folds(fold_results, group="model"):
    """Mean, standard deviation and a 95% interval of the fold scores."""
    summary = (
        fold_results.groupby(group, as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"), R2_std=("R2", "std"),
            folds=("RMSE", "size"),
        )
        .sort_values("RMSE_mean")
        .reset_index(drop=True)
    )
    summary["RMSE_se"] = summary["RMSE_std"] / np.sqrt(summary["folds"])
    return summary


def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    """Nadeau and Bengio (2003) corrected repeated k-fold t-test.

    Cross-validation folds share training data, so an ordinary paired t-test
    is too optimistic. The variance is inflated by (1/k + n_test/n_train).
    Returns the mean difference (a - b), the t statistic and the p-value.
    """
    from scipy import stats

    differences = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(differences)
    mean_difference = differences.mean()
    variance = differences.var(ddof=1)
    if variance == 0:
        return mean_difference, 0.0, 1.0
    corrected_variance = (1 / k + n_test / n_train) * variance
    t_statistic = mean_difference / np.sqrt(corrected_variance)
    p_value = 2 * stats.t.sf(abs(t_statistic), df=k - 1)
    return mean_difference, t_statistic, p_value


def scaled(model):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def tuning_candidates():
    """Candidate models, parameter grids and a complexity rank (1 = simplest).

    The complexity rank is used for the one-standard-error rule in
    04_model_optimisation.py.
    """
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    return {
        "Multiple Linear Regression": (LinearRegression(), {}, 1),
        "Ridge Regression": (scaled(Ridge()), {"model__alpha": list(np.logspace(-2, 3, 11))}, 2),
        "Lasso Regression": (scaled(Lasso(max_iter=20000)),
                             {"model__alpha": [0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]}, 2),
        "Elastic Net": (scaled(ElasticNet(max_iter=20000)),
                        {"model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}, 2),
        "Huber Regression": (scaled(HuberRegressor(max_iter=2000)),
                             {"model__epsilon": [1.1, 1.35, 1.75, 2.5], "model__alpha": [0.0001, 0.1, 1, 10, 50]}, 2),
        "Polynomial (degree 2) + Ridge": (
            Pipeline([("scaler", StandardScaler()), ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                      ("model", Ridge())]),
            {"model__alpha": [1, 10, 100, 1000, 10000]}, 3),
        "K-Nearest Neighbours": (scaled(KNeighborsRegressor()),
                                 {"model__n_neighbors": [3, 5, 7, 10, 15, 20], "model__weights": ["uniform", "distance"]}, 3),
        "Support Vector Regression (RBF)": (scaled(SVR()),
                                            {"model__C": [0.3, 1, 3, 10], "model__gamma": ["scale", 0.01, 0.05],
                                             "model__epsilon": [0.1, 0.5]}, 3),
        "Decision Tree": (DecisionTreeRegressor(random_state=RANDOM_STATE),
                          {"max_depth": [1, 2, 3, 4], "min_samples_leaf": [1, 5, 10]}, 3),
        "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
                          {"max_depth": [None, 3], "min_samples_leaf": [1, 5], "max_features": [1.0, 0.5]}, 4),
        "Extra Trees": (ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE),
                        {"max_depth": [None, 3], "min_samples_leaf": [1, 5], "max_features": [1.0, 0.5]}, 4),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=RANDOM_STATE),
                              {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [1, 2]}, 4),
    }
