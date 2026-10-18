from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "matches_104_model.csv"
AUDIT_FILE = BASE_DIR / "data" / "processed" / "final_dataset_audit.csv"


df = pd.read_csv(INPUT_FILE)

response = "goal_diff"

predictors = [
    "fifa_points_diff",
    "market_value_log_ratio",
    "squad_age_diff",
    "rest_days_log_ratio",
    "host_diff",
    "travel_1000km_diff",
    "tournament_form_diff",
    "pre_wc_form_diff",
]

identifier_columns = ["match_id", "date", "team_a", "team_b"]


# Check the dataset structure
expected_columns = identifier_columns + [response] + predictors

assert len(df) == 104
assert df["match_id"].nunique() == 104
assert pd.concat([df["team_a"], df["team_b"]]).nunique() == 48
assert len(predictors) == 8
assert list(df.columns) == expected_columns
assert df.duplicated().sum() == 0
assert (df["team_a"] != df["team_b"]).all()
assert df[expected_columns].notna().all().all()


# Check the response variable
assert pd.api.types.is_numeric_dtype(df[response])
assert (df[response] == df[response].astype(int)).all()
assert df[response].abs().max() <= 10


# Check predictor values
for predictor in predictors:
    assert pd.api.types.is_numeric_dtype(df[predictor])

assert set(df["host_diff"].unique()).issubset({-1, 0, 1})


# Make sure helper columns are not in the modelling file
forbidden_columns = [
    "team_a_goals", "team_b_goals", "extra_time", "penalty_shootout",
    "team_a_fifa_points", "team_b_fifa_points", "elo_rating_diff", "rest_days_diff",
    "team_a_market_value_m", "team_b_market_value_m",
]
assert not any(column in df.columns for column in forbidden_columns)


# Predictor summary
audit_rows = []
for predictor in predictors:
    series = df[predictor]
    audit_rows.append({
        "variable": predictor,
        "count": series.count(),
        "missing": series.isna().sum(),
        "unique_values": series.nunique(),
        "minimum": series.min(),
        "maximum": series.max(),
        "mean": series.mean(),
        "std": series.std(),
    })

audit = pd.DataFrame(audit_rows)
audit.to_csv(AUDIT_FILE, index=False)


print("Final dataset audit passed")
print("Rows:", len(df))
print("Unique matches:", df["match_id"].nunique())
print("Explanatory variables:", len(predictors))
print("Missing predictor values:", df[predictors].isna().sum().sum())
print("Saved audit to:", AUDIT_FILE)
