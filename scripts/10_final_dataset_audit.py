from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_model.csv"
)

AUDIT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_dataset_audit.csv"
)


# Load final modelling dataset
df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="raise"
)


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

identifier_columns = [
    "match_id",
    "date",
    "team",
    "opponent",
]


# Check the dataset structure
expected_columns = (
    identifier_columns
    + [response]
    + predictors
)

assert len(df) == 208
assert df["match_id"].nunique() == 104
assert df["team"].nunique() == 48
assert len(predictors) == 8
assert list(df.columns) == expected_columns
assert df.duplicated().sum() == 0

assert (
    df.groupby("match_id").size() == 2
).all()

assert (
    df["team"] != df["opponent"]
).all()

assert df[
    expected_columns
].notna().all().all()


# Check the response variable
assert pd.api.types.is_numeric_dtype(
    df[response]
)

assert (
    df["goals_scored"] >= 0
).all()

assert (
    df["goals_scored"]
    == df["goals_scored"].astype(int)
).all()

assert df[
    "goals_scored"
].sum() == 308


# Check predictor values
for predictor in predictors:
    assert pd.api.types.is_numeric_dtype(
        df[predictor]
    )

assert (
    df["rest_days"] > 0
).all()

assert (
    df["venue_altitude_m"] >= 0
).all()

assert (
    df["attacking_form_5"] >= 0
).all()

assert (
    df["opponent_defensive_form_5"] >= 0
).all()

assert set(
    df["previous_extra_time"].unique()
).issubset({0, 1})

assert (
    df["lineup_changes"]
    .between(0, 11)
    .all()
)

assert set(
    df["qualification_pressure"].unique()
).issubset({0, 1})


# Check both rows from each match agree
for match_id, match in df.groupby(
    "match_id"
):

    first = match.iloc[0]
    second = match.iloc[1]

    assert (
        first["team"]
        == second["opponent"]
    )

    assert (
        second["team"]
        == first["opponent"]
    )

    assert (
        first["date"]
        == second["date"]
    )

    assert (
        first["venue_altitude_m"]
        == second["venue_altitude_m"]
    )

    assert abs(
        first["fifa_points_diff"]
        + second["fifa_points_diff"]
    ) < 0.01


# Make sure helper columns are not in the modelling file
forbidden_columns = [
    "match_extra_time",
    "match_penalty_shootout",
    "team_fifa_rank",
    "team_fifa_points",
    "opponent_fifa_rank",
    "opponent_fifa_points",
    "retained_starters",
    "group_match_number",
    "is_group_stage",
]

assert not any(
    column in df.columns
    for column in forbidden_columns
)


# Create a simple predictor summary
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


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    AUDIT_FILE,
    index=False
)


print("Final dataset audit passed")
print("Rows:", len(df))
print(
    "Unique matches:",
    df["match_id"].nunique()
)
print(
    "Unique teams:",
    df["team"].nunique()
)
print(
    "Starting predictors:",
    len(predictors)
)
print(
    "Missing predictor values:",
    df[predictors]
    .isna()
    .sum()
    .sum()
)
print(
    "Saved audit to:",
    AUDIT_FILE
)