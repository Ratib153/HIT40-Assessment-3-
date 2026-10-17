from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_lineup.csv"
)

OUTPUT_FULL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_complete.csv"
)

OUTPUT_MODEL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_model.csv"
)

AUDIT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "qualification_pressure_audit.csv"
)


# Load dataset
df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="raise"
)


# Basic checks
assert len(df) == 208
assert df["match_id"].nunique() == 104
assert df["team"].nunique() == 48


# Identify group-stage matches
round_text = (
    df["round"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["is_group_stage"] = (
    round_text.str.contains(
        "group",
        na=False
    )
)


# Number each team's group-stage matches
df = df.sort_values(
    [
        "team",
        "date",
        "match_id"
    ]
).reset_index(drop=True)

df["group_match_number"] = 0

for team in df["team"].unique():

    team_group_mask = (
        (df["team"] == team)
        & (df["is_group_stage"])
    )

    team_group_indices = (
        df.loc[team_group_mask]
        .sort_values("date")
        .index
        .tolist()
    )

    if len(team_group_indices) != 3:
        raise ValueError(
            f"{team} does not have exactly "
            f"three group-stage matches."
        )

    for number, index in enumerate(
        team_group_indices,
        start=1
    ):
        df.loc[
            index,
            "group_match_number"
        ] = number


# Qualification pressure:
# 1 = third group match or knockout match
# 0 = first or second group match
df["qualification_pressure"] = 0

third_group_match = (
    (df["is_group_stage"])
    & (df["group_match_number"] == 3)
)

knockout_match = (
    ~df["is_group_stage"]
)

df.loc[
    third_group_match | knockout_match,
    "qualification_pressure"
] = 1

df["qualification_pressure"] = (
    df["qualification_pressure"]
    .astype(int)
)


# Check the pressure variable
assert set(
    df["qualification_pressure"].unique()
).issubset({0, 1})

group_rows = df[
    df["is_group_stage"]
]

assert len(group_rows) == 144

assert (
    group_rows["group_match_number"]
    .between(1, 3)
    .all()
)

assert (
    df["qualification_pressure"]
    == 0
).sum() == 96

assert (
    df["qualification_pressure"]
    == 1
).sum() == 112


# Save a separate audit file
audit = df[
    [
        "match_id",
        "date",
        "round",
        "team",
        "opponent",
        "is_group_stage",
        "group_match_number",
        "qualification_pressure",
    ]
].copy()


# Save the complete enriched dataset
df.to_csv(
    OUTPUT_FULL_FILE,
    index=False
)


# Final model dataset
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

model_columns = [
    "match_id",
    "date",
    "team",
    "opponent",
    "goals_scored",
] + predictors

model_df = df[
    model_columns
].copy()


# Final checks
assert len(predictors) == 8
assert len(model_df) == 208

assert model_df[
    predictors
].notna().all().all()

assert model_df[
    "goals_scored"
].notna().all()

assert model_df[
    "match_id"
].nunique() == 104

assert (
    model_df
    .groupby("match_id")
    .size()
    == 2
).all()


# Save final files
model_df.to_csv(
    OUTPUT_MODEL_FILE,
    index=False
)

audit.to_csv(
    AUDIT_FILE,
    index=False
)


print(
    "Rows:",
    len(model_df)
)

print(
    "Unique matches:",
    model_df["match_id"].nunique()
)

print(
    "Starting predictors:",
    len(predictors)
)

print(
    "Missing predictor values:",
    model_df[predictors]
    .isna()
    .sum()
    .sum()
)

print(
    "Saved final model dataset to:",
    OUTPUT_MODEL_FILE
)