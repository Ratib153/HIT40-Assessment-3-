from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "matches_104_historical.csv"
OUTPUT_FULL_FILE = BASE_DIR / "data" / "processed" / "matches_104_complete.csv"
OUTPUT_MODEL_FILE = BASE_DIR / "data" / "processed" / "matches_104_model.csv"


df = pd.read_csv(INPUT_FILE)

assert len(df) == 104
assert df["match_id"].nunique() == 104


# The complete dataset keeps every helper column (team values, Elo, etc.)
df.to_csv(OUTPUT_FULL_FILE, index=False)


# Final model dataset: 104 rows, IDs + response + exactly 8 explanatory variables.
# Every explanatory variable is team A minus team B and known before kick-off.
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

model_columns = ["match_id", "date", "team_a", "team_b", response] + predictors
model_df = df[model_columns].copy()


# Final checks
assert len(predictors) == 8
assert len(model_df) == 104
assert model_df[predictors + [response]].notna().all().all()


model_df.to_csv(OUTPUT_MODEL_FILE, index=False)

print("Rows:", len(model_df))
print("Explanatory variables:", len(predictors))
print("Missing values:", model_df[predictors].isna().sum().sum())
print("Saved final model dataset to:", OUTPUT_MODEL_FILE)
