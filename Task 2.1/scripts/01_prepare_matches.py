from pathlib import Path
import re
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "HIT40 dataset.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "matches_104_clean.csv"


# Load raw dataset
df = pd.read_csv(INPUT_FILE)

# Remove completely empty separator rows
df = df.dropna(how="all").copy()

# Keep only the columns needed for the base match dataset
df = df[["Round", "Date", "Home", "Score", "Away", "Venue", "Notes"]].copy()


# Remove FBref country codes from team names
# ("Mexico mx" -> "Mexico", "za South Africa" -> "South Africa")
def clean_home_team(name):
    return re.sub(r"\s+[a-z]{2,3}$", "", str(name)).strip()


def clean_away_team(name):
    return re.sub(r"^[a-z]{2,3}\s+", "", str(name)).strip()


df["team_a"] = df["Home"].apply(clean_home_team)
df["team_b"] = df["Away"].apply(clean_away_team)


# Extract the actual match score
# "(4) 0–0 (3)" becomes 0 and 0, so penalty shoot-out kicks are not counted.
# Extra-time goals are counted because they are part of the final score.
score_parts = df["Score"].str.extract(r"(\d+)\s*[–-]\s*(\d+)")
df["team_a_goals"] = pd.to_numeric(score_parts[0], errors="raise")
df["team_b_goals"] = pd.to_numeric(score_parts[1], errors="raise")

# Response variable for Task 2.1
df["goal_diff"] = df["team_a_goals"] - df["team_b_goals"]


# Extra-time and penalty-shootout indicators
notes = df["Notes"].fillna("")
df["extra_time"] = notes.str.contains("extra time", case=False).astype(int)
df["penalty_shootout"] = notes.str.contains(r"penalty shoot[- ]out", case=False, regex=True).astype(int)


# Clean venue names and dates
df["venue"] = df["Venue"].str.replace(r"\s*\(Neutral Site\)\s*$", "", regex=True).str.strip()
df["date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")


# Match IDs in the same format as Task 2.2 (M001 ... M104)
df = df.reset_index(drop=True)
df["match_id"] = [f"M{i:03d}" for i in range(1, len(df) + 1)]


# Build final clean dataset
clean_df = df[
    [
        "match_id", "date", "Round", "team_a", "team_b",
        "team_a_goals", "team_b_goals", "goal_diff",
        "venue", "extra_time", "penalty_shootout",
    ]
].rename(columns={"Round": "round"})


# Validation checks
assert len(clean_df) == 104, f"Expected 104 matches, found {len(clean_df)}"
assert clean_df["match_id"].nunique() == 104
assert clean_df.notna().all().all(), "Missing values found."
assert (clean_df["team_a"] != clean_df["team_b"]).all()
assert clean_df["team_a_goals"].sum() + clean_df["team_b_goals"].sum() == 308

all_teams = pd.concat([clean_df["team_a"], clean_df["team_b"]])
assert all_teams.nunique() == 48


print("Rows:", len(clean_df))
print("Unique teams:", all_teams.nunique())
print("Total goals:", clean_df["team_a_goals"].sum() + clean_df["team_b_goals"].sum())
print("Penalty shoot-outs (goal_diff = 0):", clean_df["penalty_shootout"].sum())

OUTPUT_DIR.mkdir(exist_ok=True)
clean_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved cleaned dataset to:\n{OUTPUT_FILE}")
