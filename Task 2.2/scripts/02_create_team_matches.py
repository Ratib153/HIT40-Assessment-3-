from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "matches_104_clean.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_base.csv"
)


# Load cleaned match data
matches = pd.read_csv(INPUT_FILE)
matches["date"] = pd.to_datetime(matches["date"])


# Create one row for Team A in each match
team_a_rows = pd.DataFrame({
    "match_id": matches["match_id"],
    "date": matches["date"],
    "round": matches["round"],
    "team": matches["team_a"],
    "opponent": matches["team_b"],
    "goals_scored": matches["team_a_goals"],
    "venue": matches["venue"],
    "match_extra_time": matches["extra_time"],
    "match_penalty_shootout": matches["penalty_shootout"]
})


# Create one row for Team B in each match
team_b_rows = pd.DataFrame({
    "match_id": matches["match_id"],
    "date": matches["date"],
    "round": matches["round"],
    "team": matches["team_b"],
    "opponent": matches["team_a"],
    "goals_scored": matches["team_b_goals"],
    "venue": matches["venue"],
    "match_extra_time": matches["extra_time"],
    "match_penalty_shootout": matches["penalty_shootout"]
})


# Combine both teams into one dataset
team_matches = pd.concat(
    [team_a_rows, team_b_rows],
    ignore_index=True
)

team_matches = team_matches.sort_values(
    by=["date", "match_id", "team"]
).reset_index(drop=True)


# Validation checks
assert len(team_matches) == 208

assert team_matches["match_id"].nunique() == 104

assert (
    team_matches.groupby("match_id").size() == 2
).all()

assert team_matches["goals_scored"].sum() == 308

assert team_matches[
    [
        "match_id",
        "date",
        "team",
        "opponent",
        "goals_scored",
        "venue"
    ]
].notna().all().all()

assert (
    team_matches["team"] != team_matches["opponent"]
).all()


# Save dataset
team_matches.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Rows:", len(team_matches))
print(
    "Unique matches:",
    team_matches["match_id"].nunique()
)
print(
    "Unique teams:",
    team_matches["team"].nunique()
)
print(
    "Total goals:",
    team_matches["goals_scored"].sum()
)
print("Saved to:", OUTPUT_FILE)