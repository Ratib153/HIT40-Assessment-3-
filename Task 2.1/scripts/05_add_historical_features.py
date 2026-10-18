from pathlib import Path
import numpy as np
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
MATCHES_FILE = BASE_DIR / "data" / "processed" / "matches_104_travel.csv"
HISTORY_FILE = BASE_DIR / "data" / "raw" / "international_results.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "matches_104_historical.csv"
AUDIT_FILE = BASE_DIR / "data" / "processed" / "historical_features_audit.csv"

# The World Cup started on 11 June 2026. Pre-World-Cup form uses 2026
# internationals played before this date.
WORLD_CUP_START = pd.Timestamp("2026-06-11")
FORM_WINDOW_START = pd.Timestamp("2026-01-01")


matches = pd.read_csv(MATCHES_FILE)
history = pd.read_csv(HISTORY_FILE)
matches["date"] = pd.to_datetime(matches["date"], errors="raise")
history["date"] = pd.to_datetime(history["date"], errors="raise")


# Team names that differ between the two sources (same map as Task 2.2)
name_map = {
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}


# Convert historical matches to one row per team (same method as Task 2.2)
home_rows = history[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].rename(columns={
    "home_team": "team", "away_team": "opponent", "home_score": "goals_scored", "away_score": "goals_conceded",
})
away_rows = history[["date", "away_team", "home_team", "away_score", "home_score", "tournament"]].rename(columns={
    "away_team": "team", "home_team": "opponent", "away_score": "goals_scored", "home_score": "goals_conceded",
})
team_history = pd.concat([home_rows, away_rows], ignore_index=True)
team_history["goal_diff"] = team_history["goals_scored"] - team_history["goals_conceded"]
team_history = team_history.sort_values(["team", "date"]).reset_index(drop=True)


# Check every World Cup team exists in the history data
required = {name_map.get(t, t) for t in set(matches["team_a"]) | set(matches["team_b"])}
missing = required - set(team_history["team"])
if missing:
    raise ValueError("Teams missing from historical data: " + ", ".join(sorted(missing)))


# Build features for each side of each match
audit_rows = []
for side in ["team_a", "team_b"]:
    rest_days_values = []
    pre_wc_form_values = []

    for row in matches.itertuples():
        team = getattr(row, side)
        team_previous = team_history[
            (team_history["team"] == name_map.get(team, team))
            & (team_history["date"] < row.date)
        ]

        # Rest days: days since the team's previous senior international match
        previous_match = team_previous.iloc[-1]
        rest_days = (row.date - previous_match["date"]).days

        # Pre-World-Cup form: average goal difference in 2026 internationals
        # played before the tournament started
        pre_wc = team_previous[
            (team_previous["date"] >= FORM_WINDOW_START)
            & (team_previous["date"] < WORLD_CUP_START)
        ]
        if len(pre_wc) == 0:
            raise ValueError(f"{team} has no 2026 matches before the World Cup.")

        rest_days_values.append(rest_days)
        pre_wc_form_values.append(pre_wc["goal_diff"].mean())

        audit_rows.append({
            "match_id": row.match_id,
            "date": row.date.date(),
            "team": team,
            "previous_match_date": previous_match["date"].date(),
            "previous_opponent": previous_match["opponent"],
            "previous_tournament": previous_match["tournament"],
            "rest_days": rest_days,
            "pre_wc_matches": len(pre_wc),
            "pre_wc_form": pre_wc["goal_diff"].mean(),
        })

    matches[f"{side}_rest_days"] = rest_days_values
    matches[f"{side}_pre_wc_form"] = pre_wc_form_values


# Tournament form: a team's average goal difference in its EARLIER World Cup
# matches (0 before its first match). Only matches already played are used,
# so no information from the current match leaks in.
matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
results_so_far = {}
form_a, form_b = [], []

for row in matches.itertuples():
    for team, store in [(row.team_a, form_a), (row.team_b, form_b)]:
        past = results_so_far.get(team, [])
        store.append(sum(past) / len(past) if past else 0.0)

    results_so_far.setdefault(row.team_a, []).append(row.goal_diff)
    results_so_far.setdefault(row.team_b, []).append(-row.goal_diff)

matches["team_a_tournament_form"] = form_a
matches["team_b_tournament_form"] = form_b
matches = matches.sort_values("match_id").reset_index(drop=True)


# Differences (team A minus team B)
matches["rest_days_diff"] = matches["team_a_rest_days"] - matches["team_b_rest_days"]
# Log ratio of rest days (week 9 log transform). Uruguay rested 76 days before
# its first match while most teams rested 4-7, so the raw difference has one
# extreme value (-70) that linear regression extrapolates badly. The raw
# difference is kept in the complete dataset to report as the alternative.
matches["rest_days_log_ratio"] = np.log(matches["team_a_rest_days"] / matches["team_b_rest_days"])
matches["pre_wc_form_diff"] = matches["team_a_pre_wc_form"] - matches["team_b_pre_wc_form"]
matches["tournament_form_diff"] = matches["team_a_tournament_form"] - matches["team_b_tournament_form"]


# Final checks
new_features = ["rest_days_diff", "rest_days_log_ratio", "pre_wc_form_diff", "tournament_form_diff"]
assert len(matches) == 104
assert matches[new_features].notna().all().all()
assert (matches["team_a_rest_days"] > 0).all() and (matches["team_b_rest_days"] > 0).all()


matches.to_csv(OUTPUT_FILE, index=False)
pd.DataFrame(audit_rows).to_csv(AUDIT_FILE, index=False)

print("Match rows:", len(matches))
print("Missing feature values:", matches[new_features].isna().sum().sum())
print("Saved historical dataset to:", OUTPUT_FILE)
