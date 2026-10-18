from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
RANKINGS_FILE = BASE_DIR / "data" / "raw" / "fifa_rankings_2026_06_11.csv"
MATCHES_FILE = BASE_DIR / "data" / "processed" / "matches_104_clean.csv"
OUTPUT_LOOKUP = BASE_DIR / "data" / "processed" / "fifa_world_cup_48.csv"
OUTPUT_DATASET = BASE_DIR / "data" / "processed" / "matches_104_fifa.csv"


# Load FIFA ranking data (tab separated, released 11 June 2026)
rankings = pd.read_csv(
    RANKINGS_FILE,
    sep="\t",
    header=None,
    skiprows=1,
    names=["fifa_rank", "blank_1", "rank_change", "blank_2", "ranking_team", "fifa_points", "points_change"],
)
rankings = rankings[["fifa_rank", "ranking_team", "fifa_points"]].copy()
rankings["fifa_rank"] = pd.to_numeric(rankings["fifa_rank"], errors="raise").astype(int)
rankings["fifa_points"] = pd.to_numeric(rankings["fifa_points"], errors="raise")
rankings["ranking_team"] = rankings["ranking_team"].astype(str).str.strip()


# Load matches
matches = pd.read_csv(MATCHES_FILE)


# Team names that differ between the two sources (same map as Task 2.2)
name_map = {
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}


# FIFA ranking lookup for the 48 World Cup teams
world_cup_teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))

lookup_rows = []
for team in world_cup_teams:
    match = rankings[rankings["ranking_team"] == name_map.get(team, team)]
    if len(match) != 1:
        raise ValueError(f"Could not match {team} to FIFA ranking data.")
    lookup_rows.append({
        "team": team,
        "fifa_rank": int(match.iloc[0]["fifa_rank"]),
        "fifa_points": float(match.iloc[0]["fifa_points"]),
    })

world_rankings = pd.DataFrame(lookup_rows)
assert len(world_rankings) == 48


# Add FIFA points for team A and team B
enriched = matches.merge(
    world_rankings[["team", "fifa_points"]].rename(columns={"team": "team_a", "fifa_points": "team_a_fifa_points"}),
    on="team_a", how="left", validate="many_to_one",
)
enriched = enriched.merge(
    world_rankings[["team", "fifa_points"]].rename(columns={"team": "team_b", "fifa_points": "team_b_fifa_points"}),
    on="team_b", how="left", validate="many_to_one",
)

# Difference in FIFA points between team A and team B
enriched["fifa_points_diff"] = enriched["team_a_fifa_points"] - enriched["team_b_fifa_points"]


# Final checks
assert len(enriched) == 104
assert enriched[["team_a_fifa_points", "team_b_fifa_points", "fifa_points_diff"]].notna().all().all()


world_rankings.to_csv(OUTPUT_LOOKUP, index=False)
enriched.to_csv(OUTPUT_DATASET, index=False)

print("World Cup teams matched:", len(world_rankings))
print("Match rows:", len(enriched))
print("Saved FIFA dataset to:", OUTPUT_DATASET)
