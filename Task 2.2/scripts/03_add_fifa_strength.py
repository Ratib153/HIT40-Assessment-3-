from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

RANKINGS_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "fifa_rankings_2026_06_11.csv"
)

TEAM_MATCHES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_base.csv"
)

OUTPUT_LOOKUP = (
    BASE_DIR
    / "data"
    / "processed"
    / "fifa_world_cup_48.csv"
)

OUTPUT_DATASET = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_fifa.csv"
)


# Load FIFA ranking data
rankings = pd.read_csv(
    RANKINGS_FILE,
    sep="\t",
    header=None,
    skiprows=1,
    names=[
        "fifa_rank",
        "blank_1",
        "rank_change",
        "blank_2",
        "ranking_team",
        "fifa_points",
        "points_change",
    ],
)

rankings = rankings[
    [
        "fifa_rank",
        "ranking_team",
        "fifa_points",
    ]
].copy()

rankings["fifa_rank"] = pd.to_numeric(
    rankings["fifa_rank"],
    errors="raise"
).astype(int)

rankings["fifa_points"] = pd.to_numeric(
    rankings["fifa_points"],
    errors="raise"
)

rankings["ranking_team"] = (
    rankings["ranking_team"]
    .astype(str)
    .str.strip()
)


# Load the 208-row team-match dataset
team_matches = pd.read_csv(
    TEAM_MATCHES_FILE
)


# Team names that differ between the two sources
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


# Create FIFA ranking lookup for the 48 World Cup teams
world_cup_teams = sorted(
    team_matches["team"]
    .dropna()
    .unique()
)

lookup_rows = []

for team in world_cup_teams:

    ranking_name = name_map.get(
        team,
        team
    )

    match = rankings[
        rankings["ranking_team"]
        == ranking_name
    ]

    if len(match) != 1:
        raise ValueError(
            f"Could not match {team} "
            f"to FIFA ranking data."
        )

    row = match.iloc[0]

    lookup_rows.append({
        "team": team,
        "fifa_rank": int(
            row["fifa_rank"]
        ),
        "fifa_points": float(
            row["fifa_points"]
        ),
    })


world_rankings = pd.DataFrame(
    lookup_rows
)


# Check the lookup before merging
assert len(world_rankings) == 48

assert world_rankings[
    "team"
].nunique() == 48

assert world_rankings[
    [
        "fifa_rank",
        "fifa_points",
    ]
].notna().all().all()


# Add FIFA information for each team
team_rankings = world_rankings.rename(
    columns={
        "fifa_rank":
            "team_fifa_rank",

        "fifa_points":
            "team_fifa_points",
    }
)

enriched = team_matches.merge(
    team_rankings,
    on="team",
    how="left",
    validate="many_to_one",
)


# Add FIFA information for each opponent
opponent_rankings = world_rankings.rename(
    columns={
        "team":
            "opponent",

        "fifa_rank":
            "opponent_fifa_rank",

        "fifa_points":
            "opponent_fifa_points",
    }
)

enriched = enriched.merge(
    opponent_rankings,
    on="opponent",
    how="left",
    validate="many_to_one",
)


# Difference in FIFA points between the team and opponent
enriched["fifa_points_diff"] = (
    enriched["team_fifa_points"]
    - enriched["opponent_fifa_points"]
)


# Final checks
assert len(enriched) == 208

assert enriched[
    "match_id"
].nunique() == 104

assert enriched[
    [
        "team_fifa_rank",
        "team_fifa_points",
        "opponent_fifa_rank",
        "opponent_fifa_points",
        "fifa_points_diff",
    ]
].notna().all().all()

assert (
    enriched
    .groupby("match_id")
    .size()
    == 2
).all()


# Save files
world_rankings.to_csv(
    OUTPUT_LOOKUP,
    index=False
)

enriched.to_csv(
    OUTPUT_DATASET,
    index=False
)


print(
    "World Cup teams matched:",
    len(world_rankings)
)

print(
    "Team-match rows:",
    len(enriched)
)

print(
    "Saved FIFA dataset to:",
    OUTPUT_DATASET
)