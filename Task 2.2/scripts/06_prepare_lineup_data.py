from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

APPEARANCES_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "lineups"
    / "match_appearances.csv"
)

MATCHES_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "lineups"
    / "matches.csv"
)

TEAM_MATCHES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_historical.csv"
)

AUDIT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "historical_features_audit.csv"
)

WORLD_CUP_STARTERS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "world_cup_starters.csv"
)

PREVIOUS_LINEUPS_NEEDED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "pre_world_cup_lineups_needed.csv"
)


# Load data
appearances = pd.read_csv(
    APPEARANCES_FILE
)

matches = pd.read_csv(
    MATCHES_FILE
)

team_matches = pd.read_csv(
    TEAM_MATCHES_FILE
)

audit = pd.read_csv(
    AUDIT_FILE
)


# Convert dates
matches["date"] = pd.to_datetime(
    matches["date"],
    format="%d %B %Y",
    errors="raise"
)

team_matches["date"] = pd.to_datetime(
    team_matches["date"],
    errors="raise"
)

audit["date"] = pd.to_datetime(
    audit["date"],
    errors="raise"
)

audit["previous_match_date"] = pd.to_datetime(
    audit["previous_match_date"],
    errors="raise"
)


# Keep World Cup starters only
starters = appearances[
    appearances["is_starter"] == True
].copy()

starters = starters[
    [
        "match_id",
        "team",
        "team_id",
        "player_id",
        "player_name",
        "position",
        "shirt_number",
    ]
].copy()


# Match the team naming used in the main dataset
starters["team"] = starters["team"].replace(
    {
        "Bosnia and Herzegovina": "Bosnia–Herz"
    }
)


# Add date and official match number
match_lookup = matches[
    [
        "match_id",
        "match_number",
        "date",
    ]
].copy()

starters = starters.merge(
    match_lookup,
    on="match_id",
    how="left",
    validate="many_to_one",
)


# Check World Cup starting lineups
assert len(starters) == 2288

starter_counts = (
    starters
    .groupby(
        [
            "match_id",
            "team"
        ]
    )
    .size()
)

assert len(starter_counts) == 208

assert (
    starter_counts == 11
).all()

assert (
    starters
    .duplicated(
        [
            "match_id",
            "team",
            "player_id"
        ]
    )
    .sum()
    == 0
)


# Check that the same teams appear in both datasets
lineup_teams = set(
    starters["team"].unique()
)

model_teams = set(
    team_matches["team"].unique()
)

missing_from_lineups = (
    model_teams - lineup_teams
)

missing_from_model = (
    lineup_teams - model_teams
)

if missing_from_lineups:
    raise ValueError(
        "Teams missing from lineup data: "
        + ", ".join(
            sorted(missing_from_lineups)
        )
    )

if missing_from_model:
    raise ValueError(
        "Unexpected teams in lineup data: "
        + ", ".join(
            sorted(missing_from_model)
        )
    )


# Match lineups using date and team
lineup_check = (
    starters[
        ["date", "team"]
    ]
    .drop_duplicates()
)

model_check = (
    team_matches[
        ["date", "team"]
    ]
    .drop_duplicates()
)

lineup_validation = model_check.merge(
    lineup_check,
    on=[
        "date",
        "team"
    ],
    how="left",
    indicator=True,
)

unmatched = lineup_validation[
    lineup_validation["_merge"]
    != "both"
]

if not unmatched.empty:
    raise ValueError(
        "Some team-match rows could not "
        "be matched to lineup data."
    )


# Find each team's first World Cup match
first_world_cup_matches = (
    audit
    .sort_values(
        [
            "team",
            "date"
        ]
    )
    .groupby(
        "team",
        as_index=False
    )
    .first()
)


# Build the list of pre-World-Cup lineups needed
previous_lineups_needed = (
    first_world_cup_matches[
        [
            "team",
            "date",
            "opponent",
            "previous_match_date",
            "previous_opponent",
            "previous_tournament",
        ]
    ]
    .rename(
        columns={
            "date":
                "first_world_cup_date",

            "opponent":
                "first_world_cup_opponent",
        }
    )
    .sort_values("team")
    .reset_index(drop=True)
)


# Match historical team names
historical_name_map = {
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

previous_lineups_needed[
    "historical_team"
] = (
    previous_lineups_needed["team"]
    .replace(historical_name_map)
)


def make_match_key(row):

    teams = sorted(
        [
            row["historical_team"],
            row["previous_opponent"],
        ]
    )

    return (
        f"{row['previous_match_date'].date()}_"
        f"{teams[0]}_vs_{teams[1]}"
    )


previous_lineups_needed[
    "previous_match_key"
] = (
    previous_lineups_needed.apply(
        make_match_key,
        axis=1
    )
)


# Save files
starters = starters.sort_values(
    [
        "date",
        "match_number",
        "team",
        "shirt_number",
    ]
).reset_index(drop=True)

starters.to_csv(
    WORLD_CUP_STARTERS_FILE,
    index=False
)

previous_lineups_needed.to_csv(
    PREVIOUS_LINEUPS_NEEDED_FILE,
    index=False
)


print(
    "World Cup starter rows:",
    len(starters)
)

print(
    "Team-match lineups:",
    len(starter_counts)
)

print(
    "Pre-World-Cup matches needed:",
    previous_lineups_needed[
        "previous_match_key"
    ].nunique()
)

print(
    "Saved lineup data to:",
    WORLD_CUP_STARTERS_FILE
)