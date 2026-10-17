from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

TEAM_MATCHES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_fifa_altitude.csv"
)

HISTORY_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "international_results.csv"
)

OUTPUT_FILE = (
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


# Load data
team_matches = pd.read_csv(
    TEAM_MATCHES_FILE
)

history = pd.read_csv(
    HISTORY_FILE
)

team_matches["date"] = pd.to_datetime(
    team_matches["date"],
    errors="raise"
)

history["date"] = pd.to_datetime(
    history["date"],
    errors="raise"
)


# Team names that differ between the two sources
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

team_matches["history_team"] = (
    team_matches["team"]
    .replace(name_map)
)

team_matches["history_opponent"] = (
    team_matches["opponent"]
    .replace(name_map)
)


# Convert historical matches to one row per team
home_rows = history[
    [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    ]
].rename(
    columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_score": "goals_scored",
        "away_score": "goals_conceded",
    }
)

away_rows = history[
    [
        "date",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "tournament",
    ]
].rename(
    columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_score": "goals_scored",
        "home_score": "goals_conceded",
    }
)

team_history = pd.concat(
    [home_rows, away_rows],
    ignore_index=True
)

team_history = team_history.sort_values(
    by=["team", "date"]
).reset_index(drop=True)


# Check that every World Cup team exists in the history data
history_teams = set(
    team_history["team"].unique()
)

required_teams = set(
    team_matches["history_team"].unique()
)

required_opponents = set(
    team_matches["history_opponent"].unique()
)

missing_teams = (
    required_teams
    .union(required_opponents)
    - history_teams
)

if missing_teams:
    raise ValueError(
        "Teams missing from historical data: "
        + ", ".join(sorted(missing_teams))
    )


# Extra-time information for World Cup matches
extra_time_lookup = {}

for row in team_matches.itertuples():

    key = (
        row.history_team,
        row.date
    )

    extra_time_lookup[key] = int(
        row.match_extra_time
    )


# Build historical features
feature_rows = []
audit_rows = []

for row in team_matches.itertuples():

    current_date = row.date
    team_name = row.history_team
    opponent_name = row.history_opponent

    team_previous = team_history[
        (team_history["team"] == team_name)
        & (team_history["date"] < current_date)
    ].sort_values("date")

    opponent_previous = team_history[
        (team_history["team"] == opponent_name)
        & (team_history["date"] < current_date)
    ].sort_values("date")

    if len(team_previous) < 5:
        raise ValueError(
            f"{row.team} has fewer than five "
            f"previous matches."
        )

    if len(opponent_previous) < 5:
        raise ValueError(
            f"{row.opponent} has fewer than five "
            f"previous matches."
        )


    previous_match = team_previous.iloc[-1]
    previous_match_date = previous_match["date"]

    rest_days = (
        current_date
        - previous_match_date
    ).days


    last_five_team = team_previous.tail(5)

    attacking_form_5 = (
        last_five_team["goals_scored"]
        .mean()
    )


    last_five_opponent = (
        opponent_previous.tail(5)
    )

    opponent_defensive_form_5 = (
        last_five_opponent[
            "goals_conceded"
        ].mean()
    )


    previous_extra_time_key = (
        team_name,
        previous_match_date
    )

    if previous_extra_time_key in extra_time_lookup:

        previous_extra_time = (
            extra_time_lookup[
                previous_extra_time_key
            ]
        )

    else:

        if previous_match["tournament"] != "Friendly":
            raise ValueError(
                f"Could not determine previous "
                f"extra-time status for {row.team}."
            )

        previous_extra_time = 0


    feature_rows.append({
        "rest_days":
            rest_days,

        "attacking_form_5":
            attacking_form_5,

        "opponent_defensive_form_5":
            opponent_defensive_form_5,

        "previous_extra_time":
            previous_extra_time,
    })


    # Kept separately because later lineup work uses
    # the previous match information
    audit_rows.append({
        "match_id":
            row.match_id,

        "date":
            current_date,

        "team":
            row.team,

        "opponent":
            row.opponent,

        "previous_match_date":
            previous_match_date,

        "previous_opponent":
            previous_match["opponent"],

        "previous_tournament":
            previous_match["tournament"],

        "rest_days":
            rest_days,

        "attacking_form_5":
            attacking_form_5,

        "opponent_defensive_form_5":
            opponent_defensive_form_5,

        "previous_extra_time":
            previous_extra_time,
    })


# Add features to the dataset
features = pd.DataFrame(
    feature_rows
)

enriched = pd.concat(
    [
        team_matches.reset_index(drop=True),
        features
    ],
    axis=1
)

enriched = enriched.drop(
    columns=[
        "history_team",
        "history_opponent"
    ]
)

audit = pd.DataFrame(
    audit_rows
)


# Final checks
new_features = [
    "rest_days",
    "attacking_form_5",
    "opponent_defensive_form_5",
    "previous_extra_time",
]

assert len(enriched) == 208

assert enriched[
    "match_id"
].nunique() == 104

assert (
    enriched
    .groupby("match_id")
    .size()
    == 2
).all()

assert enriched[
    new_features
].notna().all().all()

assert (
    enriched["rest_days"] > 0
).all()

assert (
    enriched["attacking_form_5"] >= 0
).all()

assert (
    enriched[
        "opponent_defensive_form_5"
    ] >= 0
).all()

assert set(
    enriched[
        "previous_extra_time"
    ].unique()
).issubset({0, 1})


# Save files
enriched.to_csv(
    OUTPUT_FILE,
    index=False
)

audit.to_csv(
    AUDIT_FILE,
    index=False
)


print(
    "Rows:",
    len(enriched)
)

print(
    "Unique teams:",
    enriched["team"].nunique()
)

print(
    "Missing feature values:",
    enriched[new_features]
    .isna()
    .sum()
    .sum()
)

print(
    "Saved historical dataset to:",
    OUTPUT_FILE
)