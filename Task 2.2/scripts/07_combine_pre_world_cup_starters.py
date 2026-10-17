from pathlib import Path
import re
import unicodedata

import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

NEEDED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "pre_world_cup_lineups_needed.csv"
)

LINEUP_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "lineups"
    / "pre_world_cup_lineups"
)

FALLBACK_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "lineups"
    / "pre_world_cup_starters_fallback.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "pre_world_cup_starters.csv"
)


# Load data
needed = pd.read_csv(NEEDED_FILE)
fallback = pd.read_csv(FALLBACK_FILE)

needed["previous_match_date"] = pd.to_datetime(
    needed["previous_match_date"],
    errors="raise"
)

fallback["date"] = pd.to_datetime(
    fallback["date"],
    errors="raise"
)


# Different sources use slightly different team names
slug_aliases = {
    "United States": [
        "usa",
        "united-states",
    ],
    "Bosnia and Herzegovina": [
        "bosnia-herzegovina",
        "bosnia-and-herzegovina",
    ],
    "Cape Verde": [
        "cape-verde",
        "cabo-verde",
    ],
    "Czech Republic": [
        "czechia",
        "czech-republic",
    ],
    "South Korea": [
        "korea-republic",
        "south-korea",
    ],
    "Turkey": [
        "turkiye",
        "turkey",
    ],
    "Republic of Ireland": [
        "rep-of-ireland",
        "republic-of-ireland",
        "ireland",
    ],
}


def make_slug(name):
    name = unicodedata.normalize(
        "NFKD",
        str(name)
    )

    name = "".join(
        character
        for character in name
        if not unicodedata.combining(character)
    )

    name = name.lower()
    name = name.replace("&", " and ")
    name = name.replace("'", "")

    name = re.sub(
        r"[^a-z0-9]+",
        "-",
        name
    )

    return name.strip("-")


def possible_slugs(name):
    if name in slug_aliases:
        return slug_aliases[name]

    return [make_slug(name)]


def find_column(columns, names):
    cleaned = {
        str(column).strip().lower(): column
        for column in columns
    }

    for name in names:
        if name in cleaned:
            return cleaned[name]

    return None


def find_lineup_file(row):
    team_slugs = possible_slugs(
        row.historical_team
    )

    opponent_slugs = possible_slugs(
        row.previous_opponent
    )

    # Some lineup sources record the match one day
    # differently, so check the day before and after.
    possible_dates = [
        row.previous_match_date,
        row.previous_match_date
        - pd.Timedelta(days=1),
        row.previous_match_date
        + pd.Timedelta(days=1),
    ]

    for date_value in possible_dates:

        date_text = date_value.strftime(
            "%Y-%m-%d"
        )

        for team_slug in team_slugs:
            for opponent_slug in opponent_slugs:

                home_file = (
                    LINEUP_DIR
                    / (
                        f"{date_text}_"
                        f"{team_slug}_vs_"
                        f"{opponent_slug}_lineups.csv"
                    )
                )

                if home_file.exists():
                    return home_file, "home"

                away_file = (
                    LINEUP_DIR
                    / (
                        f"{date_text}_"
                        f"{opponent_slug}_vs_"
                        f"{team_slug}_lineups.csv"
                    )
                )

                if away_file.exists():
                    return away_file, "away"

    raise FileNotFoundError(
        f"No lineup file found for "
        f"{row.team} around "
        f"{row.previous_match_date.date()}."
    )


# The cleaned folder should contain 24 usable files
lineup_files = list(
    LINEUP_DIR.glob("*.csv")
)

assert len(lineup_files) == 24


fallback_teams = set(
    fallback["team"].unique()
)

downloaded_rows = []


# Read downloaded lineups for teams not covered by fallback data
for row in needed.itertuples():

    if row.team in fallback_teams:
        continue

    file_path, required_side = (
        find_lineup_file(row)
    )

    lineup = pd.read_csv(
        file_path
    )

    side_column = find_column(
        lineup.columns,
        [
            "side",
            "team_side",
            "home_away",
        ]
    )

    role_column = find_column(
        lineup.columns,
        [
            "role",
            "lineup_role",
            "status",
            "is_starter",
        ]
    )

    player_column = find_column(
        lineup.columns,
        [
            "player",
            "player_name",
            "name",
        ]
    )

    if (
        side_column is None
        or role_column is None
        or player_column is None
    ):
        raise ValueError(
            f"Required lineup columns are missing "
            f"from {file_path.name}."
        )


    role_values = (
        lineup[role_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    starters = lineup[
        role_values.str.contains(
            "start",
            na=False
        )
    ].copy()


    side_values = (
        starters[side_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    team_starters = starters[
        side_values == required_side
    ].copy()


    if len(team_starters) != 11:
        raise ValueError(
            f"Expected 11 starters for "
            f"{row.team}, found "
            f"{len(team_starters)}."
        )


    for player_name in team_starters[
        player_column
    ].astype(str):

        downloaded_rows.append({
            "date":
                row.previous_match_date,

            "team":
                row.team,

            "opponent":
                row.previous_opponent,

            "player_name":
                player_name.strip(),

            "source_name":
                "Cornerflick",

            "source_file":
                file_path.name,

            "collection_method":
                "downloaded_lineup",
        })


downloaded_starters = pd.DataFrame(
    downloaded_rows
)


# Prepare fallback lineup data
fallback = fallback.copy()

fallback["source_file"] = ""

fallback = fallback[
    [
        "date",
        "team",
        "opponent",
        "player_name",
        "source_name",
        "source_file",
        "collection_method",
    ]
]


# Combine downloaded and fallback lineups
all_starters = pd.concat(
    [
        downloaded_starters,
        fallback
    ],
    ignore_index=True
)


# Create a simple player key for duplicate checks
def make_player_key(name):
    name = unicodedata.normalize(
        "NFKD",
        str(name)
    )

    name = "".join(
        character
        for character in name
        if not unicodedata.combining(character)
    )

    name = name.lower()

    return re.sub(
        r"[^a-z0-9]+",
        "",
        name
    )


all_starters["player_key"] = (
    all_starters["player_name"]
    .apply(make_player_key)
)


# Final checks
assert len(downloaded_starters) == 352

assert downloaded_starters[
    "team"
].nunique() == 32

assert len(fallback) == 176

assert fallback[
    "team"
].nunique() == 16

assert len(all_starters) == 528

assert all_starters[
    "team"
].nunique() == 48

starter_counts = (
    all_starters
    .groupby("team")
    .size()
)

assert (
    starter_counts == 11
).all()

assert set(
    all_starters["team"].unique()
) == set(
    needed["team"].unique()
)

assert (
    all_starters
    .duplicated(
        [
            "team",
            "player_key"
        ]
    )
    .sum()
    == 0
)


# Save the complete pre-World-Cup starter dataset
all_starters = all_starters.sort_values(
    [
        "team",
        "date",
        "player_name"
    ]
).reset_index(drop=True)

all_starters.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Starter rows:",
    len(all_starters)
)

print(
    "Teams:",
    all_starters["team"].nunique()
)

print(
    "Downloaded lineup teams:",
    downloaded_starters["team"].nunique()
)

print(
    "Fallback teams:",
    fallback["team"].nunique()
)

print(
    "Saved pre-World-Cup starters to:",
    OUTPUT_FILE
)