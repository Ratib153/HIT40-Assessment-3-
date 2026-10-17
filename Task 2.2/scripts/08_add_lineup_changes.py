from pathlib import Path
import re
import unicodedata

import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_historical.csv"
)

PREVIOUS_STARTERS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "pre_world_cup_starters.csv"
)

WORLD_CUP_STARTERS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "world_cup_starters.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_lineup.csv"
)

AUDIT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "lineup_changes_audit.csv"
)

ALIAS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "lineup_name_aliases.csv"
)


# Load data
model = pd.read_csv(MODEL_FILE)
previous = pd.read_csv(PREVIOUS_STARTERS_FILE)
world_cup = pd.read_csv(WORLD_CUP_STARTERS_FILE)

model["date"] = pd.to_datetime(
    model["date"],
    errors="raise"
)

previous["date"] = pd.to_datetime(
    previous["date"],
    errors="raise"
)

world_cup["date"] = pd.to_datetime(
    world_cup["date"],
    errors="raise"
)


# Match team naming used in the main dataset
world_cup["team"] = world_cup["team"].replace(
    {
        "Bosnia and Herzegovina": "Bosnia–Herz"
    }
)


# Normalise player names for comparison
CHARACTER_MAP = str.maketrans({
    "ø": "o",
    "Ø": "O",
    "ł": "l",
    "Ł": "L",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "Th",
    "æ": "ae",
    "Æ": "AE",
    "œ": "oe",
    "Œ": "OE",
    "ß": "ss",
    "ı": "i",
    "İ": "I",
    "ﬁ": "fi",
    "ﬂ": "fl",
})


def make_player_key(name):
    name = str(name).strip()
    name = name.translate(CHARACTER_MAP)

    name = unicodedata.normalize(
        "NFKD",
        name
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


previous["player_key"] = (
    previous["player_name"]
    .apply(make_player_key)
)

world_cup["player_key"] = (
    world_cup["player_name"]
    .apply(make_player_key)
)


# Same players written differently across sources
ALIASES = {
    "Türkiye": [
        ("Abdülkerim Bardakcı", "Abdulkerim BARDAKCI"),
        ("Uğurcan Çakır", "Ugurcan CAKIR"),
        ("Barış Alper Yılmaz", "Baris Alper YILMAZ"),
    ],

    "IR Iran": [
        ("Shojae Khalilzadeh", "Shoja KHALILZADEH"),
        ("Ariya Yousefi", "Arya YOUSEFI"),
        ("Shahriar Moghanlou", "Shahriyar MOGHANLOO"),
    ],

    "Norway": [
        ("Alexander Sørloth", "Alexander SORLOTH"),
        ("David Møller Wolfe", "David MOLLER WOLFE"),
        ("Martin Ødegaard", "Martin ODEGAARD"),
        ("Torbjørn Heggem", "Torbjorn HEGGEM"),
        ("Ørjan Nyland", "Orjan NYLAND"),
    ],

    "Egypt": [
        ("Marwan Attia", "MARAWAN ATTIA"),
        ("Mostafa Shobeir", "MOSTAFA SHOUBIR"),
        ("Mostafa Ziko", "MOSTAFA ZICO"),
        ("Mohanad Lasheen", "MOHANAD LASHIN"),
    ],

    "Qatar": [
        ("Mahmud Abunad", "MAHMOUD ABUNADA"),
        ("Homam Al-Amin", "HOMAM AHMED"),
        ("Jassem Gaber Abdulsallam", "JASSEM GABER"),
    ],

    "Jordan": [
        ("Mousa Tamari", "MOUSA ALTAMARI"),
        ("Noor Al-Deen Al Rawabdeh", "NOOR ALRAWABDEH"),
        ("Nizar Mahmoud Al-Rashdan", "NIZAR ALRASHDAN"),
    ],

    "Haiti": [
        ("Louicius Don Deedson", "Louicius DEEDSON"),
    ],

    "USA": [
        ("Matthew Freese", "Matt FREESE"),
        ("Alexander Freeman", "Alex FREEMAN"),
    ],

    "Saudi Arabia": [
        ("Firas Al-Buraikan", "FERAS ALBRIKAN"),
    ],

    "Congo DR": [
        ("Lionel Mpasi Nzau", "Lionel MPASI"),
    ],

    "Panama": [
        ("Yoel Bárcenas", "Edgar Yoel BARCENAS"),
    ],

    "Colombia": [
        ("Luis Javier Suárez", "Luis SUAREZ"),
    ],

    "Senegal": [
        ("Malick Diouf", "El Hadji Malick DIOUF"),
    ],

    "Paraguay": [
        ("Juan Caceres", "Juan Jose CACERES"),
    ],

    "Tunisia": [
        ("Abdelmouhib Chamakh", "Mouhib CHAMAKH"),
        ("Amine Ben Hmida", "Mohamed Amine BEN HMIDA"),
    ],

    "Morocco": [
        ("Bono", "Yassine BOUNOU"),
    ],

    "Netherlands": [
        ("Crysencio Summerville", "Crysencio"),
    ],
}


# Convert aliases into normalised player keys
alias_rows = []
alias_key_map = {}

for team, aliases in ALIASES.items():

    for previous_name, world_cup_name in aliases:

        previous_key = make_player_key(
            previous_name
        )

        world_cup_key = make_player_key(
            world_cup_name
        )

        alias_key_map[
            (
                team,
                previous_key
            )
        ] = world_cup_key

        alias_rows.append({
            "team": team,
            "previous_name": previous_name,
            "world_cup_name": world_cup_name,
            "previous_key": previous_key,
            "world_cup_key": world_cup_key,
        })


alias_table = pd.DataFrame(
    alias_rows
)


# Check that each alias exists in both datasets
for row in alias_table.itertuples():

    previous_exists = (
        (
            previous["team"] == row.team
        )
        &
        (
            previous["player_key"]
            == row.previous_key
        )
    ).any()

    current_exists = (
        (
            world_cup["team"] == row.team
        )
        &
        (
            world_cup["player_key"]
            == row.world_cup_key
        )
    ).any()

    if not previous_exists:
        raise ValueError(
            f"Previous player not found: "
            f"{row.team} - {row.previous_name}"
        )

    if not current_exists:
        raise ValueError(
            f"World Cup player not found: "
            f"{row.team} - {row.world_cup_name}"
        )


alias_table.to_csv(
    ALIAS_FILE,
    index=False
)


# Check World Cup starter data
assert world_cup[
    "player_id"
].notna().all()

wc_counts = (
    world_cup
    .groupby(
        [
            "team",
            "date"
        ]
    )
    .size()
)

assert len(wc_counts) == 208

assert (
    wc_counts == 11
).all()


# Check pre-tournament starter data
previous_counts = (
    previous
    .groupby("team")
    .size()
)

assert len(previous_counts) == 48

assert (
    previous_counts == 11
).all()


# Calculate lineup changes
change_rows = []

for team in sorted(
    world_cup["team"].unique()
):

    team_wc = world_cup[
        world_cup["team"] == team
    ].copy()

    match_dates = sorted(
        team_wc["date"].unique()
    )

    previous_wc_ids = None

    for match_number, match_date in enumerate(
        match_dates,
        start=1
    ):

        current = team_wc[
            team_wc["date"] == match_date
        ].copy()

        current_ids = set(
            current["player_id"]
        )

        current_keys = set(
            current["player_key"]
        )


        # First World Cup match uses the pre-tournament lineup
        if match_number == 1:

            previous_team = previous[
                previous["team"] == team
            ].copy()

            mapped_previous_keys = set()

            for previous_row in previous_team.itertuples():

                previous_key = (
                    previous_row.player_key
                )

                mapped_key = alias_key_map.get(
                    (
                        team,
                        previous_key
                    ),
                    previous_key
                )

                mapped_previous_keys.add(
                    mapped_key
                )

            retained = len(
                mapped_previous_keys
                & current_keys
            )

            comparison_method = (
                "pre_tournament_name_match"
            )


        # Later matches use player IDs from the World Cup data
        else:

            retained = len(
                current_ids
                & previous_wc_ids
            )

            comparison_method = (
                "world_cup_player_id"
            )


        lineup_changes = (
            11 - retained
        )

        change_rows.append({
            "team": team,
            "date": pd.Timestamp(match_date),
            "retained_starters": retained,
            "lineup_changes": lineup_changes,
            "comparison_method": comparison_method,
        })

        previous_wc_ids = current_ids


changes = pd.DataFrame(
    change_rows
)


# Check calculated values
assert len(changes) == 208

assert changes[
    "lineup_changes"
].between(
    0,
    11
).all()

assert (
    changes["retained_starters"]
    + changes["lineup_changes"]
    == 11
).all()


# Add lineup changes to the main dataset
enriched = model.merge(
    changes,
    on=[
        "team",
        "date"
    ],
    how="left",
    validate="one_to_one"
)


assert len(enriched) == 208

assert enriched[
    "lineup_changes"
].notna().all()

assert enriched[
    "lineup_changes"
].between(
    0,
    11
).all()

assert (
    enriched
    .groupby("match_id")
    .size()
    == 2
).all()


enriched["lineup_changes"] = (
    enriched["lineup_changes"]
    .astype(int)
)

enriched["retained_starters"] = (
    enriched["retained_starters"]
    .astype(int)
)


# Save files
enriched.to_csv(
    OUTPUT_FILE,
    index=False
)

changes.to_csv(
    AUDIT_FILE,
    index=False
)


print(
    "Rows:",
    len(enriched)
)

print(
    "Missing lineup changes:",
    enriched["lineup_changes"]
    .isna()
    .sum()
)

print(
    "Lineup changes range:",
    enriched["lineup_changes"].min(),
    "to",
    enriched["lineup_changes"].max()
)

print(
    "Saved lineup dataset to:",
    OUTPUT_FILE
)