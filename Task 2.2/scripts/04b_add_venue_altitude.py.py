from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

TEAM_MATCHES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_fifa.csv"
)

ALTITUDE_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "venue_altitudes.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_fifa_altitude.csv"
)


# Load data
team_matches = pd.read_csv(
    TEAM_MATCHES_FILE
)

altitudes = pd.read_csv(
    ALTITUDE_FILE
)


# Clean venue names
team_matches["venue"] = (
    team_matches["venue"]
    .astype(str)
    .str.strip()
)

altitudes["venue"] = (
    altitudes["venue"]
    .astype(str)
    .str.strip()
)


# Make sure altitude values are numeric
altitudes["venue_altitude_m"] = pd.to_numeric(
    altitudes["venue_altitude_m"],
    errors="raise"
)


# Check the altitude lookup
assert (
    altitudes["venue"].nunique()
    == len(altitudes)
)

assert altitudes[
    "venue_altitude_m"
].notna().all()


# Check that every match venue has an altitude value
missing_venues = (
    set(team_matches["venue"].unique())
    - set(altitudes["venue"].unique())
)

if missing_venues:
    raise ValueError(
        "Missing altitude data for: "
        + ", ".join(sorted(missing_venues))
    )


# Add altitude to the team-match dataset
enriched = team_matches.merge(
    altitudes,
    on="venue",
    how="left",
    validate="many_to_one"
)


# Final checks
assert len(enriched) == 208

assert enriched[
    "match_id"
].nunique() == 104

assert enriched[
    "venue_altitude_m"
].notna().all()

assert (
    enriched
    .groupby("match_id")
    .size()
    == 2
).all()


# Save updated dataset
enriched.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Rows:",
    len(enriched)
)

print(
    "Unique venues:",
    enriched["venue"].nunique()
)

print(
    "Saved altitude dataset to:",
    OUTPUT_FILE
)