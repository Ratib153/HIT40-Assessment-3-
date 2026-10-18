from pathlib import Path
import numpy as np
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
MARKET_VALUE_FILE = BASE_DIR / "data" / "raw" / "market_values_2026_06_09.csv"
ELO_FILE = BASE_DIR / "data" / "raw" / "elo_ratings_2025_12_31.csv"
SQUAD_AGE_FILE = BASE_DIR / "data" / "raw" / "squad_ages_2026_06_04.csv"
MATCHES_FILE = BASE_DIR / "data" / "processed" / "matches_104_fifa.csv"
OUTPUT_LOOKUP = BASE_DIR / "data" / "processed" / "team_strength_48.csv"
OUTPUT_DATASET = BASE_DIR / "data" / "processed" / "matches_104_squad.csv"


# Every source spells some teams differently, so map them to the FBref names
name_map = {
    "Bosnia and Herzegovina": "Bosnia–Herz",
    "Bosnia And Herzegovina": "Bosnia–Herz",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "Curacao": "Curaçao",
    "Czech Republic": "Czechia",
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",
    "United States": "USA",
}


def standard_name(name):
    name = str(name).strip()
    return name_map.get(name, name)


matches = pd.read_csv(MATCHES_FILE)
world_cup_teams = set(matches["team_a"]) | set(matches["team_b"])


# Squad market value (values as at 9 June 2026, see Data Sources.txt)
# The copied table repeats each team three times and stores values as text ("€1,523M")
market = pd.read_csv(MARKET_VALUE_FILE, header=None)
market = market[market[0].notna()]
market = pd.DataFrame({
    "team": market[0].apply(standard_name),
    "market_value_m": market[2].astype(str).str.replace(r"[€,M]", "", regex=True).astype(float),
})


# Squad age (RotoWire, from FIFA's official squad lists published 4 June 2026)
squad_age = pd.read_csv(SQUAD_AGE_FILE)
squad_age = pd.DataFrame({
    "team": squad_age["team"].apply(standard_name),
    "squad_age": squad_age["average_squad_age"],
})


# Elo rating (World Football Elo Ratings, values as at 31 December 2025)
# The copied table is one long column: 20 header cells, then 16 values per team.
# Elo is NOT one of the 8 model variables; it is kept to test it as an
# alternative to FIFA points, because the two are strongly correlated.
elo_column = pd.read_csv(ELO_FILE, header=None, dtype=str)[0].tolist()[20:]
elo = pd.DataFrame({
    "team": [standard_name(elo_column[i + 1]) for i in range(0, len(elo_column), 16)],
    "elo_rating": [float(elo_column[i + 2]) for i in range(0, len(elo_column), 16)],
})
elo = elo[elo["team"].isin(world_cup_teams)]


# One lookup row per World Cup team
strength = market.merge(squad_age, on="team", how="inner").merge(elo, on="team", how="inner")

missing = world_cup_teams - set(strength["team"])
if missing:
    raise ValueError("Missing team strength data for: " + ", ".join(sorted(missing)))

assert len(strength) == 48
assert strength.notna().all().all()


# Add team A and team B values
enriched = matches.copy()
for side in ["team_a", "team_b"]:
    side_values = strength.rename(columns={
        "team": side,
        "market_value_m": f"{side}_market_value_m",
        "squad_age": f"{side}_squad_age",
        "elo_rating": f"{side}_elo_rating",
    })
    enriched = enriched.merge(side_values, on=side, how="left", validate="many_to_one")


# Market values are very skewed (France 1523M vs Qatar 20M),
# so the log of the ratio is used instead of a plain difference
enriched["market_value_log_ratio"] = np.log(enriched["team_a_market_value_m"] / enriched["team_b_market_value_m"])
enriched["squad_age_diff"] = enriched["team_a_squad_age"] - enriched["team_b_squad_age"]
enriched["elo_rating_diff"] = enriched["team_a_elo_rating"] - enriched["team_b_elo_rating"]


# Final checks
assert len(enriched) == 104
assert enriched[["market_value_log_ratio", "squad_age_diff", "elo_rating_diff"]].notna().all().all()


strength.to_csv(OUTPUT_LOOKUP, index=False)
enriched.to_csv(OUTPUT_DATASET, index=False)

print("Teams with market value, squad age and Elo:", len(strength))
print("Match rows:", len(enriched))
print("Saved dataset to:", OUTPUT_DATASET)
