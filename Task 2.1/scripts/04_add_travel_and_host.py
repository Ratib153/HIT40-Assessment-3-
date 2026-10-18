from pathlib import Path
import sys
import unicodedata

import geonamescache
import numpy as np
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
MATCHES_FILE = BASE_DIR / "data" / "processed" / "matches_104_squad.csv"
VENUE_CACHE_FILE = BASE_DIR / "data" / "raw" / "venue_geocodes_osm.csv"
CAPITALS_OUTPUT = BASE_DIR / "data" / "processed" / "team_capitals_geonames.csv"
VENUES_OUTPUT = BASE_DIR / "data" / "processed" / "venues_16.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "matches_104_travel.csv"


# Run "python scripts/04_add_travel_and_host.py --refresh" to query
# OpenStreetMap again instead of using the saved geocoding results.
REFRESH = "--refresh" in sys.argv


matches = pd.read_csv(MATCHES_FILE)
venue_names = sorted(matches["venue"].unique())
assert len(venue_names) == 16


# ----------------------------------------------------------------------
# 1. Stadium coordinates
# The 16 stadium names come from the FBref fixtures (column "Venue").
# Each name is geocoded with OpenStreetMap Nominatim through geopy.
# Results are saved to data/raw/venue_geocodes_osm.csv so the pipeline
# can be re-run offline and gives the same numbers every time.
# ----------------------------------------------------------------------

HOST_COUNTRY_CODES = ["us", "ca", "mx"]


def geocode_venues(names):
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    # Nominatim's usage policy asks for a named user agent and at most one request per second
    geolocator = Nominatim(user_agent="cdu-hit140-task-2-1", timeout=20)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    rows = []
    for name in names:
        # FBref uses "Sponsor Field at Stadium" for one venue. If the full
        # name is not found, the stadium name after " at " is tried.
        queries = [name]
        if " at " in name:
            queries.append(name.split(" at ", 1)[1])

        location = None
        for query in queries:
            location = geocode(query, exactly_one=True, addressdetails=True, country_codes=HOST_COUNTRY_CODES)
            if location is not None:
                break

        if location is None:
            raise ValueError(f"OpenStreetMap returned no result for venue: {name}")

        raw = location.raw
        if raw.get("type") != "stadium":
            raise ValueError(f"Result for {name} is not a stadium: {raw.get('display_name')}")

        rows.append({
            "venue": name,
            "query": query,
            "osm_type": raw["osm_type"],
            "osm_id": raw["osm_id"],
            "osm_name": raw.get("name", ""),
            "osm_feature": f"{raw.get('class', 'leisure')}={raw['type']}",
            "display_name": raw["display_name"],
            "latitude": float(raw["lat"]),
            "longitude": float(raw["lon"]),
            "country_code": raw["address"]["country_code"],
            "osm_url": f"https://www.openstreetmap.org/{raw['osm_type']}/{raw['osm_id']}",
            "retrieved": pd.Timestamp.today().strftime("%Y-%m-%d"),
        })
        print(f"Geocoded {name} -> {raw['display_name']}")

    return pd.DataFrame(rows)


use_cache = VENUE_CACHE_FILE.exists() and not REFRESH
if use_cache:
    venues = pd.read_csv(VENUE_CACHE_FILE)
    if set(venues["venue"]) != set(venue_names):
        use_cache = False

if not use_cache:
    venues = geocode_venues(venue_names)
    venues.to_csv(VENUE_CACHE_FILE, index=False)
    print("Saved geocoding results to:", VENUE_CACHE_FILE)

assert set(venues["venue"]) == set(venue_names)
assert (venues["osm_feature"] == "leisure=stadium").all()
assert set(venues["country_code"]).issubset(set(HOST_COUNTRY_CODES))


# ----------------------------------------------------------------------
# 2. Capital city coordinates
# Capital names and coordinates come from GeoNames, through the
# geonamescache package (no manual coordinates).
# ----------------------------------------------------------------------

# FBref team names that are spelled differently in GeoNames
geonames_country_name = {
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "Congo DR": "Democratic Republic of the Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Netherlands": "The Netherlands",
    "Türkiye": "Turkey",
    "USA": "United States",
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
}

# England and Scotland play as separate FIFA members but GeoNames lists
# only the United Kingdom, whose capital is London. Scotland's own
# capital is used for Scotland.
home_nation_capital = {"Scotland": "Edinburgh"}


def plain(text):
    # Compare names without accents or extra spaces ("Bogotá" = "Bogota")
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.strip().lower()


gc = geonamescache.GeonamesCache()
countries = pd.DataFrame(gc.get_countries().values())
cities = pd.DataFrame(gc.get_cities().values())
cities["plain_name"] = cities["name"].apply(plain)

teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))
assert len(teams) == 48

capital_rows = []
for team in teams:
    country_name = geonames_country_name.get(team, team)
    country = countries[countries["name"] == country_name]
    if len(country) != 1:
        raise ValueError(f"GeoNames country not found for {team} ({country_name})")
    country = country.iloc[0]

    capital_name = home_nation_capital.get(team, country["capital"])

    # Several places can share a capital's name (e.g. Washington),
    # so the most populated one in that country is used
    candidates = cities[(cities["countrycode"] == country["iso"]) & (cities["plain_name"] == plain(capital_name))]
    if candidates.empty:
        raise ValueError(f"GeoNames city not found for {team}: {capital_name}")
    city = candidates.sort_values("population", ascending=False).iloc[0]

    capital_rows.append({
        "team": team,
        "iso_country_code": country["iso"],
        "capital": city["name"],
        "geonames_id": int(city["geonameid"]),
        "latitude": city["latitude"],
        "longitude": city["longitude"],
    })

capitals = pd.DataFrame(capital_rows)
assert len(capitals) == 48
assert capitals[["latitude", "longitude"]].notna().all().all()


# ----------------------------------------------------------------------
# 3. Travel distance and host flag
# ----------------------------------------------------------------------

def distance_km(lat1, lon1, lat2, lon2):
    # Great-circle (haversine) distance between two points
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


venue_lookup = venues.rename(columns={"latitude": "venue_lat", "longitude": "venue_lon"})
venue_lookup["venue_country_code"] = venue_lookup["country_code"].str.upper()

enriched = matches.merge(
    venue_lookup[["venue", "venue_country_code", "venue_lat", "venue_lon"]],
    on="venue", how="left", validate="many_to_one",
)

for side in ["team_a", "team_b"]:
    side_capitals = capitals.rename(columns={
        "team": side,
        "iso_country_code": f"{side}_country_code",
        "latitude": f"{side}_lat",
        "longitude": f"{side}_lon",
    })
    enriched = enriched.merge(
        side_capitals[[side, f"{side}_country_code", f"{side}_lat", f"{side}_lon"]],
        on=side, how="left", validate="many_to_one",
    )

    # Distance from the team's capital city to the stadium
    enriched[f"{side}_travel_km"] = distance_km(
        enriched[f"{side}_lat"], enriched[f"{side}_lon"], enriched["venue_lat"], enriched["venue_lon"]
    ).round()

    # 1 if the team's country is the country the stadium is in (only USA, Mexico, Canada can be)
    enriched[f"{side}_is_host"] = (enriched[f"{side}_country_code"] == enriched["venue_country_code"]).astype(int)


enriched["travel_1000km_diff"] = (enriched["team_a_travel_km"] - enriched["team_b_travel_km"]) / 1000
enriched["host_diff"] = enriched["team_a_is_host"] - enriched["team_b_is_host"]

enriched = enriched.drop(columns=[
    "venue_lat", "venue_lon", "team_a_lat", "team_a_lon", "team_b_lat", "team_b_lon",
    "team_a_country_code", "team_b_country_code",
])


# Final checks
assert len(enriched) == 104
assert enriched[["travel_1000km_diff", "host_diff"]].notna().all().all()
assert set(enriched["host_diff"].unique()).issubset({-1, 0, 1})

host_teams = set(enriched.loc[enriched["team_a_is_host"] == 1, "team_a"]) | set(
    enriched.loc[enriched["team_b_is_host"] == 1, "team_b"]
)
assert host_teams == {"USA", "Mexico", "Canada"}


capitals.to_csv(CAPITALS_OUTPUT, index=False)
venues.to_csv(VENUES_OUTPUT, index=False)
enriched.to_csv(OUTPUT_FILE, index=False)

print("Venue coordinates:", "saved OpenStreetMap results" if use_cache else "new OpenStreetMap query")
print("Match rows:", len(enriched))
print("Matches with a host nation playing at home:", (enriched["host_diff"] != 0).sum())
print("Saved travel dataset to:", OUTPUT_FILE)
