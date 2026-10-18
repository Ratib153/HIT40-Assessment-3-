from pathlib import Path
import re

import pandas as pd
import requests

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

MATCHES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "matches_104_clean.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "venue_altitudes.csv"
)

AUDIT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "venue_altitude_source_audit.csv"
)


# Sources
ELEVATION_API = (
    "https://api.open-elevation.com"
    "/api/v1/lookup"
)

ELEVATION_DATASET = "srtm-250m"

HOST_COUNTRIES = {
    "us",
    "ca",
    "mx",
}


# Load World Cup matches
matches = pd.read_csv(
    MATCHES_FILE
)

assert "venue" in matches.columns

venues = (
    matches["venue"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

assert len(venues) > 0


# OpenStreetMap geocoder
geolocator = Nominatim(
    user_agent=(
        "hit140-assessment-3-"
        "venue-altitude"
    ),
    timeout=15,
)

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.1,
    max_retries=2,
    error_wait_seconds=5,
    swallow_exceptions=False,
)


def make_search_queries(venue):
    queries = [
        venue,
        f"{venue} stadium",
    ]

    # Some venues use a sponsored name such as
    # "GEHA Field at Arrowhead Stadium".
    if re.search(
        r"\s+at\s+",
        venue,
        flags=re.IGNORECASE,
    ):
        simpler_name = re.split(
            r"\s+at\s+",
            venue,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[1].strip()

        queries.extend([
            simpler_name,
            f"{simpler_name} stadium",
        ])

    # Remove duplicates while keeping order
    return list(
        dict.fromkeys(queries)
    )


def is_valid_stadium(location):
    raw = location.raw

    address = raw.get(
        "address",
        {}
    )

    country_code = (
        str(
            address.get(
                "country_code",
                ""
            )
        )
        .lower()
    )

    if country_code not in HOST_COUNTRIES:
        return False

    result_type = str(
        raw.get(
            "type",
            ""
        )
    ).lower()

    result_class = str(
        raw.get(
            "category",
            raw.get(
                "class",
                ""
            )
        )
    ).lower()

    display_name = (
        location.address
        .lower()
    )

    if result_type in {
        "stadium",
        "sports_centre",
    }:
        return True

    if (
        result_class == "leisure"
        and result_type == "stadium"
    ):
        return True

    stadium_words = [
        "stadium",
        "estadio",
        "field",
    ]

    return any(
        word in display_name
        for word in stadium_words
    )


def find_venue(venue):
    queries = make_search_queries(
        venue
    )

    for query in queries:

        results = geocode(
            query,
            exactly_one=False,
            limit=10,
            addressdetails=True,
            language="en",
        )

        if not results:
            continue

        for location in results:

            if is_valid_stadium(
                location
            ):
                address = (
                    location.raw.get(
                        "address",
                        {}
                    )
                )

                return {
                    "venue":
                        venue,

                    "search_query":
                        query,

                    "latitude":
                        float(
                            location.latitude
                        ),

                    "longitude":
                        float(
                            location.longitude
                        ),

                    "geocoded_address":
                        location.address,

                    "country":
                        address.get(
                            "country"
                        ),

                    "country_code":
                        address.get(
                            "country_code"
                        ),

                    "osm_type":
                        location.raw.get(
                            "osm_type"
                        ),

                    "osm_id":
                        location.raw.get(
                            "osm_id"
                        ),
                }

    raise ValueError(
        "Could not find a stadium "
        f"location for: {venue}"
    )


# Geocode every World Cup venue
location_rows = []

for venue in venues:

    print(
        "Finding coordinates:",
        venue
    )

    location = find_venue(
        venue
    )

    print(
        " Found:",
        location[
            "geocoded_address"
        ]
    )

    location_rows.append(
        location
    )


locations = pd.DataFrame(
    location_rows
)


# Request all elevations together
request_locations = []

for row in locations.itertuples():

    request_locations.append({
        "latitude":
            row.latitude,

        "longitude":
            row.longitude,
    })


payload = {
    "locations":
        request_locations,

    "dataset":
        ELEVATION_DATASET,
}


response = requests.post(
    ELEVATION_API,
    json=payload,
    timeout=60,
)

response.raise_for_status()

results = response.json().get(
    "results",
    []
)

if len(results) != len(
    locations
):
    raise ValueError(
        "Open-Elevation returned "
        "an unexpected number "
        "of results."
    )


# Add elevation results
elevations = []
datasets = []

for result in results:

    elevation = result.get(
        "elevation"
    )

    dataset = result.get(
        "dataset"
    )

    if dataset is None:
        raise ValueError(
            "Open-Elevation did not "
            "return elevation data "
            "for one of the venues."
        )

    elevations.append(
        round(
            float(elevation)
        )
    )

    datasets.append(
        dataset
    )


locations[
    "venue_altitude_m"
] = elevations

locations[
    "elevation_dataset"
] = datasets

locations[
    "geocoding_source"
] = (
    "OpenStreetMap Nominatim"
)

locations[
    "elevation_source"
] = (
    "Open-Elevation"
)


# Validate results
assert len(locations) == len(
    venues
)

assert (
    locations["venue"].nunique()
    == len(venues)
)

assert locations[
    [
        "latitude",
        "longitude",
        "venue_altitude_m",
    ]
].notna().all().all()

assert locations[
    "country_code"
].isin(
    HOST_COUNTRIES
).all()

assert (
    locations[
        "elevation_dataset"
    ]
    == ELEVATION_DATASET
).all()


# Create main altitude lookup
altitude_lookup = locations[
    [
        "venue",
        "venue_altitude_m",
    ]
].copy()

altitude_lookup.to_csv(
    OUTPUT_FILE,
    index=False
)


# Save full source information
locations.to_csv(
    AUDIT_FILE,
    index=False
)


print(
    "\nVenue altitude data created"
)

print(
    "Venues:",
    len(
        altitude_lookup
    )
)

print(
    "\nAltitude lookup:"
)

print(
    altitude_lookup.to_string(
        index=False
    )
)

print(
    "\nSaved altitude data to:",
    OUTPUT_FILE
)

print(
    "Saved source audit to:",
    AUDIT_FILE
)