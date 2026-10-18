from pathlib import Path
import pandas as pd


# File location
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "variable_dictionary.csv"


variables = [
    {
        "variable": "goal_diff",
        "role": "Response",
        "definition": "Goals scored by team A minus goals scored by team B. Extra-time goals count; penalty shoot-out kicks do not.",
        "calculation": "team_a_goals - team_b_goals",
        "unit_or_encoding": "Goals",
        "source": "FBref FIFA World Cup 2026 Scores and Fixtures data",
        "source_file": "data/raw/HIT40 dataset.csv",
        "pre_match_available": "No - response variable",
        "shared_with_task_2_2": "No",
    },
    {
        "variable": "fifa_points_diff",
        "role": "Explanatory",
        "definition": "Difference between team A's and team B's FIFA ranking points.",
        "calculation": "Team A FIFA points - team B FIFA points",
        "unit_or_encoding": "FIFA ranking points",
        "source": "FIFA Men's World Ranking released 11 June 2026",
        "source_file": "data/raw/fifa_rankings_2026_06_11.csv",
        "pre_match_available": "Yes",
        "shared_with_task_2_2": "Yes",
    },
    {
        "variable": "market_value_log_ratio",
        "role": "Explanatory",
        "definition": "Relative squad market value of team A compared with team B.",
        "calculation": "ln(team A market value / team B market value)",
        "unit_or_encoding": "Log ratio (0 = equal value)",
        "source": "Planet Football, World Cup 2026 squads ranked by market value (values as at 9 June 2026)",
        "source_file": "data/raw/market_values_2026_06_09.csv",
        "pre_match_available": "Yes",
        "shared_with_task_2_2": "No",
    },
    {
        "variable": "squad_age_diff",
        "role": "Explanatory",
        "definition": "Difference in average squad age between team A and team B.",
        "calculation": "Team A average age - team B average age",
        "unit_or_encoding": "Years",
        "source": "RotoWire, based on FIFA official squad lists published 4 June 2026",
        "source_file": "data/raw/squad_ages_2026_06_04.csv",
        "pre_match_available": "Yes",
        "shared_with_task_2_2": "No",
    },
    {
        "variable": "rest_days_log_ratio",
        "role": "Explanatory",
        "definition": "Relative rest of team A compared with team B, using days since each team's previous senior international match (the same rest-day data as Task 2.2).",
        "calculation": "ln(team A rest days / team B rest days)",
        "unit_or_encoding": "Log ratio (0 = equal rest)",
        "source": "martj42 international football results dataset",
        "source_file": "data/raw/international_results.csv",
        "pre_match_available": "Yes",
        "shared_with_task_2_2": "Yes",
    },
    {
        "variable": "host_diff",
        "role": "Explanatory",
        "definition": "Whether a host nation (USA, Mexico, Canada) is playing in its own country.",
        "calculation": "Team A is host at this venue (1/0) - team B is host at this venue (1/0)",
        "unit_or_encoding": "-1, 0 or 1",
        "source": "Venue country from OpenStreetMap geocoding of the FBref venue names; team country from GeoNames",
        "source_file": "data/raw/venue_geocodes_osm.csv and geonamescache package",
        "pre_match_available": "Yes",
        "shared_with_task_2_2": "No",
    },
    {
        "variable": "travel_1000km_diff",
        "role": "Explanatory",
        "definition": "Difference in distance from each team's capital city to the stadium.",
        "calculation": "(Team A km - team B km) / 1000, great-circle distance",
        "unit_or_encoding": "Thousands of km",
        "source": "Stadium coordinates from OpenStreetMap (Nominatim via geopy); capital-city coordinates from GeoNames (geonamescache)",
        "source_file": "data/raw/venue_geocodes_osm.csv and geonamescache package",
        "pre_match_available": "Yes",
        "shared_with_task_2_2": "No",
    },
    {
        "variable": "tournament_form_diff",
        "role": "Explanatory",
        "definition": "Difference in average goal difference from each team's earlier World Cup 2026 matches (0 before a team's first match).",
        "calculation": "Team A average earlier goal difference - team B average earlier goal difference",
        "unit_or_encoding": "Goals per match",
        "source": "Derived from FBref World Cup 2026 results",
        "source_file": "data/raw/HIT40 dataset.csv",
        "pre_match_available": "Yes",
        "shared_with_task_2_2": "No",
    },
    {
        "variable": "pre_wc_form_diff",
        "role": "Explanatory",
        "definition": "Difference in average goal difference from 2026 internationals played before the World Cup.",
        "calculation": "Team A average goal difference - team B average goal difference (1 Jan to 10 June 2026)",
        "unit_or_encoding": "Goals per match",
        "source": "martj42 international football results dataset",
        "source_file": "data/raw/international_results.csv",
        "pre_match_available": "Yes",
        "shared_with_task_2_2": "No",
    },
]


dictionary = pd.DataFrame(variables)


# Basic checks
assert len(dictionary) == 9
assert (dictionary["role"] == "Response").sum() == 1
assert (dictionary["role"] == "Explanatory").sum() == 8

# The brief allows at most 4 explanatory variables shared between 2.1 and 2.2
shared = ((dictionary["role"] == "Explanatory") & (dictionary["shared_with_task_2_2"] == "Yes")).sum()
assert shared <= 4

assert dictionary.notna().all().all()


dictionary.to_csv(OUTPUT_FILE, index=False)

print("Variable dictionary created")
print("Documented variables:", len(dictionary))
print("Shared with Task 2.2:", shared)
print("Saved to:", OUTPUT_FILE)
