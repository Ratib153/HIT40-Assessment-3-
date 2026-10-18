from pathlib import Path
import pandas as pd


# File location
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "variable_dictionary.csv"
)


variables = [
    {
        "variable": "goals_scored",
        "role": "Response",
        "definition":
            "Number of goals scored by the team in the match. "
            "Penalty shootout kicks are not included.",
        "calculation":
            "Taken from the final match score for the team.",
        "unit_or_encoding":
            "Goals",
        "source":
            "FBref FIFA World Cup 2026 Scores and Fixtures data",
        "source_file":
            "data/raw/HIT40 dataset.csv",
        "pre_match_available":
            "No - response variable",
        "shared_with_task_2_1":
            "No",
    },
    {
        "variable": "fifa_points_diff",
        "role": "Explanatory",
        "definition":
            "Difference between the team's FIFA ranking points "
            "and the opponent's FIFA ranking points.",
        "calculation":
            "Team FIFA points - opponent FIFA points",
        "unit_or_encoding":
            "FIFA ranking points",
        "source":
            "FIFA Men's World Ranking released 11 June 2026",
        "source_file":
            "data/raw/fifa_rankings_2026_06_11.csv",
        "pre_match_available":
            "Yes",
        "shared_with_task_2_1":
            "Yes",
    },
    {
        "variable": "rest_days",
        "role": "Explanatory",
        "definition":
            "Number of days since the team's previous senior "
            "international match.",
        "calculation":
            "Current match date - previous match date",
        "unit_or_encoding":
            "Days",
        "source":
            "martj42 international football results dataset",
        "source_file":
            "data/raw/international_results.csv",
        "pre_match_available":
            "Yes",
        "shared_with_task_2_1":
            "Yes",
    },
    {
        "variable": "venue_altitude_m",
        "role": "Explanatory",
        "definition":
            "Height of the match stadium above sea level.",
        "calculation":
            "Stadium locations were geocoded using OpenStreetMap "
            "Nominatim and elevation was retrieved programmatically "
            "from Open-Elevation.",
        "unit_or_encoding":
            "Metres",
        "source":
            "OpenStreetMap Nominatim and Open-Elevation using SRTM elevation data",
        "source_file":
            "data/raw/venue_altitudes.csv and "
            "data/raw/venue_altitude_source_audit.csv",
        "pre_match_available":
            "Yes",
        "shared_with_task_2_1":
            "No",
    },
    {
        "variable": "attacking_form_5",
        "role": "Explanatory",
        "definition":
            "Average goals scored by the team in its previous five "
            "senior international matches.",
        "calculation":
            "Goals scored in the previous five matches divided by 5",
        "unit_or_encoding":
            "Average goals per match",
        "source":
            "martj42 international football results dataset",
        "source_file":
            "data/raw/international_results.csv",
        "pre_match_available":
            "Yes",
        "shared_with_task_2_1":
            "No",
    },
    {
        "variable": "opponent_defensive_form_5",
        "role": "Explanatory",
        "definition":
            "Average goals conceded by the opponent in its previous "
            "five senior international matches.",
        "calculation":
            "Goals conceded by the opponent in the previous five "
            "matches divided by 5",
        "unit_or_encoding":
            "Average goals conceded per match",
        "source":
            "martj42 international football results dataset",
        "source_file":
            "data/raw/international_results.csv",
        "pre_match_available":
            "Yes",
        "shared_with_task_2_1":
            "No",
    },
    {
        "variable": "previous_extra_time",
        "role": "Explanatory",
        "definition":
            "Shows whether the team's previous match went beyond "
            "the normal 90 minutes.",
        "calculation":
            "1 if the previous match went to extra time, otherwise 0",
        "unit_or_encoding":
            "0 = No, 1 = Yes",
        "source":
            "Historical international results and World Cup match data",
        "source_file":
            "data/raw/international_results.csv and "
            "data/raw/HIT40 dataset.csv",
        "pre_match_available":
            "Yes",
        "shared_with_task_2_1":
            "No",
    },
    {
        "variable": "lineup_changes",
        "role": "Explanatory",
        "definition":
            "Number of players in the current starting XI who did not "
            "start the team's previous match.",
        "calculation":
            "11 - number of starters retained from the previous match",
        "unit_or_encoding":
            "Count from 0 to 11",
        "source":
            "World Cup lineup data, Cornerflick and documented "
            "pre-tournament lineup sources",
        "source_file":
            "data/raw/lineups/",
        "pre_match_available":
            "Yes - after the starting XI is announced",
        "shared_with_task_2_1":
            "No",
    },
    {
        "variable": "qualification_pressure",
        "role": "Explanatory",
        "definition":
            "A simple measure used to represent greater qualification "
            "or elimination pressure.",
        "calculation":
            "1 for the third group match or any knockout match, "
            "otherwise 0",
        "unit_or_encoding":
            "0 = Lower pressure, 1 = Higher pressure",
        "source":
            "Derived from the World Cup match stage and match order",
        "source_file":
            "data/raw/HIT40 dataset.csv",
        "pre_match_available":
            "Yes",
        "shared_with_task_2_1":
            "No",
    },
]


dictionary = pd.DataFrame(variables)


# Basic checks
assert len(dictionary) == 9

assert (
    dictionary["role"] == "Response"
).sum() == 1

assert (
    dictionary["role"] == "Explanatory"
).sum() == 8

assert (
    (
        dictionary["role"] == "Explanatory"
    )
    &
    (
        dictionary["shared_with_task_2_1"]
        == "Yes"
    )
).sum() == 2

assert dictionary.notna().all().all()


dictionary.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Variable dictionary created")
print(
    "Documented variables:",
    len(dictionary)
)
print(
    "Shared with Task 2.1:",
    (
        (
            dictionary["role"] == "Explanatory"
        )
        &
        (
            dictionary["shared_with_task_2_1"] == "Yes"
        )
    ).sum()
)
print(
    "Saved to:",
    OUTPUT_FILE
)
