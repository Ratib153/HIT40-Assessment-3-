# HIT140 Assessment 3
## FIFA World Cup 2026 Data Analysis

This repository contains the data preparation and modelling work for HIT140 Assessment 3.




This section of the project focuses on **Task 2.2**, where the aim is to predict the number of goals scored by a team in a FIFA World Cup 2026 match.

## Task 2.2 Dataset

The final dataset contains:

- 208 observations
- 104 matches
- 48 teams
- 1 response variable
- 8 starting explanatory variables

Each row represents one team in one match. Therefore, every match appears twice, once from the perspective of each team.

### Response Variable

`goals_scored`

The number of goals scored by the team in the match. Penalty shootout kicks are not included.

### Explanatory Variables

1. `fifa_points_diff`  
   Difference between the team's FIFA ranking points and the opponent's FIFA ranking points.

2. `rest_days`  
   Number of days since the team's previous senior international match.

3. `venue_altitude_m`  
   Altitude of the match venue in metres above sea level.

4. `attacking_form_5`  
   Average goals scored by the team in its previous five senior international matches.

5. `opponent_defensive_form_5`  
   Average goals conceded by the opponent in its previous five senior international matches.

6. `previous_extra_time`  
   Indicates whether the team's previous match went beyond the normal 90 minutes.

7. `lineup_changes`  
   Number of players in the current starting XI who did not start the team's previous match. This feature is available once the official starting lineup has been announced before kickoff.

8. `qualification_pressure`  
   A pre-match proxy for increased qualification or elimination pressure. It is coded as 1 for a team's third group-stage match or any knockout match, and 0 for the first two group-stage matches.

The three variables shared with Task 2.1 are:

`fifa_points_diff`, `rest_days`, and `venue_altitude_m`.

## Project Structure

```text
HIT40 Assessment 3/
│
├── data/
│   ├── raw/
│   │   ├── HIT40 dataset.csv
│   │   ├── fifa_rankings_2026_06_11.csv
│   │   ├── international_results.csv
│   │   ├── venue_altitudes.csv
│   │   └── lineups/
│   │
│   └── processed/
│       ├── team_matches_208_model.csv
│       ├── final_dataset_audit.csv
│       └── variable_dictionary.csv
│
├── scripts/
│   ├── 01_prepare_matches.py
│   ├── 02_create_team_matches.py
│   ├── 03_add_fifa_strength.py
│   ├── 04_add_venue_altitude.py
│   ├── 05_add_historical_features.py
│   ├── 06_prepare_lineup_data.py
│   ├── 07_combine_pre_world_cup_starters.py
│   ├── 08_add_lineup_changes.py
│   ├── 09_add_qualification_pressure.py
│   ├── 10_final_dataset_audit.py
│   └── 11_dataset_documentation.py
│
├── modelling/
├── .gitignore
├── requirements.txt
└── README.md
