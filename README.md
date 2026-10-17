# HIT140 Assessment 3

## FIFA World Cup 2026 Data Analysis

This repository contains the data preparation work for HIT140 Assessment 3.

The work documented here focuses on **Task 2.2**, where the objective is to prepare a dataset for predicting the number of goals scored by a team in a FIFA World Cup 2026 match.

## Task 2.2 Dataset

The final dataset contains:

- 208 observations
- 104 matches
- 48 teams
- 1 response variable
- 8 starting explanatory variables

Each row represents one team in one match. Therefore, every match appears twice, once from the perspective of each team.

## Response Variable

### `goals_scored`

The number of goals scored by the team in the match.

Penalty shootout kicks are not included.

## Explanatory Variables

The dataset contains exactly eight starting explanatory variables.

### 1. `fifa_points_diff`

Difference between the team's FIFA ranking points and the opponent's FIFA ranking points.

Calculated as:

```text
Team FIFA points - Opponent FIFA points
```

The ranking values are based on the FIFA Men's World Ranking released on 11 June 2026, before the tournament.

### 2. `rest_days`

Number of calendar days between the team's current match and its immediately previous senior international match.

### 3. `venue_altitude_m`

Altitude of the match venue in metres above sea level.

### 4. `attacking_form_5`

Average number of goals scored by the team across its previous five senior international matches.

The current match is not included.

### 5. `opponent_defensive_form_5`

Average number of goals conceded by the opponent across its previous five senior international matches.

The current match is not included.

### 6. `previous_extra_time`

Indicates whether the team's immediately previous match went beyond the normal 90 minutes.

```text
0 = No
1 = Yes
```

### 7. `lineup_changes`

Number of players in the current starting XI who did not start the team's previous match.

It is calculated as:

```text
11 - number of retained starters
```

The value ranges from 0 to 11.

This variable is available before kickoff once the official starting lineup has been announced.

### 8. `qualification_pressure`

A simple pre-match proxy for increased qualification or elimination pressure.

```text
0 = First or second group-stage match
1 = Third group-stage match or knockout match
```

This variable represents the stage of the tournament rather than directly measuring player psychology.

## Variables Shared With Task 2.1

Three explanatory variables are shared with Task 2.1:

```text
fifa_points_diff
rest_days
venue_altitude_m
```

## Project Structure

```text
project/
│
├── data/
│   │
│   ├── raw/
│   │   ├── HIT40 dataset.csv
│   │   ├── fifa_rankings_2026_06_11.csv
│   │   ├── international_results.csv
│   │   ├── venue_altitudes.csv
│   │   │
│   │   └── lineups/
│   │       ├── match_appearances.csv
│   │       ├── matches.csv
│   │       ├── pre_world_cup_starters_fallback.csv
│   │       │
│   │       └── pre_world_cup_lineups/
│   │           └── Pre-tournament lineup CSV files
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
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Preparation

All data cleaning, transformation and feature engineering are completed in Python.

The scripts are numbered in the order they should be run.

### Step 1: Prepare World Cup Matches

```bash
python scripts/01_prepare_matches.py
```

Cleans the original World Cup match data and creates one row for each of the 104 matches.

### Step 2: Create Team-Match Observations

```bash
python scripts/02_create_team_matches.py
```

Converts each match into two observations, one for each team.

This creates the required 208 team-match observations.

### Step 3: Add FIFA Strength

```bash
python scripts/03_add_fifa_strength.py
```

Adds pre-tournament FIFA ranking information and calculates `fifa_points_diff`.

### Step 4: Add Venue Altitude

```bash
python scripts/04_add_venue_altitude.py
```

Matches each World Cup venue with its altitude and adds `venue_altitude_m`.

### Step 5: Add Historical Features

```bash
python scripts/05_add_historical_features.py
```

Uses matches played before each World Cup match to calculate:

```text
rest_days
attacking_form_5
opponent_defensive_form_5
previous_extra_time
```

Only matches occurring before the current World Cup match are used.

### Step 6: Prepare World Cup Lineup Data

```bash
python scripts/06_prepare_lineup_data.py
```

Extracts the World Cup starting lineups and identifies the pre-tournament matches required for each team's first World Cup lineup comparison.

### Step 7: Prepare Pre-Tournament Starters

```bash
python scripts/07_combine_pre_world_cup_starters.py
```

Combines the available pre-tournament lineup files with the supplementary fallback starter data.

The resulting dataset contains 11 previous-match starters for each of the 48 World Cup teams.

### Step 8: Calculate Lineup Changes

```bash
python scripts/08_add_lineup_changes.py
```

Calculates the number of changes between the current starting XI and the team's previous starting XI.

For a team's first World Cup match, player names are matched against the pre-tournament lineup.

For later World Cup matches, player IDs from the World Cup lineup data are used.

### Step 9: Add Qualification Pressure

```bash
python scripts/09_add_qualification_pressure.py
```

Creates the `qualification_pressure` variable and produces the final Task 2.2 dataset containing exactly eight starting explanatory variables.

### Step 10: Audit the Final Dataset

```bash
python scripts/10_final_dataset_audit.py
```

Runs final checks on the completed dataset.

### Step 11: Create the Variable Dictionary

```bash
python scripts/11_dataset_documentation.py
```

Creates a variable dictionary containing the definition, calculation, source and role of each variable.

## Final Dataset

The final Task 2.2 dataset is:

```text
data/processed/team_matches_208_model.csv
```

Its columns are:

```text
match_id
date
team
opponent
goals_scored
fifa_points_diff
rest_days
venue_altitude_m
attacking_form_5
opponent_defensive_form_5
previous_extra_time
lineup_changes
qualification_pressure
```

The first four columns are identifiers or contextual information:

```text
match_id
date
team
opponent
```

They are not included as explanatory variables.

The response variable is:

```text
goals_scored
```

The remaining eight columns are the starting explanatory variables.

## Dataset Validation

The final dataset is checked automatically by:

```text
scripts/10_final_dataset_audit.py
```

The checks include:

- 208 observations
- 104 unique matches
- 48 teams
- exactly 8 starting explanatory variables
- no missing predictor values
- no duplicate rows
- exactly two observations for each match
- valid team and opponent pairs
- non-negative goal values
- valid binary variables
- `lineup_changes` between 0 and 11
- matching venue altitude for both teams in the same match
- mirrored FIFA points differences between opposing teams
- removal of helper columns that are not part of the final dataset

The final audit has passed with:

```text
Rows: 208
Unique matches: 104
Unique teams: 48
Starting predictors: 8
Missing predictor values: 0
```

A numerical summary of the final explanatory variables is stored in:

```text
data/processed/final_dataset_audit.csv
```

## Variable Dictionary

Definitions and details for the response and explanatory variables are stored in:

```text
data/processed/variable_dictionary.csv
```

The dictionary records:

- variable name
- variable role
- definition
- calculation
- unit or encoding
- source
- source file
- whether the information is available before the match
- whether the variable is shared with Task 2.1

## Data Sources

The project uses several data sources because different information is required to create the explanatory variables.

### FIFA World Cup Match Data

The original World Cup match dataset is stored as:

```text
data/raw/HIT40 dataset.csv
```

It provides the tournament matches, teams, scores, rounds and venues used to create the 104-match and 208-row datasets.

### FIFA Men's World Ranking

Pre-tournament FIFA ranking data is stored as:

```text
data/raw/fifa_rankings_2026_06_11.csv
```

The FIFA Men's World Ranking released on 11 June 2026 is used so that the ranking information is available before the World Cup matches.

### Historical International Results

Historical senior international match results are stored as:

```text
data/raw/international_results.csv
```

This data is used to calculate:

```text
rest_days
attacking_form_5
opponent_defensive_form_5
previous_extra_time
```

Historical match information is based on the martj42 international football results dataset.

### Venue Altitude

Venue altitude information is stored as:

```text
data/raw/venue_altitudes.csv
```

This file contains the altitude in metres for the World Cup venues used in the project.

### World Cup Starting Lineups

World Cup lineup information is stored under:

```text
data/raw/lineups/
```

The main files used are:

```text
match_appearances.csv
matches.csv
```

These files provide the World Cup starting players and player IDs.

### Pre-Tournament Starting Lineups

Pre-tournament lineup files are stored in:

```text
data/raw/lineups/pre_world_cup_lineups/
```

These files are used to identify the starters from each team's immediately previous match before the World Cup where the data was available.

Additional starter records required to complete the 48 teams are stored in:

```text
data/raw/lineups/pre_world_cup_starters_fallback.csv
```

These records are combined with the available pre-tournament lineup files before the `lineup_changes` variable is calculated.

## Intermediate Files

The scripts create several intermediate CSV files while the dataset is being prepared.

These files are useful for checking each stage of the data preparation process but are not required as final outputs.

The `.gitignore` file excludes generated intermediate files while keeping the main completed files:

```text
team_matches_208_model.csv
final_dataset_audit.csv
variable_dictionary.csv
```

## Python Requirements

The current data preparation pipeline requires Python and pandas.

Install the required package using:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
pandas
```
