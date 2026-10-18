# Task 2.1 - Predicting match goal difference

Linear regression of `goal_diff` (team A goals minus team B goals) for the 104
FIFA World Cup 2026 matches, using 8 explanatory variables that are all known
before kick-off.

## How to run

Tested with Python 3.13 (works with 3.11 or newer). From inside the `Task 2.1` folder:

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                  # about 6 minutes
```

`python run_all.py --data-only` rebuilds only the dataset.
`python run_all.py --refresh` also re-queries OpenStreetMap for the stadium
locations (needs internet; otherwise the saved results in
`data/raw/venue_geocodes_osm.csv` are used).

Each script can also be run on its own, in number order, for example
`python scripts/01_prepare_matches.py` or `python modelling/03_model_comparison.py`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Clean results | `scripts/01_prepare_matches.py` | 104 matches, goal_diff |
| FIFA points | `scripts/02_add_fifa_strength.py` | fifa_points_diff |
| Market value, squad age, Elo | `scripts/03_add_market_value_age_elo.py` | market_value_log_ratio, squad_age_diff |
| Travel and host | `scripts/04_add_travel_and_host.py` | travel_1000km_diff, host_diff |
| History | `scripts/05_add_historical_features.py` | rest_days_log_ratio, pre_wc_form_diff, tournament_form_diff |
| Model dataset | `scripts/06_create_model_dataset.py` | `data/processed/matches_104_model.csv` |
| Audit | `scripts/07_final_dataset_audit.py` | structural checks |
| Dictionary | `scripts/08_dataset_documentation.py` | `data/processed/variable_dictionary.csv` |
| EDA | `modelling/01_eda.py` | correlations, VIF, split check |
| MLR | `modelling/02_mlr_model.py` | train/test result, OLS inference, assumption tests |
| Comparison | `modelling/03_model_comparison.py` | 14 models, repeated 5-fold CV |
| Tuning | `modelling/04_model_optimisation.py` | nested CV, model selection |
| Final model | `modelling/05_final_model.py` | one test evaluation, bootstrap intervals |
| Variable evidence | `modelling/06_variable_selection.py` | specification checks, best subsets, Lasso path |
| Swapped teams | `modelling/07_swapped_teams_check.py` | listing-order effect, symmetric models |

## Validation design

1. 21 matches (20%, stratified by result) are held out as the test set.
2. All comparison, tuning and variable checks use repeated 5-fold
   cross-validation on the 83 training matches (the validation folds).
3. Hyperparameters are tuned inside nested cross-validation.
4. The model is chosen with the one-standard-error rule, then evaluated once
   on the test set.
5. A time-based check trains on the group stage and tests on the knockouts.
6. A swapped-teams check tests whether the listing order of the two teams
   affects the predictions.

Shared settings (variables, split, folds) are in `modelling/settings.py`.
