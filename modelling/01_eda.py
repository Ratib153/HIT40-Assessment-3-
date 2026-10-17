from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# File locations
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "team_matches_208_model.csv"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "eda"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Variables
response = "goals_scored"

predictors = [
    "fifa_points_diff",
    "rest_days",
    "venue_altitude_m",
    "attacking_form_5",
    "opponent_defensive_form_5",
    "previous_extra_time",
    "lineup_changes",
    "qualification_pressure",
]

continuous_predictors = [
    "fifa_points_diff",
    "rest_days",
    "venue_altitude_m",
    "attacking_form_5",
    "opponent_defensive_form_5",
    "lineup_changes",
]

binary_predictors = [
    "previous_extra_time",
    "qualification_pressure",
]


# Load dataset
df = pd.read_csv(DATA_FILE)

required_columns = [
    response
] + predictors

assert len(df) == 208
assert len(predictors) == 8
assert df[required_columns].notna().all().all()


# Descriptive statistics
summary = (
    df[required_columns]
    .describe()
    .T
)

summary["median"] = (
    df[required_columns]
    .median()
)

summary.to_csv(
    OUTPUT_DIR
    / "descriptive_statistics.csv"
)


# Correlation matrix
correlation = (
    df[required_columns]
    .corr()
)

correlation.to_csv(
    OUTPUT_DIR
    / "correlation_matrix.csv"
)


# Check strong correlations between predictors
predictor_correlation = (
    df[predictors]
    .corr()
    .abs()
)

strong_pairs = []

for i in range(len(predictors)):
    for j in range(i + 1, len(predictors)):

        value = predictor_correlation.iloc[
            i,
            j
        ]

        if value >= 0.70:
            strong_pairs.append({
                "predictor_1": predictors[i],
                "predictor_2": predictors[j],
                "absolute_correlation": value,
            })


strong_correlations = pd.DataFrame(
    strong_pairs,
    columns=[
        "predictor_1",
        "predictor_2",
        "absolute_correlation",
    ]
)

strong_correlations.to_csv(
    OUTPUT_DIR
    / "strong_predictor_correlations.csv",
    index=False
)


# Distribution of goals scored
plt.figure(
    figsize=(7, 5)
)

max_goals = int(
    df[response].max()
)

plt.hist(
    df[response],
    bins=range(
        0,
        max_goals + 2
    ),
    align="left",
    edgecolor="black"
)

plt.xlabel(
    "Goals Scored"
)

plt.ylabel(
    "Number of Team-Match Observations"
)

plt.title(
    "Distribution of Goals Scored"
)

plt.xticks(
    range(
        0,
        max_goals + 1
    )
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "goals_distribution.png",
    dpi=300
)

plt.close()


# Predictor distributions
fig, axes = plt.subplots(
    2,
    4,
    figsize=(14, 7)
)

axes = axes.flatten()

for index, predictor in enumerate(
    predictors
):

    axes[index].hist(
        df[predictor],
        bins=12,
        edgecolor="black"
    )

    axes[index].set_title(
        predictor
    )

    axes[index].set_ylabel(
        "Frequency"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "predictor_distributions.png",
    dpi=300
)

plt.close()


# Correlation heatmap
plt.figure(
    figsize=(11, 8)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0
)

plt.title(
    "Correlation Matrix"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "correlation_heatmap.png",
    dpi=300
)

plt.close()


# Continuous predictors against goals scored
fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8)
)

axes = axes.flatten()

for index, predictor in enumerate(
    continuous_predictors
):

    axes[index].scatter(
        df[predictor],
        df[response],
        alpha=0.6
    )

    axes[index].set_xlabel(
        predictor
    )

    axes[index].set_ylabel(
        "Goals Scored"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "goals_vs_continuous_predictors.png",
    dpi=300
)

plt.close()


# Binary predictors against goals scored
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 5)
)

for index, predictor in enumerate(
    binary_predictors
):

    sns.boxplot(
        data=df,
        x=predictor,
        y=response,
        ax=axes[index]
    )

    axes[index].set_xlabel(
        predictor
    )

    axes[index].set_ylabel(
        "Goals Scored"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "goals_by_binary_predictors.png",
    dpi=300
)

plt.close()


# Print useful results
print("EDA complete")

print(
    "Rows:",
    len(df)
)

print(
    "Starting predictors:",
    len(predictors)
)

print(
    "\nGoals scored summary:"
)

print(
    df[response].describe()
)

print(
    "\nCorrelation with goals scored:"
)

print(
    correlation[response]
    .drop(response)
    .sort_values(
        key=abs,
        ascending=False
    )
)

if strong_correlations.empty:
    print(
        "\nNo predictor pairs have "
        "|correlation| >= 0.70"
    )
else:
    print(
        "\nPredictor pairs with "
        "|correlation| >= 0.70:"
    )

    print(
        strong_correlations.to_string(
            index=False
        )
    )

print(
    "\nEDA outputs saved to:",
    OUTPUT_DIR
)