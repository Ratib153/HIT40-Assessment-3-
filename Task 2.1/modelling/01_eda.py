import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from settings import PREDICTORS, RESPONSE, holdout_split, load_data, output_dir


OUTPUT_DIR = output_dir("eda")

df = load_data()
train_index, test_index = holdout_split(df)
train = df.iloc[train_index]
test = df.iloc[test_index]

# Exploration uses the 83 training matches only, so nothing learned here
# (correlations, transformations, collinearity) comes from the test set.
required_columns = [RESPONSE] + PREDICTORS
continuous_predictors = [p for p in PREDICTORS if p != "host_diff"]


# Descriptive statistics
summary = train[required_columns].describe().T
summary["median"] = train[required_columns].median()
summary["skewness"] = train[required_columns].skew()
summary.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")


# Pearson and Spearman correlation
# Spearman is rank-based, so it checks whether a relationship exists even if it is not straight-line
correlation = train[required_columns].corr()
correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

target_correlation = pd.DataFrame({
    "pearson_r": correlation[RESPONSE].drop(RESPONSE),
    "spearman_rho": train[required_columns].corr(method="spearman")[RESPONSE].drop(RESPONSE),
})
target_correlation["pearson_p_value"] = [stats.pearsonr(train[p], train[RESPONSE])[1] for p in PREDICTORS]
target_correlation = target_correlation.reindex(target_correlation["pearson_r"].abs().sort_values(ascending=False).index)
target_correlation.to_csv(OUTPUT_DIR / "correlation_with_goal_diff.csv")


# Collinearity between predictors (week 9): pairs with |r| >= 0.70 and VIF
predictor_correlation = train[PREDICTORS].corr().abs()
strong_pairs = []
for i in range(len(PREDICTORS)):
    for j in range(i + 1, len(PREDICTORS)):
        value = predictor_correlation.iloc[i, j]
        if value >= 0.70:
            strong_pairs.append({
                "predictor_1": PREDICTORS[i],
                "predictor_2": PREDICTORS[j],
                "absolute_correlation": value,
            })

strong_correlations = pd.DataFrame(strong_pairs, columns=["predictor_1", "predictor_2", "absolute_correlation"])
strong_correlations.to_csv(OUTPUT_DIR / "strong_predictor_correlations.csv", index=False)

# VIF above 5 is a common warning level and above 10 a serious one
vif_data = add_constant(train[PREDICTORS])
vif = pd.DataFrame({
    "predictor": PREDICTORS,
    "VIF": [variance_inflation_factor(vif_data.values, i + 1) for i in range(len(PREDICTORS))],
}).sort_values("VIF", ascending=False)
vif.to_csv(OUTPUT_DIR / "variance_inflation_factors.csv", index=False)


# Check the held-out test set looks like the training set
# (Kolmogorov-Smirnov test: a small p-value would mean the two samples differ)
split_rows = []
for column in required_columns:
    ks = stats.ks_2samp(train[column], test[column])
    split_rows.append({
        "variable": column,
        "train_mean": train[column].mean(),
        "test_mean": test[column].mean(),
        "train_std": train[column].std(),
        "test_std": test[column].std(),
        "ks_statistic": ks.statistic,
        "ks_p_value": ks.pvalue,
    })
split_check = pd.DataFrame(split_rows)
split_check.to_csv(OUTPUT_DIR / "train_test_split_check.csv", index=False)


# Distribution of goal difference
plt.figure(figsize=(7, 5))
low, high = int(df[RESPONSE].min()), int(df[RESPONSE].max())
plt.hist([train[RESPONSE], test[RESPONSE]], bins=range(low, high + 2), align="left",
         edgecolor="black", stacked=True, label=["Training (83)", "Test (21)"])
plt.xlabel("Goal Difference (Team A - Team B)")
plt.ylabel("Number of Matches")
plt.title("Distribution of Goal Difference")
plt.xticks(range(low, high + 1))
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "goal_diff_distribution.png", dpi=300)
plt.close()


# Predictor distributions
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()
for index, predictor in enumerate(PREDICTORS):
    axes[index].hist(train[predictor], bins=12, edgecolor="black")
    axes[index].set_title(predictor)
    axes[index].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predictor_distributions.png", dpi=300)
plt.close()


# Correlation heatmap
plt.figure(figsize=(11, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0)
plt.title("Correlation Matrix (Training Matches)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()


# Continuous predictors against goal difference, with a straight-line fit
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for index, predictor in enumerate(continuous_predictors):
    sns.regplot(data=train, x=predictor, y=RESPONSE, ax=axes[index], ci=None,
                scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    axes[index].set_ylabel("Goal Difference")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "goal_diff_vs_continuous_predictors.png", dpi=300)
plt.close()


# Host nation indicator against goal difference
plt.figure(figsize=(6, 5))
sns.boxplot(data=train, x="host_diff", y=RESPONSE)
plt.xlabel("host_diff (-1 = team B is host, 0 = no host, 1 = team A is host)")
plt.ylabel("Goal Difference")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "goal_diff_by_host.png", dpi=300)
plt.close()


# Print useful results
print("EDA complete (training matches only)")
print("Training matches:", len(train), " Test matches:", len(test))
print("\nCorrelation with goal difference:")
print(target_correlation.round(3).to_string())
print("\nVariance inflation factors:")
print(vif.round(2).to_string(index=False))

if strong_correlations.empty:
    print("\nNo predictor pairs have |correlation| >= 0.70")
else:
    print("\nPredictor pairs with |correlation| >= 0.70:")
    print(strong_correlations.round(3).to_string(index=False))

print("\nSmallest KS p-value between training and test sets: %.3f" % split_check["ks_p_value"].min())
print("\nEDA outputs saved to:", OUTPUT_DIR)
