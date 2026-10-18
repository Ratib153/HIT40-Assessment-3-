"""
Builds the three composite figures used in the Objective 2 report.

Reads only the processed datasets and the CSV outputs written by the
Task 2.1 and Task 2.2 modelling scripts (run those first, including
Task 2.2/modelling/06_count_model_check.py).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
T1 = ROOT / "Task 2.1"
T2 = ROOT / "Task 2.2"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Palette (validated categorical slots 1-3) + neutral ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.labelcolor": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.fontsize": 6.8, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

P1 = ["fifa_points_diff", "market_value_log_ratio", "squad_age_diff", "rest_days_log_ratio",
      "host_diff", "travel_1000km_diff", "tournament_form_diff", "pre_wc_form_diff"]
P2 = ["fifa_points_diff", "rest_days", "venue_altitude_m", "attacking_form_5",
      "opponent_defensive_form_5", "previous_extra_time", "lineup_changes",
      "qualification_pressure"]
SHARED = {"fifa_points_diff", "rest_days_log_ratio", "rest_days"}

d1 = pd.read_csv(T1 / "data/processed/matches_104_model.csv")
d2 = pd.read_csv(T2 / "data/processed/team_matches_208_model.csv")


def grid_x(ax):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------
# Figure 1 - correlation screening, both tasks
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"wspace": 0.95})
for ax, df, preds, resp, title in [
    (axes[0], d1, P1, "goal_diff", "(a) Task 2.1: correlation with goal difference"),
    (axes[1], d2, P2, "goals_scored", "(b) Task 2.2: correlation with goals scored"),
]:
    r = df[preds].corrwith(df[resp]).sort_values()
    colors = [ORANGE if v in SHARED else BLUE for v in r.index]
    ax.barh(range(len(r)), r.values, color=colors, height=0.62)
    ax.set_yticks(range(len(r)), r.index)
    ax.axvline(0, color=MUTED, lw=0.6)
    for i, v in enumerate(r.values):
        ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.5, color=INK)
    ax.set_xlim(-0.4, 0.8)
    ax.set_xlabel("Pearson r (all rows)")
    ax.set_title(title)
    grid_x(ax)
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
fig.legend(handles, ["Unique to the task", "Shared between 2.1 and 2.2"],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16))
fig.savefig(OUT / "fig1_screening.png")
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2 - nested CV model comparison, both tasks
# ---------------------------------------------------------------------
f1 = pd.read_csv(T1 / "modelling/outputs/model_optimisation/nested_cv_folds.csv")

# Mean baseline on the same 2.1 outer splits (RepeatedKFold 5x3, seed 0, 83 training matches)
def band(g):
    return np.clip(g, -2, 2)
tr_idx, _ = train_test_split(np.arange(len(d1)), test_size=0.2, random_state=0,
                             stratify=band(d1["goal_diff"]))
y_tr = d1["goal_diff"].iloc[tr_idx].reset_index(drop=True)
base_rows = []
for k, (a, b) in enumerate(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(y_tr)):
    base_rows.append({"model": "Mean Baseline", "RMSE":
                      float(np.sqrt(((y_tr.iloc[b] - y_tr.iloc[a].mean()) ** 2).mean()))})
f1 = pd.concat([f1[["model", "RMSE"]], pd.DataFrame(base_rows)])

f2a = pd.read_csv(T2 / "modelling/outputs/model_optimisation/nested_cv_folds.csv")
f2b = pd.read_csv(T2 / "modelling/outputs/count_model_check/nested_cv_folds.csv")
f2 = pd.concat([f2a[["model", "RMSE"]],
                f2b[f2b.model != "Lasso Regression"][["model", "RMSE"]]])

short = {"Support Vector Regression (RBF)": "SVR (RBF)", "Polynomial (degree 2) + Ridge":
         "Polynomial-2 + Ridge", "Multiple Linear Regression": "MLR (class model)",
         "K-Nearest Neighbours": "k-NN", "Mean Baseline": "Mean baseline"}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"wspace": 0.75,
                                                                 "width_ratios": [1.15, 1]})
for ax, f, title, selected in [
    (axes[0], f1, "(a) Task 2.1: nested CV (5 folds x 3 repeats, 83 matches)", "Lasso Regression"),
    (axes[1], f2, "(b) Task 2.2: nested grouped CV (5 folds by match)", "Lasso Regression"),
]:
    s = f.groupby("model")["RMSE"].agg(["mean", "std", "count"])
    s["se"] = s["std"] / np.sqrt(s["count"])
    s = s.sort_values("mean", ascending=False)
    mlr = s.loc["Multiple Linear Regression", "mean"]
    for i, (m, row) in enumerate(s.iterrows()):
        c = ORANGE if m == selected else INK if m == "Multiple Linear Regression" else \
            MUTED if m == "Mean Baseline" else AQUA if m == "Poisson Regression" else BLUE
        ax.errorbar(row["mean"], i, xerr=row["se"], fmt="o", color=c, ms=4.2,
                    elinewidth=1.1, capsize=0)
        ax.text(row["mean"] + row["se"] + 0.012, i, f"{row['mean']:.3f}", va="center",
                fontsize=6.3, color=INK)
    ax.axvline(mlr, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(len(s)), [short.get(m, m.replace(" Regression", "")) for m in s.index])
    ax.set_xlabel("Out-of-fold RMSE (goals), mean +/- 1 SE; lower is better")
    ax.set_title(title, fontsize=7.8)
    grid_x(ax)
    lo, hi = (s["mean"] - s["se"]).min(), (s["mean"] + s["se"]).max()
    ax.set_xlim(lo - 0.03, hi + 0.12)
fig.savefig(OUT / "fig2_model_comparison.png")
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3 - final model behaviour
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6),
                         gridspec_kw={"hspace": 0.62, "wspace": 0.85})

# (a) 2.1 held-out test: actual vs predicted
tp = pd.read_csv(T1 / "modelling/outputs/final_model/test_predictions.csv")
tperf = pd.read_csv(T1 / "modelling/outputs/final_model/test_set_performance.csv").set_index("model")
ax = axes[0, 0]
lim = (-4.5, 5.5)
ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)))
for col, c, lab, key in [("Multiple Linear Regression", BLUE, "MLR", "Multiple Linear Regression"),
                         ("Lasso Regression (selected)", ORANGE, "Lasso", "Lasso Regression (selected)")]:
    ax.scatter(tp["actual_goal_diff"], tp[col], s=16, color=c, edgecolor="white", lw=0.6,
               label=f"{lab}: RMSE {tperf.loc[key, 'RMSE']:.2f}, R² {tperf.loc[key, 'R2']:.2f}",
               zorder=3)
ax.set_xlim(lim); ax.set_ylim(-4.5, 7.2)
ax.set_xlabel("Actual goal difference"); ax.set_ylabel("Predicted goal difference")
ax.set_title("(a) Task 2.1: unseen 21-match test set")
ax.legend(loc="upper left", handletextpad=0.2)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)


def per_sd_effects(df, preds, resp):
    X = sm.add_constant(df[preds])
    ols = sm.OLS(df[resp], X).fit()
    return pd.Series({p: ols.params[p] * df[p].std() for p in preds})


def dot_compare(ax, series, labels, colors, order, title, xlabel):
    n = len(series)
    offs = np.linspace(-0.22, 0.22, n) if n > 1 else [0]
    for s, lab, c, o in zip(series, labels, colors, offs):
        vals = s.reindex(order).values
        ax.scatter(vals, np.arange(len(order)) + o, s=15, color=c, label=lab, zorder=3,
                   edgecolor="white", lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(order)), order)
    ax.set_title(title); ax.set_xlabel(xlabel)
    grid_x(ax)
    ax.legend(loc="lower right", handletextpad=0.1)


# (b) 2.1 effect sizes: MLR vs Lasso (both fitted to all 104 matches)
lasso1 = pd.read_csv(T1 / "modelling/outputs/final_model/final_coefficients.csv").set_index("predictor")
lasso1 = lasso1.drop("intercept")["standardised_coefficient"]
mlr1 = per_sd_effects(d1, P1, "goal_diff")
order1 = mlr1.abs().sort_values().index.tolist()
dot_compare(axes[0, 1], [mlr1, lasso1], ["MLR", "Lasso (selected)"], [BLUE, ORANGE], order1,
            "(b) Task 2.1: effect of +1 SD", "Change in goal difference per +1 SD")

# (c) 2.2 effect sizes: MLR, Lasso, Poisson average marginal effect
lasso2 = pd.read_csv(T2 / "modelling/outputs/final_model/final_lasso_coefficients.csv").set_index("predictor")["standardised_coefficient"]
mlr2 = per_sd_effects(d2, P2, "goals_scored")
glm = pd.read_csv(T2 / "modelling/outputs/count_model_check/poisson_glm_coefficients.csv").set_index("variable")
pois2 = pd.Series({p: glm.loc[p, "coefficient_log_scale"] * d2[p].std() * d2["goals_scored"].mean()
                   for p in P2})
order2 = mlr2.abs().sort_values().index.tolist()
dot_compare(axes[1, 0], [mlr2, lasso2, pois2], ["MLR", "Lasso (selected)", "Poisson (avg. effect)"],
            [BLUE, ORANGE, AQUA], order2, "(c) Task 2.2: effect of +1 SD",
            "Change in goals scored per +1 SD")

# (d) 2.2 goal-count distribution: observed vs Poisson out-of-fold
gc = pd.read_csv(T2 / "modelling/outputs/count_model_check/goal_count_fit.csv")
ax = axes[1, 1]
x = np.arange(len(gc)); w = 0.38
ax.bar(x - w / 2 - 0.01, gc["observed_share"], w, color=BLUE, label="Observed")
ax.bar(x + w / 2 + 0.01, gc["poisson_expected_share"], w, color=AQUA, label="Poisson (out-of-fold)")
ax.set_xticks(x, gc["goals"])
ax.set_xlabel("Goals scored by a team in a match"); ax.set_ylabel("Share")
ax.set_title("(d) Task 2.2: goal counts vs Poisson fit")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="upper right")
fig.savefig(OUT / "fig3_final_models.png")
plt.close(fig)

print("Figures written to", OUT)
print("2.1 baseline nested RMSE:", pd.DataFrame(base_rows)["RMSE"].mean())
print("MLR per-SD 2.1:\n", mlr1.round(3), "\nMLR per-SD 2.2:\n", mlr2.round(3), "\nPoisson AME per-SD:\n", pois2.round(3))
