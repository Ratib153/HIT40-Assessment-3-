// Builds the Objective 2 report (.docx). Run: node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, ShadingType, VerticalAlign,
} = require("docx");

const FONT = "Calibri";
const SZ = 20;          // 10 pt
const SMALL = 16;       // 8 pt (tables, captions)
const PAGE_W = 11906, MARGIN = 1020;          // A4, 1.8 cm margins
const CONTENT_W = PAGE_W - 2 * MARGIN;        // 9866 DXA
const FIG = (f) => path.join(__dirname, "figures", f);

// Inline markup: **bold**, *italic*
function runs(text, size = SZ, extra = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size, ...extra }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size, ...extra }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, size, ...extra }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size, ...extra }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 70, line: 240 }, ...opts,
});
const H = (text) => new Paragraph({
  children: [new TextRun({ text, bold: true, font: FONT, size: 22, color: "1F3864" })],
  spacing: { before: 110, after: 50 }, keepNext: true,
});
const H2 = (text) => new Paragraph({
  children: [new TextRun({ text, bold: true, italics: true, font: FONT, size: SZ, color: "1F3864" })],
  spacing: { before: 60, after: 30 }, keepNext: true,
});
const CAPTION = (text) => new Paragraph({
  children: runs(text, SMALL, { color: "404040" }), alignment: AlignmentType.JUSTIFIED,
  spacing: { before: 20, after: 90 },
});

function figure(file, _w, _h, widthDxa = CONTENT_W) {
  const buf = fs.readFileSync(FIG(file));
  const wPx = buf.readUInt32BE(16), hPx = buf.readUInt32BE(20);
  const wPt = widthDxa / 20;              // points
  const widthPx = Math.round(wPt * 96 / 72);
  const heightPx = Math.round(widthPx * hPx / wPx);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 40, after: 0 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG(file)),
      transformation: { width: widthPx, height: heightPx } })],
  });
}

// ---------- tables ----------
const thin = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function cell(text, width, { header = false, fill = null, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    margins: { top: 15, bottom: 15, left: 60, right: 60 },
    verticalAlign: VerticalAlign.CENTER,
    shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill } : undefined,
    borders: { top: thin, bottom: thin, left: none, right: none },
    children: [new Paragraph({ alignment: align, spacing: { after: 0, line: 216 },
      children: runs(text, SMALL, header ? { bold: true } : {}) })],
  });
}
function table(widths, header, rows, fills = []) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) =>
        cell(h, widths[i], { header: true, fill: "DCE6F2", align: i > 1 ? AlignmentType.CENTER : AlignmentType.LEFT })) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) =>
        cell(c, widths[i], { fill: fills[ri] || null, align: i > 1 && widths.length > 4 ? AlignmentType.CENTER : AlignmentType.LEFT })) })),
    ],
  });
}

// Table 1: variables
const T1W = [2050, 560, 4000, 3256];
const t1rows = [
  ["**Task 2.1: response** goal_diff", "", "Team A goals minus team B goals (extra time counted, shoot-outs not).", "Range -4 to 6, mean 0.48, 23% draws."],
  ["fifa_points_diff", "Both", "Team A minus team B FIFA ranking points (release of 11 June 2026, the day before kick-off).", "Official, published strength measure."],
  ["market_value_log_ratio", "2.1", "ln(team A / team B Transfermarkt squad value, 9 June 2026).", "Values span about 75x (EUR 20m to 1,523m); a ratio treats 'twice as valuable' the same at any level."],
  ["squad_age_diff", "2.1", "Difference in average squad age (FIFA squad lists, 4 June).", "Experience vs physical freshness."],
  ["rest_days_log_ratio", "Both*", "ln(team A / team B days since previous senior international).", "Raw days skewed (Uruguay rested 76 days); log ratio cut CV RMSE by 0.22 vs raw difference."],
  ["host_diff", "2.1", "+1 / -1 if team A / team B is a host nation playing in its own country.", "Home crowd, no travel, familiar climate."],
  ["travel_1000km_diff", "2.1", "Difference in great-circle distance, capital city to stadium (OSM, GeoNames).", "Proxy for travel fatigue and fan presence."],
  ["tournament_form_diff", "2.1", "Difference in mean goal difference in each team's earlier 2026 World Cup matches (0 before first game).", "Only earlier matches used, so known pre-match."],
  ["pre_wc_form_diff", "2.1", "Difference in mean goal difference in 2026 internationals before 11 June.", "Recent form entering the tournament."],
  ["**Task 2.2: response** goals_scored", "", "Goals scored by one team in one match (208 rows = 104 matches x 2 perspectives).", "Count 0 to 7, mean 1.48, variance 1.94, 26% zeros."],
  ["rest_days", "Both*", "Days since the team's previous senior international.", "Fatigue vs rhythm."],
  ["venue_altitude_m", "2.2", "Stadium elevation (OSM coordinates submitted to Open-Elevation SRTM).", "Thin air at Mexico City (2,281 m) and Guadalajara (1,671 m)."],
  ["attacking_form_5", "2.2", "Team's mean goals scored in its previous five senior internationals.", "Attacking quality, current match excluded."],
  ["opponent_defensive_form_5", "2.2", "Opponent's mean goals conceded in its previous five internationals.", "Defensive weakness of the opponent."],
  ["previous_extra_time", "2.2", "1 if the team's previous match went past 90 minutes.", "Accumulated fatigue (8 of 208 rows)."],
  ["lineup_changes", "2.2", "Starters not in the previous starting XI (0-11); FIFA match reports, Cornerflick and federation line-ups.", "Rotation; known once the XI is announced before kick-off."],
  ["qualification_pressure", "2.2", "1 for the third group match or any knockout match.", "Stage proxy for risk-taking."],
];
const t1fills = t1rows.map((r) => (r[0].startsWith("**Task") ? "F2F2F2" : null));

// Table 2: headline performance
const T2W = [2266, 1300, 1300, 1150, 1950, 1900];
const t2rows = [
  ["Mean baseline", "1.976", "2.196 (0.00)", "1.621", "1.394 (-0.06)", "1.127"],
  ["Multiple linear regression", "1.640", "**1.340 (0.63)**", "1.381", "1.249 (0.14)", "0.956"],
  ["Lasso (selected)", "**1.557**", "1.497 (0.53)", "**1.329**", "**1.226 (0.18)**", "**0.941**"],
  ["Poisson regression", "n/a", "n/a", "n/a", "1.245 (0.15)", "0.955"],
  ["Best ensemble", "1.605 (ET)", "1.626 (0.45) (ET)", "n/a", "1.258 (0.13) (GB)", "0.965 (GB)"],
];

// ---------- content ----------
const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: [new TextRun({ text: "Predicting FIFA World Cup 2026 Results with Linear Regression", bold: true, font: FONT, size: 30, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: runs("HIT140 Foundations of Data Science, Assessment 3, Objective 2 (Linear Regression 2.1 and 2.2)", SZ, { italics: true }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1F3864", space: 4 } },
    children: runs("[Student Name 1] (S000000)  |  [Student Name 2] (S000000)  |  [Student Name 3] (S000000)  |  [Student Name 4] (S000000)", 18) }),

  H("1. Introduction"),
  P("Objective 2 asks two related questions about the 2026 World Cup: can information known before kick-off predict the **goal difference** of a match (Task 2.1, 104 rows), and the **number of goals a team scores** (Task 2.2, 208 team-match rows)? Each dataset has exactly eight explanatory variables, and only two are shared (FIFA points and rest days, well below the limit of four). We built the multiple linear regression (MLR) taught in Weeks 8-9 for each task and evaluated it on a held-out test set. We then tested it, under the same validation design, against eleven competing scikit-learn regressors: regularised linear, robust, kernel, nearest-neighbour, tree and ensemble models, plus a Poisson count model for Task 2.2. Our story has three findings. First, **pre-match team strength carries almost all of the predictable signal**. Second, a sparse **Lasso** regression is the best or joint-best model in both tasks, and no non-linear or ensemble method beats it. Third, Task 2.2 is limited less by the model than by the randomness of goals: our best model is close to the error floor that pure Poisson chance allows."),

  H("2. Deriving the explanatory variables"),
  P("We chose the variables from football reasoning, grouped into four themes: **strength** (rankings, squad value), **form** (recent scoring and conceding), **fatigue** (rest, extra time, rotation, travel) and **context** (hosting, altitude, tournament stage). Every value had to be knowable before the match. Form windows use only matches strictly before the fixture, rankings come from the release of 11 June 2026 (not the post-tournament update), and line-up changes use the starting XI that is published before kick-off. All raw data were collected and engineered in Python, with source URLs stored next to every file. The scripts include automated audits: 104 and 208 rows, no missing values, both rows of a match mirrored, and altitude identical for both teams. Task 2.1 is written from team A's perspective, so every variable is a **difference or log ratio** between the teams. A log ratio was used where values are multiplicative (squad value) or heavily skewed (rest days), following the Week 9 advice on non-linear transforms. These choices were tested rather than assumed (Table 1)."),
  new Paragraph({ spacing: { after: 30 }, keepNext: true, children: runs("**Table 1.** The 16 explanatory variables and two responses (*Both = shared; *rest days enters 2.1 as a log ratio and 2.2 as raw days).", SMALL) }),
  table(T1W, ["Variable", "Task", "How it was derived (source)", "Why it should matter / evidence"], t1rows, t1fills),
  new Paragraph({ spacing: { after: 60 }, children: [] }),

  figure("fig1_screening.png", 2346, 830),
  CAPTION("**Figure 1.** Correlation of each explanatory variable with the response. Strength measures dominate both tasks. In Task 2.1, FIFA points and market value are correlated with each other (r = 0.80, VIF 5.2), which shapes the regression results below. In Task 2.2 no pair of predictors exceeds |r| = 0.46, so collinearity is not a concern."),

  H("3. Methodology"),
  P("**Validation design.** Following the Padlet guidance, each task uses a train/validation/test structure. *Task 2.1:* 21 matches (20%) were held out once, stratified by result band so wins, draws and losses appear in both sets (Kolmogorov-Smirnov p > 0.55 for every variable). All comparison and tuning used repeated 5-fold CV (10 repeats) and **nested CV** (5 folds x 3 repeats with an inner 5-fold tuning loop) on the 83 training matches only. *Task 2.2:* each match contributes two mirrored rows, so a random split would leak information between training and test sets. We therefore split by match: a 60/40 GroupShuffleSplit for the first MLR, then **nested GroupKFold** (5 outer, 4 inner folds) for model comparison and tuning. **Pre-processing:** predictors were standardised inside each pipeline so that coefficients are comparable and penalties are fair. **Competing algorithms:** Ridge, Lasso, Elastic Net, Huber, polynomial (degree 2) + Ridge, SVR (RBF), k-NN, decision tree, random forest, extra trees and gradient boosting (Task 2.1), and Ridge, Lasso, random forest, gradient boosting and **Poisson regression** (Task 2.2). Hyper-parameters were tuned by grid search inside the inner folds only. **Inference:** fold differences were tested with the Nadeau-Bengio corrected resampled t-test (2.1) or paired t-tests (2.2), test-set differences with a paired bootstrap, and the Task 2.1 choice also applied the one-standard-error rule, which prefers the simpler model when scores are within noise. Metrics are MAE, MSE, RMSE, NRMSE (RMSE / range) and R², as defined in Weeks 8-9."),

  H("4. Results: Linear Regression 2.1 (goal difference)"),
  P("**Class MLR.** Fitted to the 83 training matches with all eight variables, the model is highly significant (F = 8.12, p < 0.001; R² = 0.467, adjusted R² = 0.410). The standard assumption tests all pass: linearity (Ramsey RESET p = 0.53), constant variance (Breusch-Pagan p = 0.29), normal residuals (Shapiro-Wilk p = 0.84) and independence (Durbin-Watson 1.97). On the unseen test set it achieved **RMSE 1.34 goals, R² 0.63 and NRMSE 0.134**, against 2.20 for a mean-only baseline. Individually, only **market value** is significant (b = 0.67 per log unit, p = 0.005): a squad worth 2.7 times its opponent's adds about 0.7 goals to the expected margin. FIFA points has a strong simple correlation (r = 0.65) but is not significant (p = 0.17), because it shares most of its information with market value (r = 0.80). This is the collinearity effect described in Week 9, not evidence that ranking is irrelevant. Hosting is the only other meaningful effect (+1.22 goals, p = 0.04 on all 104 matches)."),
  P("**Too many variables for 104 matches.** The test R² of 0.63 is optimistic. Across 50 CV folds the same MLR averaged R² 0.19 (RMSE 1.65), so the 21-match test set happened to be easy. An exhaustive search of all 255 variable subsets found that the **best two-variable model** (FIFA points + market value, CV RMSE 1.514) beats the full eight-variable model (1.651). Beyond two variables, every addition increased error, so the extra six variables mostly fit noise. The Lasso reaches the same conclusion automatically. With its penalty tuned by CV (alpha = 0.2), it keeps market value, FIFA points and a small host effect and sets the other five coefficients to zero (Figure 3b). Across 500 bootstrap resamples these choices are stable: market value is kept 99.6% of the time, FIFA points 95.6% and host 57%, and every other variable less than 34%."),
  P("**Competing algorithms (Figure 2a).** Under nested CV the **Lasso was best (RMSE 1.557)**, followed by Elastic Net (1.570). The ensembles were mid-table: extra trees 1.605, random forest 1.608 and gradient boosting 1.681. The flexible non-linear models were worst: a single tree (1.823) and degree-2 polynomial features (1.841) overfit 83 matches badly. No model beat the MLR significantly (all corrected p > 0.20), which tells us the relationship is essentially **linear and additive**: non-linearity adds variance without capturing any extra structure. On the held-out test set the ranking flipped (MLR 1.34 vs Lasso 1.50). The paired bootstrap interval for that difference spans zero (-0.13 to +0.40), so the flip is within sampling noise for 21 matches. We added two robustness checks. In a **temporal validation** (train on the first 72 matches, test on the last 32, mostly knockouts) the Lasso won (RMSE 1.33 vs MLR 1.38 vs baseline 1.62). In a **listing-order check**, FBref usually lists the stronger team first, which explains the raw mean goal difference of +0.48 (p = 0.03). Once strength is in the model the intercept is not significant (p = 0.43), and a symmetric no-intercept model performs identically (p = 0.95), so the predictions do not depend on which team is called 'A'."),

  figure("fig2_model_comparison.png", 2414, 851),
  CAPTION("**Figure 2.** Systematic comparison of competing algorithms under nested cross-validation (tuning inside each training fold, so scores are unbiased). The dashed line marks the class MLR. Orange is the selected model, grey the mean baseline and green the Poisson count model."),
  H("5. Results: Linear Regression 2.2 (goals scored by a team)"),
  P("**Class MLR.** On all 208 rows the model is significant (F = 9.02, p < 0.001) but explains much less: R² = 0.266, adjusted 0.237. On the 40% match-grouped test set it scored **RMSE 1.29, R² 0.19, NRMSE 0.184** (baseline 1.44). Only **FIFA points** is significant: +0.28 goals per 100-point advantage (p < 0.001). The assumption checks expose a count response. Linearity holds (RESET p = 0.21), but residuals are heteroscedastic (Breusch-Pagan p = 0.03) and non-normal (Shapiro-Wilk p < 0.001, skew 0.86), and 1.9% of out-of-fold MLR predictions were *negative goals*. Because the two rows of a match share conditions, we also estimated match-clustered standard errors. The residual correlation within a match was small (r = 0.15, p = 0.13), and FIFA points remained significant either way. Notably, attacking form (r = 0.22) adds nothing once FIFA points is included: strong teams score more recently *because* they are strong, so the information overlaps."),
  P("**Competing algorithms (Figure 2b).** With its penalty tuned (alpha = 0.1 in every outer fold), the **Lasso had the lowest error (RMSE 1.226, R² 0.18, MAE 0.94)**, followed by Ridge (1.239), Poisson (1.245), MLR (1.249), gradient boosting (1.258) and random forest (1.293). All models beat the baseline (1.394; Poisson vs baseline p = 0.003), but the top four are statistically tied (Lasso vs Poisson p = 0.23). An untuned Lasso (default alpha = 1) removes every variable and collapses to the baseline, which shows why tuning inside CV matters. The final Lasso keeps five variables: FIFA points dominates, with small positive effects for previous extra time, opponent defensive form and line-up changes, and a slight negative effect for rest days. Altitude, attacking form and qualification pressure are dropped (Figure 3c)."),
  P("**Why Poisson regression matters here.** Goals are counts, so we fitted scikit-learn's PoissonRegressor (log link). It can never predict negative goals, its Pearson dispersion is **0.99** (the Poisson variance assumption holds almost exactly), and its out-of-fold predictions reproduce the observed distribution of goals: 25.9% predicted zero-goal games vs 26.4% observed (Figure 3d). Its coefficients read as rate ratios: **+1 SD of FIFA points (237 points) multiplies expected goals by 1.57**. Coming off an extra-time match multiplies goals by 1.53 (clustered p = 0.008), but this rests on only 8 team-matches, so we treat it as a hypothesis. **The ceiling on Task 2.2:** if goals are Poisson given the true scoring rate, about 1.48 of the 1.94 total variance is pure chance. On that approximation the best achievable R² is about 0.24 and the lowest achievable RMSE is about sqrt(1.48) = 1.22. Our Lasso (R² 0.18, RMSE 1.226) is close to that floor. This is why no model, however flexible, can do much better with pre-match data."),
  H("6. Discussion and recommendations"),
  new Paragraph({ spacing: { after: 30 }, keepNext: true, children: runs("**Table 2.** Headline performance (RMSE in goals, R² in brackets; ET = extra trees, GB = gradient boosting). Bold = best per column.", SMALL) }),
  table(T2W, ["Model", "2.1 nested CV RMSE", "2.1 test RMSE (R²)", "2.1 temporal RMSE", "2.2 nested CV RMSE (R²)", "2.2 nested CV MAE"], t2rows),
  new Paragraph({ spacing: { after: 50 }, children: [] }),
  P("**Comparing the tasks.** The same lesson appears twice: once team strength is measured, form, fatigue and context add little that is reliable. Goal difference (2.1) is more predictable than one team's goals (2.2) because it measures the *gap* between two teams, which strength describes directly, while a single team's tally carries more Poisson noise. **Recommendations.** (1) For Task 2.1 we recommend the **Lasso** with three active variables (market value, FIFA points, host). It won nested CV and the temporal test, it is stable under bootstrap, and it is easier to explain. The class MLR's better 21-match test score is not statistically meaningful. (2) For Task 2.2 we recommend the **Lasso for point predictions** and the **Poisson model when probabilities are needed** (for example, the chance a team fails to score), because it respects the count nature of goals at almost no cost in accuracy. For example, with the other variables at their averages, a team 300 FIFA points weaker than its opponent is expected to score 0.74 goals and to fail to score 48% of the time; a team 300 points stronger is expected to score 2.34 goals and to fail to score only 10% of the time. (3) Non-linear and ensemble methods are not justified at this sample size: they were never better and sometimes much worse. **Limitations.** One tournament gives small samples (21-match test set, 8 extra-time cases). Rankings and squad values are single pre-tournament snapshots. Line-up changes are only known about an hour before kick-off. The inputs were gathered from several sources, which leaves room for small matching errors even with the audits. Pooling previous World Cups would be the most valuable next step."),

  figure("fig3_final_models.png", 2326, 1280),
  CAPTION("**Figure 3.** (a) Held-out test predictions for Task 2.1. Points near the dashed line are accurate; both models compress extreme scorelines towards zero. (b, c) Effect of a one-standard-deviation increase in each variable (MLR and Lasso refitted on all rows; Poisson shown as its average marginal effect). The Lasso shrinks weak variables to exactly zero, and the Poisson and Lasso agree on what matters. (d) Observed goal counts against the probabilities implied by out-of-fold Poisson predictions."),

  H("7. Conclusion"),
  P("We built two pre-match datasets with sixteen distinct, well-motivated explanatory variables and evaluated them rigorously: held-out tests, nested and match-grouped cross-validation, assumption tests and statistical comparison of twelve algorithms. For goal difference, a sparse Lasso driven by squad value and FIFA ranking explains about 30% of unseen variance in cross-validation and 53% on the test set, and the relationship is linear. For goals scored by a team, FIFA ranking is the one dependable predictor, and the best models sit close to the limit set by the randomness of goals. More complex algorithms did not help; better and larger data would."),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 },
      margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "HIT140_A3_Objective2_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out);
});
