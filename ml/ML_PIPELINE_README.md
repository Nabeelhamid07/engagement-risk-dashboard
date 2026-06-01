# Stage 2 — ML Pipeline

This is the machine-learning core of the artifact. It trains, calibrates, and explains two predictive models from the v3 synthetic dataset, then writes everything the dashboard needs as plain data files.

## What we predict

Two independent binary outcomes per student, sampled during data generation from sigmoid(latent + context + Gaussian noise):

- **`withdrew`** — will this student withdraw within the semester window?  *(15.2% base rate in the cohort)*
- **`failed_next_assessment`** — will this student score below 50% on their next assessment?  *(28.7% base rate)*

Critically, neither outcome is a deterministic function of any feature. The Gaussian noise injected at data-generation time means even a perfect model has an irreducible error floor — so any ROC-AUC < 1.0 is honest, not a bug.

## Methodology

For each outcome:

1. **Three candidate model families** are trained on the 420-student **train** split with class-weighted loss:
   - Logistic regression (regularised linear baseline)
   - Random forest (400 trees, depth ≤ 8, leaf size ≥ 8, balanced-subsample weighting)
   - XGBoost (400 trees, depth 4, lr 0.05, scale_pos_weight, hist tree method)
2. **Validation-set ROC-AUC** on the 90-student val split selects the winning family. (Random Forest won both targets.)
3. **Sigmoid (Platt) calibration** is fit on the val split using `CalibratedClassifierCV` + `FrozenEstimator`. This keeps the winner frozen and only learns a 2-parameter logistic mapping from raw scores to probabilities — so predicted probabilities mean what they say.
4. **Decision threshold** is chosen by maximising F1 on the val split, subject to a recall floor (0.60 for withdraw, 0.70 for failure). The system is biased toward catching real positives over avoiding false alarms — appropriate for an early-warning use case.
5. **Honest evaluation** on the 90-student **test** split, which never touched training, selection, calibration, or threshold tuning.

## Feature set (35 input features, 28 after preprocessing)

Engineered in `features.py`:

- **Cross-sectional engagement** (already on the student row): attendance rate, attendance slope (4w vs prior 8w), VLE logins last week, VLE hours last week, VLE login/hours slope, materials accessed last 4w.
- **Weekly-derived temporal features** (joined in via `weekly_engagement.csv`): attendance volatility, VLE-login volatility, weeks with zero logins, weeks since last login, largest weekly attendance drop.
- **Submissions**: on-time count, late count.
- **Grades**: most recent assessment grade, previous semester GPA.
- **Upcoming assessment**: days to next assessment, accessed brief flag.
- **Intervention history**: prior count and prior improved count (proxy for "system already attempting to help").
- **Demographics / context**: year of study, commuter, part-time, disability, international, financial support, programme, derived engagement-trend label.

Preprocessing inside the sklearn `Pipeline`:

- `StandardScaler` for the numeric block
- pass-through for the binary block
- `OneHotEncoder` for the categorical block

## Test-set performance (honest, never-seen-before split)

| Target | Winner | Test ROC-AUC | Test PR-AUC | Brier | Recall (cls 1) | Precision (cls 1) |
|---|---|---:|---:|---:|---:|---:|
| `withdrew`               | RF | **0.851** | 0.501 | 0.107 | 0.571 | 0.333 |
| `failed_next_assessment` | RF | **0.954** | 0.909 | 0.079 | 0.958 | 0.657 |

Both ROC-AUCs are comfortably above 0.85 and below 1.0 — exactly the believable zone for a real student-success model. The Brier scores (0.08–0.11) show the calibrated probabilities are honest, not overconfident.

For comparison, validation leaderboard (informational only — selection used these):

| Target | logreg | rf | xgb |
|---|---:|---:|---:|
| `withdrew`               | 0.787 | **0.833** | 0.782 |
| `failed_next_assessment` | 0.894 | **0.920** | 0.917 |

## Explanations — SHAP

Per-student "key drivers" are computed with SHAP TreeExplainer on the underlying Random Forest (unwrapped from the calibrator). For each student × target we keep the top 5 features by absolute SHAP value, with a sign indicating whether the feature pushed risk **up (+)** or **down (-)**. Output is `data/shap_drivers.csv` in long format.

The dashboard never sees the term "SHAP" — features are renamed via `humanise_feature()` to readable labels like `"Recent attendance (4-week avg)"`. The bar chart in the student detail page will show the magnitude; the colour will show the direction.

## Outputs

Saved to `models/`:

| File | Purpose |
|---|---|
| `withdrew_model.joblib` | Calibrated RF pipeline for withdrawal |
| `failed_next_assessment_model.joblib` | Calibrated RF pipeline for failure |
| `feature_config.json` | Feature column lists (used by inference) |
| `metrics.json` | Machine-readable metrics for the dashboard's "Model Performance" page |
| `metrics_report.md` | Human-readable report for the dissertation appendix |

Saved to `data/` (consumed by the dashboard at runtime):

| File | Rows | Purpose |
|---|---:|---|
| `predictions.csv` | 600 | One row per student: `withdraw_prob`, `withdraw_pred`, `fail_prob`, `fail_pred`, blended `overall_risk` + `risk_band` |
| `shap_drivers.csv` | 6,000 | Long format, top 5 drivers per (student × target) |

The dashboard never imports `sklearn` or `shap` — it reads these CSVs. Faster cold start, smaller deployment footprint, and clean separation of training-time and serving-time concerns.

## How to (re)run

```powershell
cd "ml"
python train.py          # train + calibrate + save models + write metrics
python score_cohort.py   # batch-score all 600 students + write SHAP drivers
```

Reproducibility: every model uses `random_state=42`. Running both scripts on the same v3 dataset produces identical artifacts.

## What this gives the dissertation narrative

- A genuinely **predictive** model: outcomes are sampled with independent noise, so the model has to infer them, not look them up.
- Honest **validation methodology**: train / val / test separation, model selection on val, calibration on val, threshold on val, all reported on a never-seen test set.
- **Calibrated probabilities**, not just rankings — so the dashboard can say "this student has a 47% chance of withdrawing" and mean it.
- **Per-student explanations** via SHAP — no model is a black box at point of use.
- Clear room in the write-up to discuss the **precision/recall trade-off** for the withdrawal target (rare positive class), which is a real, defensible methodological choice rather than a flaw.
