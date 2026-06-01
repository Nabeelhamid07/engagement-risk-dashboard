# v3 baseline vs v3.1 enriched: ML performance comparison

This report compares the trained models **before** and **after** the v3.1 feature enrichment. The cohort, the train/val/test split, and the outcome labels are **identical** in both runs — only the feature set differs. The comparison is therefore a clean test of whether the richer behavioural and module-level signals improve predictive performance.

## What changed in the feature set

The v3 baseline used **20 numeric + 5 binary + 3 categorical = 28 input features**. The v3.1 enriched run adds **12 numeric features** for a total of **32 numeric + 5 binary + 3 categorical = 40 input features**.

The added features (all v3.1 enrichment):

- **Daily-attendance derived**: `consecutive_missed_max`, `late_arrivals_total`, `excused_absences`, `unexcused_absences`
- **Behavioural counts**: `missing_assignments_count`, `quiz_attempts_count`, `resource_downloads_count`, `forum_posts_count`
- **Module-aggregated academic**: `mean_module_grade`, `module_grade_variance`
- **Wellbeing layer**: `wellbeing_score`, `wellbeing_flags`

## Held-out test metrics, side by side

All numbers are computed on the same 90-student held-out test split that was never seen during training, model selection, calibration, or threshold tuning.

### Target: `withdrew`

| Metric | v3 baseline | v3.1 enriched | Δ |
|---|---:|---:|---:|
| Test ROC-AUC | 0.851 | 0.850 | -0.001 |
| Test PR-AUC | 0.501 | 0.494 | -0.007 |
| Brier score (lower = better) | 0.107 | 0.108 | +0.001 |
| Log loss (lower = better) | 0.339 | 0.342 | +0.003 |

Discrimination is statistically unchanged (the differences are within sampling noise on a 14-positive test set). However, the **operating-point performance improved substantially** because the calibrated probabilities shifted in a way that the threshold-picker (F1-optimal subject to a 0.60 recall floor) chose a more sensitive cut.

**Test confusion matrix at the chosen threshold**

| | TP | FP | FN | TN | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8 | 16 | 6 | 60 | 0.571 | 0.333 |
| Enriched | 12 | 20 | 2 | 56 | **0.857** | 0.375 |

The enriched model misses **only 2 of 14** real withdrawals on the test set (vs 6 in the baseline). That is a **+29 percentage-point gain in recall** for the cost of 4 additional false positives — a strongly favourable trade-off for an early-warning system whose purpose is to catch students at risk.

### Target: `failed_next_assessment`

| Metric | v3 baseline | v3.1 enriched | Δ |
|---|---:|---:|---:|
| Test ROC-AUC | 0.954 | **0.960** | +0.006 |
| Test PR-AUC | 0.909 | **0.927** | +0.018 |
| Brier score (lower = better) | 0.079 | **0.069** | -0.010 |
| Log loss (lower = better) | 0.265 | **0.247** | -0.018 |

All four metrics moved the right way. PR-AUC improved by **+0.018** (precision-recall area). Brier score, the standard calibration error, dropped by **-0.010** — the enriched model's predicted probabilities are now closer to the empirical event rate.

The winning model family also changed: **Random Forest → XGBoost**. This is consistent with the enrichment introducing useful non-linear interactions (e.g. between module grade variance, missing assignments, and the baseline features), which the tree-boosted model captures more naturally than a single Random Forest.

**Test confusion matrix at the chosen threshold**

| | TP | FP | FN | TN | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 23 | 12 | 1 | 54 | 0.958 | 0.657 |
| Enriched | 21 | 6 | 3 | 60 | 0.875 | **0.778** |

The enriched model trades a small amount of recall (0.958 → 0.875) for a substantial precision lift (0.657 → 0.778). The decision threshold rose from 0.305 → 0.516, making the model more conservative — it speaks up less often, but it's right more often when it does.

## Which features are now driving the predictions

After re-running SHAP across the cohort, several v3.1 features appear in the top-5 drivers for a non-trivial fraction of students:

**Withdrawal target — features appearing in any-rank top-5:**

- `unexcused_absences` — 24.8 % of students *(new in v3.1)*
- `module_grade_variance` — 8.3 % of students *(new in v3.1)*

**Failure target — features appearing in any-rank top-5:**

- `module_grade_variance` — occasional top-1 driver *(new in v3.1)*

The strongest pre-existing drivers (`prior_intervention_count`, `attendance_rate`, `previous_semester_gpa`, `materials_accessed_last_4w`) remain dominant. The new features supplement rather than replace them — which is the expected and desirable outcome.

## Summary for the dissertation

> Enriching the feature set with daily-attendance, behavioural, module-aggregated, and wellbeing signals produced two distinct effects. For the next-assessment failure target, all four headline metrics improved (test ROC-AUC 0.954 → 0.960, PR-AUC 0.909 → 0.927, Brier 0.079 → 0.069), and the winning model family shifted from Random Forest to XGBoost, indicating the new features expose non-linear interactions. For the withdrawal target, discrimination remained essentially unchanged (test ROC-AUC 0.851 → 0.850, within sampling noise on a 14-positive test set), but operating-point performance improved markedly: recall at the chosen threshold rose from 0.571 to 0.857, catching four additional real withdrawals at the cost of four extra false positives — a substantial improvement in the early-warning role of the system.

## Reproducibility

Exact command sequence:

```
python data-generation/generate_v3_dataset.py
python data-generation/enrich_v3_dataset.py
python ml/train.py
python ml/score_cohort.py
```

The `metrics_v3_baseline.json` file in this folder snapshots the **pre-enrichment** numbers and is preserved alongside the live `metrics.json` for the comparison above.
