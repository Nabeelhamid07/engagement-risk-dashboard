# ML Model Performance Report

All metrics are computed on the **held-out test split** (90 students) that was never seen during training or model selection. Validation is used only to choose the winning model family, fit the sigmoid (Platt) calibrator, and pick the decision threshold.

## Target: `withdrew`

**Winning model:** `rf`  ·  **Decision threshold:** 0.136  (F1-optimal on val, recall floor 0.60)

At the chosen threshold the **validation** confusion is TP=11, FP=19, FN=3, TN=57  (precision=0.37, recall=0.79, F1=0.50).

### Validation leaderboard

| Model | Val ROC-AUC | Val PR-AUC |
|---|---:|---:|
| `rf` | 0.832 | 0.554 |
| `logreg` | 0.803 | 0.363 |
| `xgb` | 0.790 | 0.444 |

### Test-set performance

- N = **90** students, positives = **14**
- ROC-AUC: **0.850**
- PR-AUC:  **0.494**
- Brier score (calibration error, lower is better): 0.108
- Log loss: 0.342

**Confusion matrix** @ threshold 0.136:

|              | Pred 0 | Pred 1 |
|---           |---:|---:|
| **Actual 0** | 56 | 20 |
| **Actual 1** | 2 | 12 |

Class 1 (positive) — precision: **0.375**, recall: **0.857**, F1: **0.522**

## Target: `failed_next_assessment`

**Winning model:** `xgb`  ·  **Decision threshold:** 0.516  (F1-optimal on val, recall floor 0.70)

At the chosen threshold the **validation** confusion is TP=24, FP=4, FN=5, TN=57  (precision=0.86, recall=0.83, F1=0.84).

### Validation leaderboard

| Model | Val ROC-AUC | Val PR-AUC |
|---|---:|---:|
| `xgb` | 0.919 | 0.805 |
| `rf` | 0.916 | 0.801 |
| `logreg` | 0.871 | 0.777 |

### Test-set performance

- N = **90** students, positives = **24**
- ROC-AUC: **0.960**
- PR-AUC:  **0.927**
- Brier score (calibration error, lower is better): 0.069
- Log loss: 0.247

**Confusion matrix** @ threshold 0.516:

|              | Pred 0 | Pred 1 |
|---           |---:|---:|
| **Actual 0** | 60 | 6 |
| **Actual 1** | 3 | 21 |

Class 1 (positive) — precision: **0.778**, recall: **0.875**, F1: **0.824**
