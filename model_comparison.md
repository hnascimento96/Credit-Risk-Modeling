# Model Comparison — Credit Risk Modeling
**Dataset:** Give Me Some Credit (Kaggle)  
**Target variable:** `SeriousDlqin2yrs` (1 = default, 0 = non-default)  
**Default rate:** ~6.7% (imbalanced dataset)  
**Split:** 80% train / 20% test (stratified by target variable) 
**Classification threshold:** 0.5 (default) for Precision and Recall  

---

## Results

| Model | AUC | KS | Gini | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.8151 | 0.5194 | 0.6303 | 0.2251 | 0.6798 |
| Random Forest | 0.8691 | 0.5827 | 0.7383 | 0.2387 | 0.7137 |
| XGBoost | **0.8691** | **0.5827** | **0.7383** | 0.2267 | **0.7691** |
| LightGBM (Optuna) | 0.8677 | 0.5798 | 0.7354 | **0.2416** | 0.7372 |

---

## Metrics interpretation

- **AUC-ROC:** overall ability to rank customers by risk. Threshold-independent.
- **KS (Kolmogorov-Smirnov):** maximum separation between score distributions of good and bad payers. Standard metric in the credit market.
- **Gini:** normalized version of AUC (`Gini = 2 × AUC - 1`). Widely used in risk reports.
- **Precision:** of all customers flagged as defaulters, how many actually are.
- **Recall:** of all actual defaulters, how many the model identified.

> On imbalanced datasets like this one, AUC, KS and Gini are the primary metrics.
> Precision and Recall depend on the chosen threshold and should be analyzed
> alongside the credit policy simulation.

---

## Class imbalance strategy

All models were trained with class weight adjustment to compensate
for the ~6.7% default rate:

| Model | Parameter | Value |
|---|---|---|
| Logistic Regression | `class_weight` | `'balanced'` |
| Random Forest | `class_weight` | `'balanced'` |
| XGBoost | `scale_pos_weight` | 13 |
| LightGBM | `scale_pos_weight` | 13 |

> Without balancing, all models achieved recall < 5% on the default class,
> predicting almost everything as non-default and obtaining artificially
> high accuracy (~93%) with no real learning.

---

## Model analysis

### Logistic Regression — baseline

- AUC 0.8151: solid result for a linear model without advanced feature engineering.
- KS 0.5194: acceptable separation, but lower than ensemble models.
- Recall 67.98%: identifies 2 out of every 3 actual defaulters at default threshold.
- Serves as the performance floor: any more complex model must surpass
  these metrics to justify the added complexity.
- All ensemble models outperformed the baseline across all metrics.

### Random Forest

- AUC and KS identical to XGBoost: ranks customers with the same global quality.
- Recall 71.37%: captures more defaulters than logistic regression,
  but 5.5 percentage points below XGBoost.
- Slightly higher Precision than XGBoost (23.87% vs 22.67%): 
  when it rejects, it is slightly more accurate.
- Stable results without intensive hyperparameter tuning.
- With ~30,000 customers in the test set and ~2,000 actual defaulters,
  the 5.5-point recall gap between RF and XGBoost represents
  approximately 110 additional defaulters identified by XGBoost.

### XGBoost — final model

- Best recall in the comparison: 76.91%: identifies 3 out of every 4 actual defaulters.
- AUC and KS tied with Random Forest, but with superior recall.
- Result achieved without Optuna tuning. Reasonable parameters produced
  stable and reproducible performance.
- `scale_pos_weight=13` was critical: without balancing, recall dropped to 20%.
- Selected as the final model based on the criteria described in the section below.

### LightGBM (Optuna) — experimental

- AUC 0.8677 and KS 0.5798: close to XGBoost, but slightly lower.
- Recall 73.72%: between Random Forest and XGBoost.
- Highest Precision in the comparison: 24.16%: when it rejects, it is more accurate.
- Optuna tuning required multiple diagnostic iterations:

  | Attempt | AUC | Issue identified |
  |---|---|---|
  | No balancing | 0.59 | `scale_pos_weight` missing |
  | Simple balancing | 0.78 | Default `num_leaves` too restrictive |
  | Manual `num_leaves=63` | 0.85 | Best manual result |
  | Optuna v1 | 0.59 | `scale_pos_weight` dropped from `best_params` |
  | Optuna v2 | 0.79 | `num_leaves=20` selected — underfitting |
  | Optuna v3 (floor 63) | 0.79 | CV overfitting |
  | Optuna v4 (floor 80) | 0.79 | Same pattern |
  | `learning_rate=0.011` + `n_estimators=1500` | 0.86 | Adequate convergence |

- LightGBM showed high sensitivity to hyperparameters on this dataset.
AUC varied between 0.59 and 0.87 depending on configuration.
- High sensitivity represents operational risk in periodic retraining,
  as optimal hyperparameters may shift with new incoming data.

---

## Final model decision: XGBoost

### Selection criteria

**1. Best recall**  
76.91%: the highest in the comparison. In credit risk, the cost of approving
a defaulter (direct financial loss) is typically higher than the cost of
rejecting a good customer (lost revenue). Maximizing recall is the priority.

**2. AUC and KS equivalent to Random Forest**  
Same global ranking ability, with superior recall.

**3. Robustness to hyperparameters**  
Reasonable parameters produced AUC 0.8691 without intensive tuning.
Lower configuration dependency reduces risk in monthly retraining cycles.

**4. SHAP compatibility**  
Fully compatible with `TreeExplainer`, efficient and exact interpretability
for the SHAP analysis.

**5. Established market reference**  
XGBoost is widely used in production credit models, which
facilitates comparison with literature benchmarks and other institutions.

### Final model hyperparameters

```python
XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=13,
    random_state=42,
    n_jobs=-1
)
```

---