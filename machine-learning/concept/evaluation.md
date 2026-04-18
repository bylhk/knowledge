# Evaluation Best Practices

Evaluation answers one question: is this model better than what we have now, and better in a way that matters to the business? Offline metrics are a proxy — they are only useful if they correlate with the business outcome you actually care about.

---

## 1. Choose the Right Metric

### Classification

| Metric | When to use | When to avoid |
|--------|------------|---------------|
| AUC-ROC | Ranking quality, imbalanced classes | When a specific operating point matters |
| AUC-PR | Highly imbalanced, minority class is important | When false negatives and false positives have equal cost |
| F1 | Balanced precision/recall trade-off | When costs are asymmetric |
| Precision @ K | Top-K recommendation, ranking | When all predictions matter equally |
| Log loss | Probability calibration quality | When only the rank matters |
| Accuracy | Balanced classes only | Imbalanced classes — misleading |

### Regression

| Metric | When to use |
|--------|------------|
| RMSE | When large errors are disproportionately costly |
| MAE | When all errors have equal cost |
| MAPE | When relative error matters more than absolute |
| R² | Explaining variance — use alongside an absolute metric |

### Define the business metric first

Before choosing an ML metric, define the business outcome:

```
Business metric:  revenue per session
ML proxy metric:  AUC-PR on acceptance probability
Assumption:       higher AUC-PR → better ranked prices → higher acceptance → more revenue
```

The assumption must be validated with an online experiment. A model with higher AUC-PR does not automatically produce more revenue.

---

## 2. Baseline Models

Always evaluate against a baseline. Without a baseline, there is no reference point — a model with 0.82 AUC is meaningless without knowing what a naive approach achieves.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

# Baseline 1 — always predict the majority class
majority_class = np.bincount(y_test).argmax()
baseline_preds = np.full(len(y_test), majority_class)

# Baseline 2 — predict the class prior probability
prior          = y_train.mean()
prior_preds    = np.full(len(y_test), prior)
prior_auc      = roc_auc_score(y_test, prior_preds)

# Baseline 3 — simple rule-based model
rule_preds     = (X_test[:, feature_idx] > threshold).astype(float)
rule_auc       = roc_auc_score(y_test, rule_preds)

# Candidate model
model_auc      = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Prior baseline AUC: {prior_auc:.4f}")
print(f"Rule baseline AUC:  {rule_auc:.4f}")
print(f"Model AUC:          {model_auc:.4f}")
print(f"Lift over rule:     {(model_auc - rule_auc) / rule_auc * 100:.1f}%")
```

---

## 3. Calibration

A well-calibrated model produces probabilities that reflect true likelihoods — a prediction of 0.8 should be correct ~80% of the time. Uncalibrated probabilities produce incorrect expected value calculations and misleading confidence scores.

```python
from sklearn.calibration import calibration_curve
import numpy as np

def check_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """
    Compute calibration curve.

    Returns
    -------
    dict[str, np.ndarray]
        fraction_of_positives and mean_predicted_value per bin.
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return {
        "fraction_of_positives": fraction_pos,
        "mean_predicted_value":  mean_pred,
    }
```

If calibration is poor, apply isotonic regression or Platt scaling post-hoc:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
calibrated_model.fit(X_val, y_val)
```

---

## 4. Threshold Selection

For binary classifiers, the default 0.5 threshold is rarely optimal. Select the threshold on the validation set based on the business cost of false positives vs false negatives.

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

def select_threshold(
    y_val: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.7,
) -> float:
    """
    Select the highest-recall threshold that meets a minimum precision constraint.

    Parameters
    ----------
    y_val : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    min_precision : float
        Minimum acceptable precision.

    Returns
    -------
    float
        Selected decision threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

    valid_mask = precision[:-1] >= min_precision
    if not valid_mask.any():
        raise ValueError(f"No threshold achieves precision >= {min_precision}")

    best_idx = np.argmax(recall[:-1][valid_mask])
    return float(thresholds[valid_mask][best_idx])
```

---

## 5. In-Training Evaluation (Deep Learning)

For deep learning models, evaluation during training is essential — not optional. A single end-of-training metric tells you nothing about when the model peaked, whether it overfit, or whether training was stable. Monitoring metrics continuously throughout training catches these problems early and saves compute.

### What to track during training

| Signal | What it reveals |
|--------|----------------|
| Training loss per step | Whether the model is learning at all |
| Validation loss per epoch | Whether the model is generalising or overfitting |
| Train/val loss gap | Overfitting — gap widens as model memorises training data |
| Learning rate schedule | Whether LR decay is happening as expected |
| Gradient norm | Exploding or vanishing gradients |
| Metric on val set (AUC, F1) | Whether loss improvement translates to metric improvement |

### TensorBoard

TensorBoard is the standard tool for visualising in-training metrics. It reads event files written during training and renders live-updating charts — loss curves, metric trends, histograms, and more.

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

writer = SummaryWriter(log_dir="runs/experiment_01")

for epoch in range(n_epochs):
    # --- training step ---
    train_loss = run_train_epoch(model, train_loader, optimiser)

    # --- validation step ---
    val_loss, val_auc = run_val_epoch(model, val_loader)

    # Log scalars — visible in TensorBoard as live curves
    writer.add_scalar("loss/train",      train_loss, epoch)
    writer.add_scalar("loss/val",        val_loss,   epoch)
    writer.add_scalar("metric/val_auc",  val_auc,    epoch)

    # Log gradient norms to detect exploding/vanishing gradients
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    writer.add_scalar("grad/norm", total_norm ** 0.5, epoch)

writer.close()
```

```bash
# Launch TensorBoard — opens at http://localhost:6006
tensorboard --logdir runs/
```

### Comparing runs

TensorBoard overlays multiple runs on the same chart when each run writes to a separate subdirectory. This makes hyperparameter comparison visual and immediate:

```
runs/
├── lr_0.001_depth_4/
├── lr_0.001_depth_6/
└── lr_0.0001_depth_4/
```

```python
# Name the run from the config so it is identifiable in TensorBoard
run_name = f"lr_{config['lr']}_depth_{config['depth']}"
writer   = SummaryWriter(log_dir=f"runs/{run_name}")
```

### Early stopping with in-training evaluation

In-training evaluation enables early stopping — halt training when validation loss stops improving rather than running for a fixed number of epochs:

```python
class EarlyStopping:
    """
    Stop training when validation loss does not improve for `patience` epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping.
    min_delta : float
        Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.wait       = 0
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait      = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True


# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(n_epochs):
    train_loss        = run_train_epoch(model, train_loader, optimiser)
    val_loss, val_auc = run_val_epoch(model, val_loader)

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/val",   val_loss,   epoch)

    early_stopping.step(val_loss)
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Rules

- Always evaluate on the validation set at the end of every epoch — never only on training loss
- Log both train and val metrics to TensorBoard — the gap between them is the overfitting signal
- Use early stopping to avoid wasting compute and overfitting — do not run for a fixed epoch count
- Save the best checkpoint (lowest val loss) during training, not the final checkpoint
- Name TensorBoard runs from the config — anonymous `run_1`, `run_2` names are useless for comparison

---

## 6. Offline vs Online Evaluation

Offline evaluation measures model quality on historical data. Online evaluation measures business impact on live traffic. They answer different questions.

| | Offline | Online |
|---|---------|--------|
| What it measures | Model quality on historical data | Business impact on live users |
| Speed | Fast — runs on saved data | Slow — requires live traffic |
| Cost | Low | High — real users affected |
| Reliability | Limited — assumes past = future | High — measures actual behaviour |
| When to use | Before deployment, for model selection | After deployment, for business validation |

### Offline evaluation is necessary but not sufficient

A model can improve AUC while degrading revenue if:
- The metric does not align with the business objective
- The historical data does not reflect current user behaviour
- The model exploits patterns that no longer exist in production

Always follow offline evaluation with an online experiment before declaring a model better.

---

## 6. Champion / Challenger

The champion is the currently deployed model. A challenger is a candidate replacement. Never replace the champion without a controlled comparison.

```
challenger trained → offline evaluation → passes threshold?
    → shadow deployment → metrics comparable?
        → A/B test → statistically significant improvement?
            → promote challenger to champion
```

### Shadow deployment

Run the challenger in parallel with the champion — both score every request, but only the champion's output is served. Compare offline metrics on live traffic without any user impact.

```python
def score_with_shadow(
    request: dict,
    champion: object,
    challenger: object,
    logger,
) -> np.ndarray:
    champion_score   = champion.predict(request)
    challenger_score = challenger.predict(request)   # scored but not served

    logger.info("shadow_score", extra={
        "request_id":       request["request_id"],
        "champion_score":   float(champion_score),
        "challenger_score": float(challenger_score),
    })

    return champion_score   # only champion is returned
```

### A/B test

Route a percentage of traffic to the challenger and measure the business metric directly:

| Traffic split | When to use |
|--------------|------------|
| 5% challenger | High-risk change — minimise exposure |
| 50/50 | Standard A/B test — fastest statistical power |
| Gradual ramp | Start at 5%, increase if metrics hold |

### Promotion criteria

Define promotion criteria before the experiment starts — not after seeing the results:

```python
PROMOTION_CRITERIA = {
    "min_auc_improvement":        0.005,   # at least 0.5% AUC improvement
    "max_latency_increase_ms":    10,       # no more than 10ms added latency
    "min_business_metric_lift":   0.01,    # at least 1% lift on business KPI
    "min_sample_size":            10_000,  # minimum requests before deciding
    "confidence_level":           0.95,    # statistical significance threshold
}
```

### Rules

- Never promote a challenger based on offline metrics alone
- Define promotion criteria before the experiment — post-hoc criteria are p-hacking
- Keep the champion deployed until the challenger is proven — never swap speculatively
- Log both champion and challenger scores during shadow deployment for post-hoc analysis
