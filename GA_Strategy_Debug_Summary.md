
# 🔍 Strategy Debugging Summary: GA vs Raw Rule Performance

## 🧠 Problem Overview
The current implementation of the backtesting system reveals a consistent and critical issue:
> The GA-optimized ensemble strategy **underperforms** individual rules and an equal-weight strategy — often returning near-zero or negative results, even on the training data it was optimized on.

---

## 🔎 Key Findings

### 1. **Raw Rules Perform Better Than the Trained Strategy**
- Example: `Rule2` alone achieved ~28% return.
- GA strategy frequently delivers < 0.1% or negative return.

### 2. **Signal Inversion Identified**
- Many strong rules had **negative correlation** with the GA-generated final signal.
- Inverting GA weights improved rule alignment, but performance remained poor.

### 3. **Equal-Weight Ensemble Outperforms GA**
- A naive equal-weighted strategy beat the GA in most cases.
- Suggests the GA is not learning meaningful combinations, or is overfitting to noise.

---

## 🚨 Root Causes Identified

| Cause                           | Explanation |
|--------------------------------|-------------|
| **Signal cancellation**        | Combining many rules blurs direction and cancels signal polarity. |
| **Overfitting in GA**          | Optimizing raw returns leads to unstable, fragile strategies. |
| **Inadequate fitness function**| Raw return sum does not penalize noise or volatility. |
| **Thresholding losses**        | `np.sign()` and percentile-based thresholds discard useful signal magnitudes. |
| **Correlation noise**          | High redundancy and interaction between rules confuses GA optimization. |

---

## ✅ Action Plan

### 🔧 Short-Term Fixes
- [x] Normalize signals before weighting
- [x] Add diagnostic tools to compare strategies
- [x] Test simple rule ensembles (top 3 rules)
- [x] Test `weights = -weights` inversion

### 🛠 Recommended Improvements
- Replace GA fitness function with **Sharpe-like metric**
- Constrain GA weights to be **positive-only** (`[0, 1]`)
- Prune or cluster **correlated rules** before training
- Consider adaptive or **soft-thresholding** strategies (e.g., `tanh(weighted_signal)`)

---

## 📊 Diagnostic Results Snapshot

| Strategy        | Total Return | Sharpe  |
|----------------|--------------|---------|
| GA Strategy     | -0.10%       |  -0.00  |
| Equal-Weight    | +0.16%       |  +0.005 |
| Rule2 (Top Rule)| +28.5%       |   High  |

---

## ✅ Conclusion

> The current GA optimization pipeline **reduces signal quality**, likely due to unbounded weight search, poor fitness alignment, and destructive rule mixing. With small changes — especially to the fitness function and aggregation logic — performance can be significantly improved.
