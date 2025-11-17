# Phase 3 Implementation: Modeling & Forecasting

## Overview

This document describes the implementation of **Phase 3** of the BDC Analytics & Forecasting Platform, which includes:

1. **Baseline Models** - Simple benchmarks (naive, rolling mean, ridge regression)
2. **ML Models** - Advanced models (XGBoost, LightGBM, Random Forest)
3. **Backtesting Framework** - Expanding-window cross-validation for time-series
4. **Evaluation Metrics** - MAE, RMSE, MAPE, R²
5. **Prediction/Inference** - Generate forecasts for dashboard

---

## 1. Baseline Models (`src/models/baselines.py`)

### Purpose

Establish performance benchmarks with simple, interpretable models.

### Models Implemented

1. **NaiveForecaster**
   - Predicts zero change (or last value)
   - Baseline: assumes no change from current quarter
   - No training required

2. **RollingMeanForecaster**
   - Predicts mean of last N quarters (configurable window: 2, 4, 8)
   - Captures historical average trend
   - Simple but often competitive

3. **RidgeForecaster**
   - L2-regularized linear regression
   - Prevents overfitting with regularization
   - Multiple alpha values tested (0.1, 1.0, 10.0)

### Usage

```python
from src.models.baselines import get_baseline_models

models = get_baseline_models()
# Returns: [NaiveForecaster, RollingMeanForecaster(2), RollingMeanForecaster(4), 
#           RollingMeanForecaster(8), RidgeForecaster(0.1), RidgeForecaster(1.0), 
#           RidgeForecaster(10.0)]
```

---

## 2. ML Models (`src/models/ml_models.py`)

### Purpose

Advanced gradient boosting and ensemble models for better predictive performance.

### Models Implemented

1. **XGBoostForecaster**
   - Gradient boosting with XGBoost
   - Handles missing values natively
   - Provides feature importance
   - Configurations: n_estimators=[50, 100, 200], max_depth=[4, 6], learning_rate=[0.05, 0.1]

2. **LightGBMForecaster**
   - Fast gradient boosting with LightGBM
   - Efficient on large datasets
   - Similar configurations to XGBoost

3. **RandomForestForecaster**
   - Ensemble of decision trees
   - Robust to overfitting
   - Configurations: n_estimators=[50, 100, 200], max_depth=[10, 15, 20]

### Feature Importance

Both XGBoost and LightGBM provide feature importance scores:
```python
model = XGBoostForecaster(...)
model.fit(X_train, y_train)
importance = model.get_feature_importance()  # Returns pd.Series
```

### Dependencies

- **XGBoost**: `pip install xgboost`
- **LightGBM**: `pip install lightgbm`
- **Random Forest**: Included in scikit-learn

Models gracefully handle missing dependencies (only available models are used).

---

## 3. Backtesting Framework (`src/models/backtesting.py`)

### Purpose

Proper time-series cross-validation that prevents data leakage.

### Implementation

**Class: `ExpandingWindowCV`**

- **Expanding window**: Train on all data up to cutoff, test on next period(s)
- **Grouped by BDC**: Ensures no future data leaks into training
- **Configurable**: `n_test_periods`, `min_train_periods`

**Example:**
```
Periods: [2022_Q4, 2023_Q1, 2023_Q2, 2023_Q3, 2023_Q4, 2024_Q1]

Fold 1: Train=[2022_Q4], Test=[2023_Q1]
Fold 2: Train=[2022_Q4, 2023_Q1], Test=[2023_Q2]
Fold 3: Train=[2022_Q4, 2023_Q1, 2023_Q2], Test=[2023_Q3]
...
```

**Function: `backtest_model()`**

- Fits model on training fold
- Predicts on test fold
- Returns predictions with `y_true`, `y_pred`, `fold`, `period`, `cik`

**Function: `backtest_multiple_models()`**

- Backtests all models in parallel
- Returns dictionary mapping model name to predictions DataFrame

### Why This Matters

- **No data leakage**: Future periods never used for training
- **Realistic evaluation**: Mimics real-world forecasting scenario
- **BDC-specific**: Each BDC's history is respected (grouped by CIK)

---

## 4. Evaluation Metrics (`src/models/evaluation.py`)

### Purpose

Compute regression metrics to compare model performance.

### Metrics Implemented

1. **MAE (Mean Absolute Error)**
   - Average absolute difference between predicted and actual
   - Units: Same as target (percentage points for % change)

2. **RMSE (Root Mean Squared Error)**
   - Square root of mean squared error
   - Penalizes large errors more than MAE

3. **MAPE (Mean Absolute Percentage Error)**
   - Percentage error (useful for interpretability)
   - Handles division by zero gracefully

4. **R² (Coefficient of Determination)**
   - Proportion of variance explained
   - Range: -∞ to 1 (1 = perfect, 0 = no better than mean)

### Functions

- `compute_metrics(y_true, y_pred)`: Compute all metrics
- `evaluate_model_predictions(df, y_true_col, y_pred_col, group_by)`: Evaluate grouped by entity/period
- `compare_models(results_dict)`: Compare multiple models side-by-side

### Output Format

```python
{
    "mae": 2.45,
    "rmse": 3.12,
    "mape": 15.3,
    "r2": 0.68,
    "n_samples": 1200
}
```

---

## 5. Training Module (`src/models/train.py`)

### Purpose

Orchestrates model training, backtesting, evaluation, and artifact saving.

### Main Function: `train_and_evaluate_models()`

**Process:**
1. Load feature panel
2. Prepare features and target
3. Create expanding-window CV
4. Backtest all models (baselines + ML)
5. Evaluate and compare
6. Save predictions and comparison table

**Outputs:**
- `data/processed/models/backtest_predictions.parquet`: All model predictions
- `data/processed/models/model_comparison.csv`: Performance comparison table

### Command-Line Usage

```bash
python -m src.models.train
```

**Output:**
```
Training data: 1655 rows, 39 features
Target: target_next_qtr_pct
Target stats: mean=0.23, std=5.67

Backtesting 13 models...
Cross-validation: 11 folds

Model Comparison:
                    mae     rmse    mape      r2
XGBoost_n100_d6_lr0.1  2.12   3.45   12.3   0.72
LightGBM_n100_d6_lr0.1  2.18   3.52   12.8   0.71
Ridge_alpha1.0          2.45   3.89   14.2   0.65
...
```

---

## 6. Prediction Module (`src/models/predict.py`)

### Purpose

Generate forecasts for the latest period using trained models.

### Functions

1. **`load_model(model_path)`**
   - Loads trained model and metadata from pickle files

2. **`predict_with_model(model, df, feature_cols)`**
   - Generates predictions for new data

3. **`generate_forecasts(df, model_name, target_col)`**
   - Generates forecasts for latest period using specified model

4. **`generate_all_forecasts(df, target_col)`**
   - Automatically selects best model (lowest MAE) and generates forecasts

5. **`save_forecasts(forecasts_df, output_path)`**
   - Saves forecasts to Parquet

### Output Format

```python
# forecasts.parquet
period    cik      name                    forecast  model_name
2025_Q1  1234567  Ares Capital Corp       2.34     XGBoost_n100_d6_lr0.1
2025_Q1  2345678  Blackstone Private...   1.89     XGBoost_n100_d6_lr0.1
...
```

### Command-Line Usage

```bash
python -m src.models.predict
```

---

## File Structure

```
src/models/
├── __init__.py
├── baselines.py          # Baseline models
├── ml_models.py         # ML models (XGBoost, LightGBM, RF)
├── backtesting.py       # CV framework
├── evaluation.py        # Metrics
├── train.py             # Training orchestration
└── predict.py           # Inference/forecasting

data/processed/models/
├── backtest_predictions.parquet    # All model predictions
├── model_comparison.csv            # Performance comparison
├── {model_name}.pkl                # Trained models
├── {model_name}_metadata.pkl      # Model metadata
└── forecasts.parquet               # Latest forecasts
```

---

## Usage Workflow

### 1. Train Models

```bash
python -m src.models.train
```

This will:
- Load feature panel
- Backtest all models
- Save predictions and comparison

### 2. Generate Forecasts

```bash
python -m src.models.predict
```

This will:
- Load best model (lowest MAE)
- Generate forecasts for latest period
- Save to `forecasts.parquet`

### 3. Programmatic Usage

```python
from src.models.train import train_and_evaluate_models, evaluate_all_models
from src.models.predict import generate_all_forecasts
import pandas as pd

# Load feature panel
df = pd.read_parquet("data/processed/bdc_feature_panel.parquet")

# Train and evaluate
predictions_dict = train_and_evaluate_models(df, target_col="target_next_qtr_pct")
comparison = evaluate_all_models(predictions_dict)
print(comparison.sort_values("mae"))

# Generate forecasts
forecasts = generate_all_forecasts(df)
print(forecasts)
```

---

## Model Selection Strategy

1. **Backtesting**: All models evaluated on historical data
2. **Metric**: MAE (Mean Absolute Error) used as primary metric
3. **Selection**: Model with lowest MAE on backtest is used for forecasting
4. **Robustness**: Multiple models tested to find best performer

---

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**Core:**
- `pandas`, `numpy`, `scikit-learn` (required)
- `xgboost`, `lightgbm` (optional, for ML models)
- `shap` (optional, for explainability - future enhancement)

---

## Future Enhancements

1. **SHAP Explainability**
   - Add SHAP values for model interpretability
   - Show feature contributions to predictions

2. **Hyperparameter Tuning**
   - Grid search or Bayesian optimization
   - Automated model selection

3. **Ensemble Methods**
   - Stacking or voting ensembles
   - Combine multiple models

4. **Confidence Intervals**
   - Quantile regression for prediction intervals
   - Uncertainty quantification

5. **Model Monitoring**
   - Track model drift over time
   - Alert when performance degrades

---

## Notes

- **Target Variable**: Default is `target_next_qtr_pct` (next quarter % change in net assets)
- **Missing Values**: Handled by filling with median (models may handle NaN differently)
- **Feature Selection**: All numeric columns used (excluding identifiers and targets)
- **Time-Series**: Proper CV ensures no future leakage
- **BDC-Specific**: Grouped by CIK to respect entity boundaries

---

## Troubleshooting

### Issue: ModuleNotFoundError for sklearn/xgboost/lightgbm

**Solution:** Install dependencies:
```bash
pip install scikit-learn xgboost lightgbm
```

### Issue: No models available

**Check:** Ensure feature panel exists and has target column:
```bash
python -m src.features.feature_engineering
```

### Issue: All models have high error

**Possible causes:**
- Target variable may be too noisy
- Features may not be predictive
- Need more training data
- Consider feature engineering improvements

---

## Next Steps

With Phase 3 complete, proceed to:

1. **Dashboard Phase** (`dashboards/`):
   - Load forecasts and model comparison
   - Display predictions with confidence intervals
   - Show model performance metrics
   - Visualize feature importance

2. **Documentation**:
   - Model cards (assumptions, limitations)
   - Feature importance analysis
   - Final report with results

