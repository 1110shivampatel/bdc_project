# Modeling Issues and Fixes

## Issue Summary

When running `python -m src.models.train`, several issues were encountered:

1. **Target column has NaN/inf values**: The target `target_next_qtr_pct` contains invalid values
2. **Models failing**: Many models fail with "Label contains NaN, infinity or a value too large"
3. **Extremely high errors**: Models that do run have absurdly high MAE (4.75e10, 1.89e36)

## Root Cause

The target creation in `create_target_labels()` uses:
```python
df[f"target_next_qtr_pct"] = df.groupby("cik")[target_metric].shift(-1).pct_change() * 100
```

This can create:
- **NaN values**: When there's no next quarter (last period for each BDC)
- **Inf values**: When dividing by zero (current value is 0)
- **Invalid values**: When the base metric itself has issues

## Fixes Applied

### 1. Data Cleaning in Training (`src/models/train.py`)

Added cleaning steps:
- Remove rows with NaN/inf target values
- Replace infinity in features with NaN
- Filter out invalid data before training

### 2. Data Cleaning in Backtesting (`src/models/backtesting.py`)

Added cleaning in each fold:
- Filter out NaN/inf targets
- Replace infinity in features

### 3. Improved Error Messages (`src/models/predict.py`)

- Better error message when models haven't been trained
- Filter out models with absurdly high errors
- Clear guidance on data quality issues

## Next Steps to Fix

### Option 1: Fix Target Creation (Recommended)

Modify `src/features/feature_engineering.py` in `create_target_labels()`:

```python
def create_target_labels(feature_panel: pd.DataFrame, target_metric: str = "net_assets") -> pd.DataFrame:
    df = feature_panel.copy()
    df = df.sort_values(["cik", "period"])

    if target_metric not in df.columns:
        raise ValueError(f"Target metric '{target_metric}' not found in feature panel")

    # Get next quarter value
    next_qtr = df.groupby("cik")[target_metric].shift(-1)
    
    # Compute percentage change, handling edge cases
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_change = (next_qtr - df[target_metric]) / df[target_metric].replace(0, np.nan) * 100
        abs_change = next_qtr - df[target_metric]
    
    # Replace inf with NaN
    pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
    
    df["target_next_qtr_pct"] = pct_change
    df["target_next_qtr_abs"] = abs_change

    return df
```

### Option 2: Use Absolute Change Instead

If percentage change is too noisy, use absolute change:
```python
target_col = "target_next_qtr_abs"  # Instead of target_next_qtr_pct
```

### Option 3: Filter Target Values

Add filtering in training to only use reasonable target values:
```python
# In train.py, after loading data:
train_df = train_df[
    (train_df[target_col].notna()) &
    (train_df[target_col] > -100) &  # Filter extreme negative changes
    (train_df[target_col] < 100)     # Filter extreme positive changes
].copy()
```

## Current Status

- ✅ Data cleaning code added
- ✅ Error handling improved
- ⚠️ Target creation needs fixing
- ⚠️ Models need retraining after target fix

## Testing After Fix

1. Fix target creation (Option 1 above)
2. Regenerate feature panel:
   ```bash
   python -m src.features.feature_engineering
   ```
3. Retrain models:
   ```bash
   python -m src.models.train
   ```
4. Generate forecasts:
   ```bash
   python -m src.models.predict
   ```

## Expected Results After Fix

- Target should have valid numeric values (not all NaN)
- Models should train successfully
- MAE should be reasonable (likely 2-5% for percentage change)
- Forecasts should be generated successfully

