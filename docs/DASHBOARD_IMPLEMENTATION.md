# Dashboard Implementation: BDC Analytics & Forecasting Platform

## Overview

A comprehensive, interactive Streamlit dashboard that provides beautiful visualizations and insights into BDC performance, portfolio composition, and forecasting.

---

## Architecture

### File Structure

```
dashboards/
├── streamlit_app.py    # Main dashboard application
├── utils.py            # Data loading and utility functions
└── README.md           # Dashboard documentation

scripts/
└── run_dashboard.py    # Convenience script to launch dashboard
```

---

## Features by Tab

### 📈 Overview Tab

**Purpose**: Single BDC deep-dive with key metrics and forecasts

**Components**:
1. **BDC Selector**: Dropdown to select any BDC from 185 available entities
2. **Key Metrics Row** (4 columns):
   - Total Assets (formatted as currency)
   - Net Assets (formatted as currency)
   - Net Investment Income (formatted as currency)
   - Next Quarter Forecast (with delta indicator)
3. **Historical Trends** (2 charts):
   - Total Assets over time (interactive line chart)
   - Net Investment Income over time (interactive line chart)
4. **Leverage & Efficiency Metrics** (3 columns):
   - Debt to Assets ratio
   - NII to Assets ratio
   - Debt to Equity ratio
5. **Forecast Details**:
   - Forecast explanation
   - Projected Net Assets calculation

**Visual Design**:
- Clean metric cards with borders
- Interactive Plotly charts with hover tooltips
- Color-coded forecast indicators (green for positive, red for negative)
- Responsive column layout

---

### 💼 Portfolio Tab

**Purpose**: Analyze portfolio composition and characteristics

**Components**:
1. **BDC Selector**: Same as Overview tab
2. **Portfolio Characteristics** (2 columns):
   - **Fair Value vs Cost Bar Chart**: Visual comparison of portfolio valuation
   - **Top 10 Concentration**: Percentage metric
3. **Rate Type Distribution**:
   - **Pie Chart**: Floating rate vs Fixed rate exposure (donut chart)
   - **Portfolio HHI**: Herfindahl-Hirschman Index for concentration
4. **Sector Exposure**:
   - Top sector weight percentage
5. **Valuation Gap**:
   - Fair Value - Cost Gap with delta indicator

**Visual Design**:
- Bar charts with custom colors (#1f77b4 for fair value, #ff7f0e for cost)
- Donut pie chart for rate distribution
- Metric cards with clear labels

---

### ⚖️ Compare Tab

**Purpose**: Side-by-side comparison of multiple BDCs

**Components**:
1. **Multi-Select BDC Picker**: Select 2-5 BDCs to compare
2. **Comparison Table**: 
   - Columns: BDC, Total Assets, Net Assets, NII, Debt/Assets, Forecast
   - Formatted currency and percentages
   - Sortable and scrollable
3. **Historical Comparison Charts** (2 charts):
   - Total Assets comparison (multi-line chart)
   - Net Investment Income comparison (multi-line chart)
   - Each BDC has a different color
   - Interactive with hover tooltips

**Visual Design**:
- Multi-line charts with distinct colors per BDC
- Clear legend for identification
- Responsive table layout

---

### 🤖 Model Performance Tab

**Purpose**: View ML model performance and backtest results

**Components**:
1. **Best Model Summary** (4 metrics):
   - Best Model name
   - Best MAE
   - Best RMSE
   - Best R²
2. **Model Comparison Table**:
   - All 16 models with metrics
   - Sorted by MAE (best first)
   - Columns: MAE, RMSE, MAPE, R², n_samples
3. **Performance Visualizations** (2 charts):
   - **MAE Bar Chart**: Top 10 models by Mean Absolute Error
   - **R² Bar Chart**: Top 10 models by R² score (color-coded)
4. **Backtest Results**:
   - Aggregated prediction statistics by model
   - Mean and std of predictions vs actuals

**Visual Design**:
- Bar charts with rotated x-axis labels for readability
- Color gradients for R² visualization
- Clear metric displays

---

## Technical Implementation

### Data Loading

All data loading functions use `@st.cache_data` decorator for performance:
- `load_feature_panel()`: Main feature panel
- `load_quarterly_panel()`: Quarterly aggregated data
- `load_entity_reference()`: BDC master table
- `load_forecasts()`: Latest forecasts
- `load_model_comparison()`: Model metrics
- `load_backtest_predictions()`: Historical predictions

### Utility Functions

- `get_bdc_list()`: Get sorted list of BDC names
- `get_latest_period()`: Get most recent period in data
- `format_currency()`: Format numbers as $M, $B, $K
- `format_percentage()`: Format numbers as percentages
- `get_bdc_data()`: Filter data for specific BDC
- `get_bdc_forecast()`: Get forecast for specific BDC

### Styling

**Custom CSS**:
- Main header with blue accent color
- Metric cards with borders and shadows
- Forecast color coding (green/red)
- Professional color scheme

**Plotly Theme**:
- White background for charts
- Consistent color palette
- Interactive hover tooltips
- Responsive sizing

---

## Design Principles

1. **User-Friendly**: Clear labels, intuitive navigation, helpful tooltips
2. **Informative**: Comprehensive metrics without overwhelming
3. **Visual**: Charts and graphs for easy pattern recognition
4. **Interactive**: Hover tooltips, zoom, pan capabilities
5. **Professional**: Clean design suitable for presentations
6. **Responsive**: Works on different screen sizes

---

## Color Scheme

- **Primary Blue**: #1f77b4 (main accent)
- **Success Green**: #28a745 (positive indicators)
- **Warning Orange**: #ff7f0e (cost/alternative metrics)
- **Danger Red**: #dc3545 (negative indicators)
- **Background**: White (#ffffff)
- **Card Background**: Light gray (#f0f2f6)

---

## Usage

### Launch Dashboard

```bash
# Option 1: Direct Streamlit command
streamlit run dashboards/streamlit_app.py

# Option 2: Using convenience script
python scripts/run_dashboard.py
```

### Navigation

1. Use tabs at the top to switch between views
2. Select BDCs from dropdowns in each tab
3. Hover over charts for detailed tooltips
4. Use sidebar (if added) for additional filters

---

## Data Flow

```
Data Files (Parquet/CSV)
    ↓
@st.cache_data functions
    ↓
DataFrames loaded into memory
    ↓
Filtered/transformed for display
    ↓
Plotly/Streamlit components
    ↓
Rendered in browser
```

---

## Performance Optimizations

1. **Caching**: All data loading functions are cached
2. **Lazy Loading**: Data only loaded when needed
3. **Efficient Filtering**: Pandas operations for fast filtering
4. **Chart Optimization**: Plotly charts are efficient for large datasets

---

## Future Enhancements

Potential additions:
1. **Export Functionality**: Download charts/data as PDF/CSV
2. **Time Range Selector**: Filter by date range
3. **Advanced Filters**: Filter by sector, size, performance
4. **Alert System**: Notify on significant forecast changes
5. **Portfolio Builder**: Create custom BDC portfolios
6. **Scenario Analysis**: "What-if" analysis with parameter sliders
7. **SHAP Values**: Feature importance visualization
8. **Correlation Matrix**: Show relationships between metrics

---

## Dependencies

- `streamlit>=1.28.0`: Web framework
- `plotly>=5.17.0`: Interactive charts
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical operations

---

## Notes

- **Disclaimer**: Prominently displayed on every page
- **Data Freshness**: Uses latest period automatically
- **Error Handling**: Graceful handling of missing data
- **Accessibility**: Clear labels and tooltips for all elements

---

## Testing Checklist

- [x] All tabs render correctly
- [x] Data loads without errors
- [x] Charts display properly
- [x] BDC selector works
- [x] Forecasts display correctly
- [x] Model comparison shows all models
- [x] Comparison tab handles multiple BDCs
- [x] Responsive layout works
- [x] No console errors

---

The dashboard is production-ready and provides a comprehensive, beautiful interface for BDC analytics and forecasting! 🎉

