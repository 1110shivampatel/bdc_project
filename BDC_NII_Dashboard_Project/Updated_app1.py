"""
Streamlit multi-page-style dashboard for BDC financial analytics & forecasting.
REFACTORED VERSION with consolidated pages and multi-company comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# ---------------------------
# BDC-Specific Configuration
# ---------------------------

COLOR_SCHEME = {
    'positive': '#00D100',
    'negative': '#FF4B4B', 
    'neutral': '#1F77B4',
    'forecast': '#FF7F0E',
    'confidence': 'rgba(255, 127, 14, 0.2)',
    'sectors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
}

# ---------------------------
# Enhanced Utilities & Caching
# ---------------------------

@st.cache_data
def load_csv_if_exists(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as e:
        return None

@st.cache_data
def parse_financial_period(period_str):
    """Robust quarter parsing for financial data"""
    if pd.isna(period_str):
        return None
    
    period_str = str(period_str).strip().upper()
    
    # Handle YYYYQN format (2025Q3)
    if 'Q' in period_str and len(period_str) in [5, 6]:
        try:
            year = int(period_str[:4]) if period_str[:4].isdigit() else int(period_str[-4:])
            quarter = int(period_str.split('Q')[-1][0])
            quarter_end_month = quarter * 3
            return datetime(year, quarter_end_month, 1) + timedelta(days=32) - timedelta(days=1)
        except:
            pass
    
    # Handle QN YYYY format (Q3 2025)
    if period_str.startswith('Q') and len(period_str) in [6, 7]:
        try:
            parts = period_str.split()
            quarter = int(parts[0][1])
            year = int(parts[1])
            quarter_end_month = quarter * 3
            return datetime(year, quarter_end_month, 1) + timedelta(days=32) - timedelta(days=1)
        except:
            pass
    
    # Fallback to pandas parser
    try:
        return pd.to_datetime(period_str, errors='coerce')
    except:
        return None

@st.cache_data
def parse_date_columns(df, dataset_type="extended"):
    """Enhanced date parsing with financial period support"""
    if df is None:
        return None
    
    df = df.copy()
    
    try:
        # Handle different dataset structures
        if dataset_type == "extended" or dataset_type == "training":
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
        elif dataset_type == "forecast":
            if 'Quarter' in df.columns:
                # Convert Quarter column to datetime
                df['Quarter'] = df['Quarter'].apply(parse_financial_period)
                # For any that failed, set default dates based on quarter string
                mask = df['Quarter'].isna()
                if mask.any():
                    for idx in df[mask].index:
                        quarter_str = str(df.loc[idx, 'Quarter']).upper()
                        if '2025Q3' in quarter_str:
                            df.loc[idx, 'Quarter'] = datetime(2025, 9, 30)
                        elif '2025Q4' in quarter_str:
                            df.loc[idx, 'Quarter'] = datetime(2025, 12, 31)
                        else:
                            df.loc[idx, 'Quarter'] = datetime(2025, 12, 31)  # Default
    except Exception as e:
        st.warning(f"Date parsing encountered issues: {str(e)}")
    
    return df

def df_info(df):
    if df is None:
        return "Not loaded"
    return f"{df.shape[0]:,} rows × {df.shape[1]:,} cols"

def download_link(df: pd.DataFrame, filename: str, link_text: str):
    """Create a download link for a dataframe as CSV"""
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def calculate_portfolio_summary(df, dataset_type="extended"):
    """Calculate portfolio-level summary statistics"""
    if df is None:
        return {}
    
    summary = {}
    
    if dataset_type == "extended" or dataset_type == "training":
        if 'name' in df.columns:
            summary['num_companies'] = df['name'].nunique()
        
        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
            summary['date_range'] = {
                'start': df['date'].min(),
                'end': df['date'].max()
            }
        
        if 'value' in df.columns:
            # Get latest value for each company
            latest_dates = df.groupby('name')['date'].idxmax()
            latest_data = df.loc[latest_dates]
            summary['total_portfolio_value'] = latest_data['value'].sum()
            summary['avg_company_value'] = latest_data['value'].mean()
            
            # Calculate growth rates for companies with multiple periods
            growth_rates = []
            for company in df['name'].unique():
                company_data = df[df['name'] == company].sort_values('date')
                if len(company_data) > 1:
                    latest_val = company_data['value'].iloc[-1]
                    earliest_val = company_data['value'].iloc[0]
                    if earliest_val > 0:
                        growth_rate = (latest_val - earliest_val) / earliest_val * 100
                        growth_rates.append(growth_rate)
            
            if growth_rates:
                summary['avg_growth_rate'] = np.mean(growth_rates)
    
    elif dataset_type == "forecast":
        if 'BDC' in df.columns:
            summary['num_companies_forecast'] = df['BDC'].nunique()
        
        if 'Predicted_NII' in df.columns:
            summary['avg_predicted_nii'] = df['Predicted_NII'].mean()
    
    elif dataset_type == "summary":
        if 'BDC' in df.columns:
            summary['num_companies_summary'] = df['BDC'].nunique()
        
        if 'Growth_Pct' in df.columns:
            summary['avg_growth_rate'] = df['Growth_Pct'].mean()
            summary['positive_growth_companies'] = (df['Growth_Pct'] > 0).sum()
    
    return summary

def get_latest_company_data(df, dataset_type="extended"):
    """Get most recent data for each company"""
    if dataset_type == "extended" or dataset_type == "training":
        if 'name' not in df.columns or 'date' not in df.columns:
            return None
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            return None
        
        idx = df.groupby('name')['date'].idxmax()
        return df.loc[idx]
    
    return None

# ---------------------------
# Enhanced Visualization Components
# ---------------------------

def create_forecast_timeseries(historical_df, forecast_df, company):
    """Create continuous timeline with confidence intervals"""
    fig = go.Figure()
    
    # Add historical data
    if historical_df is not None and 'name' in historical_df.columns and 'date' in historical_df.columns:
        hist_company_data = historical_df[historical_df['name'] == company].copy()
        if not hist_company_data.empty and 'value' in historical_df.columns:
            hist_company_data = hist_company_data.sort_values('date')
            
            fig.add_trace(go.Scatter(
                x=hist_company_data['date'],
                y=hist_company_data['value'],
                mode='lines+markers',
                name=f'{company} - Historical NII',
                line=dict(color=COLOR_SCHEME['neutral'], width=2),
                marker=dict(size=6)
            ))
    
    # Add forecast points with confidence intervals
    if forecast_df is not None and 'BDC' in forecast_df.columns:
        fore_company_data = forecast_df[forecast_df['BDC'] == company]
        
        if not fore_company_data.empty:
            # Add forecast points with confidence intervals
            for _, row in fore_company_data.iterrows():
                forecast_date = None
                
                # Handle Quarter column safely
                if 'Quarter' in row:
                    if pd.notnull(row['Quarter']):
                        if hasattr(row['Quarter'], 'strftime'):  # It's a datetime
                            forecast_date = row['Quarter']
                        else:
                            # Parse quarter string
                            quarter_str = str(row['Quarter']).upper()
                            if '2025Q3' in quarter_str:
                                forecast_date = datetime(2025, 9, 30)
                            elif '2025Q4' in quarter_str:
                                forecast_date = datetime(2025, 12, 31)
                            else:
                                forecast_date = datetime(2025, 12, 31)  # Default
                
                if forecast_date is None:
                    forecast_date = datetime(2025, 12, 31)  # Final fallback
                
                # Main forecast point
                if pd.notnull(row.get('Predicted_NII')):
                    fig.add_trace(go.Scatter(
                        x=[forecast_date],
                        y=[row['Predicted_NII']],
                        mode='markers',
                        name=f'{company} - Forecast {quarter_str if "quarter_str" in locals() else "2025"}',
                        marker=dict(size=12, color=COLOR_SCHEME['forecast'], symbol='star'),
                        showlegend=True
                    ))
                
                # Confidence interval
                if (pd.notnull(row.get('Lower_85CI')) and 
                    pd.notnull(row.get('Upper_85CI')) and
                    pd.notnull(row.get('Predicted_NII'))):
                    
                    fig.add_trace(go.Scatter(
                        x=[forecast_date, forecast_date],
                        y=[row['Lower_85CI'], row['Upper_85CI']],
                        mode='lines',
                        line=dict(width=8, color=COLOR_SCHEME['confidence']),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    fig.update_layout(
        title=f"Historical vs Forecast NII - {company}",
        xaxis_title="Date",
        yaxis_title="Net Investment Income (NII)",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_multi_company_comparison(historical_df, forecast_df, selected_companies):
    """Create comparison chart for multiple companies"""
    fig = go.Figure()
    
    # Define color palette for multiple companies
    colors = px.colors.qualitative.Set1
    
    # Add historical data for each selected company
    for i, company in enumerate(selected_companies):
        if historical_df is not None and 'name' in historical_df.columns:
            hist_company_data = historical_df[historical_df['name'] == company].copy()
            if not hist_company_data.empty and 'value' in historical_df.columns:
                hist_company_data = hist_company_data.sort_values('date')
                
                fig.add_trace(go.Scatter(
                    x=hist_company_data['date'],
                    y=hist_company_data['value'],
                    mode='lines+markers',
                    name=f'{company} - Historical',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ))
    
    # Add forecast points for each selected company
    if forecast_df is not None and 'BDC' in forecast_df.columns:
        for i, company in enumerate(selected_companies):
            fore_company_data = forecast_df[forecast_df['BDC'] == company]
            
            if not fore_company_data.empty:
                for _, row in fore_company_data.iterrows():
                    forecast_date = None
                    
                    if 'Quarter' in row and pd.notnull(row['Quarter']):
                        if hasattr(row['Quarter'], 'strftime'):
                            forecast_date = row['Quarter']
                        else:
                            quarter_str = str(row['Quarter']).upper()
                            if '2025Q3' in quarter_str:
                                forecast_date = datetime(2025, 9, 30)
                            elif '2025Q4' in quarter_str:
                                forecast_date = datetime(2025, 12, 31)
                    
                    if forecast_date is None:
                        forecast_date = datetime(2025, 12, 31)
                    
                    if pd.notnull(row.get('Predicted_NII')):
                        fig.add_trace(go.Scatter(
                            x=[forecast_date],
                            y=[row['Predicted_NII']],
                            mode='markers',
                            name=f'{company} - Forecast',
                            marker=dict(
                                size=12, 
                                color=colors[i % len(colors)], 
                                symbol='star',
                                line=dict(width=2, color='black')
                            ),
                            showlegend=True
                        ))
    
    fig.update_layout(
        title="Multi-Company NII Comparison",
        xaxis_title="Date",
        yaxis_title="Net Investment Income (NII)",
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    
    return fig

def create_portfolio_composition_chart(df):
    """Create portfolio allocation visualization"""
    if df is None or 'name' not in df.columns or 'value' not in df.columns:
        return None
    
    latest_data = get_latest_company_data(df)
    if latest_data is None:
        return None
    
    # Top 10 companies by value
    composition = latest_data.nlargest(10, 'value')
    
    fig = px.pie(
        composition, 
        values='value', 
        names='name',
        title="Top 10 BDCs by Net Investment Income",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def generate_correlation_insights(corr):
    """Generate human-readable insights with trend direction arrows."""
    insights = []

    strong_pos = []
    strong_neg = []
    moderate = []

    # Extract correlations
    for c1 in corr.columns:
        for c2 in corr.columns:
            if c1 >= c2:
                continue  # skip duplicates + diagonal

            value = corr.loc[c1, c2]

            if value >= 0.70:
                strong_pos.append((c1, c2, value))
            elif value <= -0.40:
                strong_neg.append((c1, c2, value))
            elif 0.30 <= abs(value) < 0.70:
                moderate.append((c1, c2, value))

    # ---- Strong Positive (⬆️⬆️) ----
    if strong_pos:
        insights.append("📈 **Strong Positive Relationships (move together)**")
        for a, b, v in strong_pos[:6]:
            arrow = "⬆️⬆️"
            insights.append(f"- {arrow} **{a}** and **{b}** rise and fall together (**{v}**).")

    # ---- Strong Negative (⬇️⬇️) ----
    if strong_neg:
        insights.append("📉 **Strong Negative Relationships (move opposite)**")
        for a, b, v in strong_neg[:6]:
            arrow = "⬇️⬇️"
            insights.append(f"- {arrow} **{a}** and **{b}** move in opposite directions (**{v}**).")

    # ---- Moderate Correlations (➡️) ----
    if moderate:
        insights.append("➡️ **Moderate Trend Relationships**")
        for a, b, v in moderate[:6]:
            arrow = "➡️"
            direction = "slightly together" if v > 0 else "slightly opposite"
            insights.append(f"- {arrow} **{a}** and **{b}** trend {direction} (**{v}**).")

    # If nothing meaningful found
    if not insights:
        insights.append("No significant correlation trends detected among companies.")

    return "\n".join(insights)



def create_risk_heatmap(df):
    """Create correlation heatmap AND auto-generate insights."""
    if df is None:
        return None, None  # return (fig, insights)

    # Ensure required columns exist
    if 'name' in df.columns and 'date' in df.columns and 'value' in df.columns:
        # Pivot companies vs dates
        pivot_df = df.pivot_table(index='date', columns='name', values='value')

        # Correlation matrix
        corr_matrix = pivot_df.corr().round(2)

        # Limit to top 15 companies for readability
        top_companies = corr_matrix.sum().nlargest(15).index
        corr_matrix = corr_matrix.loc[top_companies, top_companies]

        # ---- Plotly Heatmap ----
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="BDC Correlation Matrix (NII Trends)",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=600)

        # ---- Insight Extraction ----
        insights = generate_correlation_insights(corr_matrix)

        return fig, insights

    return None, None


def create_performance_small_multiples(df, top_n=6):
    """Create small multiples grid for top BDCs"""
    if df is None or 'name' not in df.columns or 'date' not in df.columns or 'value' not in df.columns:
        return None
    
    latest_data = get_latest_company_data(df)
    if latest_data is None:
        return None
    
    # Get top companies by value
    top_companies = latest_data.nlargest(top_n, 'value')['name'].tolist()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=top_companies,
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    for i, company in enumerate(top_companies):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        company_data = df[df['name'] == company].sort_values('date')
        
        fig.add_trace(
            go.Scatter(
                x=company_data['date'],
                y=company_data['value'],
                mode='lines',
                name=company,
                line=dict(width=2)
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text="NII Trends - Top 6 BDCs",
        showlegend=False
    )
    
    # Update y-axis titles
    for i in range(1, 7):
        fig.update_yaxes(title_text="NII", row=(i-1)//3 + 1, col=(i-1)%3 + 1)
    
    return fig

def create_growth_comparison_chart(summary_df):
    """Create growth comparison visualization"""
    if summary_df is None or 'BDC' not in summary_df.columns or 'Growth_Pct' not in summary_df.columns:
        return None
    
    # Sort by growth rate
    sorted_df = summary_df.sort_values('Growth_Pct', ascending=False)
    
    fig = px.bar(
        sorted_df.head(15),
        x='Growth_Pct',
        y='BDC',
        orientation='h',
        title="Top 15 BDCs by Forecasted Growth Rate (Q3 to Q4 2025)",
        color='Growth_Pct',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Growth Percentage (%)",
        yaxis_title=""
    )
    
    return fig

# ---------------------------
# Load available files with proper parsing
# ---------------------------
st.set_page_config(layout="wide", page_title="BDC NII Analytics & Forecasting", initial_sidebar_state="expanded")

st.title("🏦 BDC Net Investment Income (NII) Dashboard")

with st.spinner("Loading and enhancing BDC NII data..."):
    df_extended = load_csv_if_exists("bdc_nii_extended_data_2022q4_2025m7.csv")
    df_forecast = load_csv_if_exists("bdc_nii_predictions_2025_q3_q4_final.csv")
    df_training = load_csv_if_exists("bdc_nii_quarterly_training_data.csv")
    df_summary = load_csv_if_exists("bdc_nii_predictions_summary_by_bdc.csv")

# Enhanced date parsing for each dataset type
df_extended = parse_date_columns(df_extended, "extended")
df_forecast = parse_date_columns(df_forecast, "forecast")
df_training = parse_date_columns(df_training, "training")
df_summary = parse_date_columns(df_summary, "summary")

# ---------------------------
# Enhanced Sidebar
# ---------------------------
st.sidebar.header("📊 Navigation")
page = st.sidebar.selectbox("Go to", [
    "Overview", 
    "Portfolio Analytics", 
    "Company Analysis & Forecasting", 
    "Insights & Risk", 
    "Conclusion"
])

st.sidebar.markdown("---")

# ---------------------------
# Page: Portfolio Dashboard (Forecasted Companies Only)
# ---------------------------
if page == "Overview":
    st.header("📊 Dashboard")
    
    if df_extended is not None and df_forecast is not None and 'BDC' in df_forecast.columns:
        # Filter historical data for forecasted companies only
        forecasted_companies = df_forecast['BDC'].unique()
        df_extended_filtered = df_extended[df_extended['name'].isin(forecasted_companies)]
        df_summary_filtered = df_summary[df_summary['BDC'].isin(forecasted_companies)] if df_summary is not None else None

        # ---------------------------
        # KPI Summary Cards
        # ---------------------------
        st.subheader("Portfolio KPIs")
        summary = calculate_portfolio_summary(df_extended_filtered, "extended")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Companies", summary.get('num_companies', 'N/A'))
        with col2:
            portfolio_value = summary.get('total_portfolio_value', 0)
            st.metric("Total NII", f"${portfolio_value:,.0f}" if portfolio_value else "N/A")
        with col3:
            avg_value = summary.get('avg_company_value', 0)
            st.metric("Avg Company NII", f"${avg_value:,.0f}" if avg_value else "N/A")
        with col4:
            forecast_summary = df_forecast.groupby('Quarter')['Predicted_NII'].sum().reset_index()
            st.metric("Projected Growth Rate (Q3 to Q4)", f"{((forecast_summary['Predicted_NII'].iloc[-1] - forecast_summary['Predicted_NII'].iloc[0]) / forecast_summary['Predicted_NII'].iloc[0]*100):.2f}%")
        
        st.markdown("---")

        # ---------------------------
        # Mini Trend Tiles - Key Highlights
        # ---------------------------
        st.markdown("### 🌟 Portfolio Highlights")
        col1, col2 = st.columns(2)

        # Latest NII for filtered companies
        latest_vals = df_extended_filtered.groupby('name')['value'].last()

        # Top Performer by Latest NII
        top_company = latest_vals.idxmax()
        top_value = latest_vals.max()
        with col1:
            st.markdown(f"""
                <div style="
                    padding:12px;
                    border-radius:12px;
                    background:#19A34A;
                    color:white;
                    font-weight:600;
                    text-align:center;">
                    🚀 Top Performer<br>
                    {top_company}<br>
                    ${top_value:,.0f}
                </div>
            """, unsafe_allow_html=True)

        # Highest Forecast Growth
        forecast_growth_vals = {}
        for company in latest_vals.index:
            fdata = df_forecast[df_forecast['BDC'] == company]
            if not fdata.empty:
                forecast_growth_vals[company] = ((fdata['Predicted_NII'].iloc[0] - latest_vals[company])/latest_vals[company]*100)
        if forecast_growth_vals:
            highest_forecast_company = max(forecast_growth_vals, key=forecast_growth_vals.get)
            highest_forecast_value = forecast_growth_vals[highest_forecast_company]
            with col2:
                st.markdown(f"""
                    <div style="
                        padding:12px;
                        border-radius:12px;
                        background:#FFD700;
                        color:black;
                        font-weight:600;
                        text-align:center;">
                        🔮 Highest Forecast Growth<br>
                        {highest_forecast_company}<br>
                        {highest_forecast_value:.1f}%
                    </div>
                """, unsafe_allow_html=True)

        # ---------------------------
        # Portfolio Composition + Risk Metrics
        # ---------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Quick Insights")
            st.metric("Companies Forecasted", len(forecasted_companies))
            if df_summary_filtered is not None and 'Growth_Pct' in df_summary_filtered.columns:
                positive_growth = (df_summary_filtered['Growth_Pct'] > 0).sum()
                total_companies = len(df_summary_filtered)
                st.metric("Positive Growth Forecast", f"{positive_growth}/{total_companies}")

        with col2:
            st.subheader("Portfolio Composition")
            composition_chart = create_portfolio_composition_chart(df_extended_filtered)
            if composition_chart:
                st.plotly_chart(composition_chart, use_container_width=True)
            else:
                st.info("Portfolio composition data not available")

        with col3:
            st.subheader("Risk & Forecast Metrics")
            portfolio_values = latest_vals
            total_value = portfolio_values.sum()
            market_shares = (portfolio_values / total_value) * 100
            hhi = (market_shares**2).sum()
            top5_conc = market_shares.nlargest(5).sum()
            
            st.metric("HHI Concentration", f"{hhi:.0f}")
            st.metric("Top 5 Concentration", f"{top5_conc:.1f}%")
            
            # Alerts
            negative_growth = (df_summary_filtered['Growth_Pct'] < 0).sum() if df_summary_filtered is not None else 0
            if negative_growth > 0:
                st.warning(f"⚠️ {negative_growth} companies forecast negative growth.")
            
            st.markdown("### Portfolio Insights")
            st.info(
                "HHI shows portfolio concentration: higher HHI → higher risk of overexposure.\n"
                "Top 5 concentration indicates dependency on few companies.\n"
                "Alerts highlight companies expected to underperform.\n"
                "All metrics reflect only forecasted companies."
            )

        st.markdown("---")

        # ---------------------------
        # Performance Overview - Small Multiples
        # ---------------------------
        st.subheader("Performance Overview - BDCs")
        small_multiples = create_performance_small_multiples(df_extended_filtered)
        if small_multiples:
            st.plotly_chart(small_multiples, use_container_width=True)
        else:
            st.info("Insufficient data for performance overview")
        
        st.markdown("---")

        # ---------------------------
        # Portfolio-Level Forecast (Cumulative)
        # ---------------------------
        st.subheader("Portfolio Forecast Overview")
        forecast_summary = df_forecast.groupby('Quarter')['Predicted_NII'].sum().reset_index()
        forecast_chart = px.line(
            forecast_summary,
            x='Quarter',
            y='Predicted_NII',
            markers=True,
            title="Cumulative Portfolio Forecast (NII)"
        )
        forecast_chart.update_layout(yaxis_title="Total NII ($)")
        st.plotly_chart(forecast_chart, use_container_width=True)
        st.info(
            f"Projected portfolio growth: {((forecast_summary['Predicted_NII'].iloc[-1] - forecast_summary['Predicted_NII'].iloc[0]) / forecast_summary['Predicted_NII'].iloc[0]*100):.2f}% from first to last forecast quarter."
        )

        # ---------------------------
        # Historical Portfolio NII (Forecasted Companies Only)
        # ---------------------------
        st.subheader("Portfolio Historic Overview")
        hist_portfolio = df_extended_filtered.groupby('date')['value'].sum().reset_index()
        hist_chart = px.line(
            hist_portfolio,
            x='date',
            y='value',
            markers=True,
            title="Historic Portfolio (NII) - Forecasted Companies"
        )
        hist_chart.update_layout(
            yaxis_title="Total NII ($)",
            xaxis_title="Date",
            hovermode="x unified"
        )
        st.plotly_chart(hist_chart, use_container_width=True)

        growth_hist = ((hist_portfolio['value'].iloc[-1] - hist_portfolio['value'].iloc[0]) / hist_portfolio['value'].iloc[0]) * 100

        if growth_hist >= 0:
            st.info(f"Portfolio has grown {growth_hist:.2f}% over the historical period.")
        else:
            st.info(f"Portfolio has declined {growth_hist:.2f}% over the historical period.")


    else:
        st.info("Upload extended dataset and forecast data to see portfolio dashboard")



# ---------------------------
# Page: Portfolio Analytics (Enhanced Multi-Company Visualization)
# ---------------------------
elif page == "Portfolio Analytics":
    st.header("🔍 Portfolio Analytics")

    if df_extended is None or df_forecast is None:
        st.info("Extended historical dataset or forecast dataset not available.")
    else:
        # Filter companies to only those with forecast data
        forecast_companies = df_forecast['BDC'].unique()
        df_extended_filtered = df_extended[df_extended['name'].isin(forecast_companies)]

        # Risk Heatmap
        st.subheader("BDC Correlation Analysis")
        heatmap_fig, heatmap_insights = create_risk_heatmap(df_extended_filtered)

        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
            if heatmap_insights:
                st.markdown("### 🧠 Trend Insights from Correlation Matrix")
                st.info(heatmap_insights)
        else:
            st.info("Insufficient data for correlation analysis")

        # Multi-Company Comparison
        st.markdown("---")
        st.subheader("📊 Multi-Company Comparison")

        companies = sorted(df_extended_filtered['name'].unique())
        selected_companies = st.multiselect(
            "Select companies to compare",
            options=companies,
            default=companies[:3] if len(companies) >= 3 else companies,
            help="Select multiple companies to compare their NII trends (only companies with forecast data are shown)"
        )

        show_forecast = st.checkbox("Show forecast comparison", value=True)

        if selected_companies:
            # --- 1. NII Trend Line Chart ---
            comparison_chart = create_multi_company_comparison(
                df_extended_filtered,
                df_forecast if show_forecast else None,
                selected_companies
            )
            st.plotly_chart(comparison_chart, use_container_width=True)

            # --- 2. Growth Bar Chart ---
            hist_growth_list = []
            forecast_growth_list = []
            for company in selected_companies:
                hist_data = df_extended_filtered[df_extended_filtered['name'] == company].sort_values('date')
                latest_val = hist_data['value'].iloc[-1] if not hist_data.empty else None
                earliest_val = hist_data['value'].iloc[0] if len(hist_data) > 1 else None
                hist_growth = ((latest_val - earliest_val) / earliest_val * 100) if earliest_val else 0
                hist_growth_list.append(hist_growth)

                forecast_val = None
                if show_forecast:
                    fdata = df_forecast[df_forecast['BDC'] == company]
                    if not fdata.empty:
                        forecast_val = fdata['Predicted_NII'].iloc[0]
                forecast_growth = ((forecast_val - latest_val)/latest_val*100) if latest_val and forecast_val else 0
                forecast_growth_list.append(forecast_growth)

            growth_fig = go.Figure()
            growth_fig.add_trace(go.Bar(
                x=selected_companies, y=hist_growth_list,
                name='Historical Growth (%)', marker_color='steelblue'
            ))
            if show_forecast:
                growth_fig.add_trace(go.Bar(
                    x=selected_companies, y=forecast_growth_list,
                    name='Forecast Growth → Q3 2025 (%)', marker_color='darkorange'
                ))
            growth_fig.update_layout(
                title="Historical vs Forecast Growth Comparison",
                xaxis_title="Company",
                yaxis_title="Growth (%)",
                barmode='group',
                height=450
            )
            st.plotly_chart(growth_fig, use_container_width=True)

            # --- 3. NII Contribution Pie Chart ---
            latest_vals_dict = {}
            for company in selected_companies:
                hist_data = df_extended_filtered[df_extended_filtered['name'] == company].sort_values('date')
                latest_val = hist_data['value'].iloc[-1] if not hist_data.empty else 0
                latest_vals_dict[company] = latest_val

            pie_fig = px.pie(
                names=list(latest_vals_dict.keys()),
                values=list(latest_vals_dict.values()),
                title="Contribution of Selected Companies to Latest NII"
            )
            st.plotly_chart(pie_fig, use_container_width=True)

            # --- 4. Metrics Table ---
            st.markdown("### 📈 Key Metrics")
            metrics_data = []
            for company in selected_companies:
                hist_data = df_extended_filtered[df_extended_filtered['name'] == company].sort_values('date')
                latest_val = hist_data['value'].iloc[-1] if not hist_data.empty else None
                earliest_val = hist_data['value'].iloc[0] if len(hist_data) > 1 else None
                hist_growth = ((latest_val - earliest_val)/earliest_val*100) if earliest_val else None

                forecast_val = df_forecast[df_forecast['BDC'] == company]['Predicted_NII'].iloc[0]
                forecast_growth = ((forecast_val - latest_val)/latest_val*100) if latest_val and forecast_val else None

                metrics_data.append({
                    "Company": company,
                    "Latest NII": f"${latest_val:,.0f}" if latest_val else "N/A",
                    "Historical Growth (%)": f"{hist_growth:.1f}%" if hist_growth else "N/A",
                    "Forecast Growth (%)": f"{forecast_growth:.2f}%" if forecast_growth else "N/A"
                })

            st.table(pd.DataFrame(metrics_data))

            # --- 5. Insights Section ---
            st.markdown("### 🧠 Insights")
            insights = []

            # Top latest NII
            top_company = max(latest_vals_dict, key=latest_vals_dict.get)
            insights.append(f"🚀 **{top_company}** currently has the highest latest NII among selected companies.")

            # Top historical growth
            growth_vals_dict = {c: ((df_extended_filtered[df_extended_filtered['name'] == c]['value'].iloc[-1] - df_extended_filtered[df_extended_filtered['name'] == c]['value'].iloc[0]) / df_extended_filtered[df_extended_filtered['name'] == c]['value'].iloc[0]*100)
                                for c in selected_companies if len(df_extended_filtered[df_extended_filtered['name'] == c]) > 1}
            if growth_vals_dict:
                best_growth_company = max(growth_vals_dict, key=growth_vals_dict.get)
                insights.append(f"📈 **{best_growth_company}** shows the strongest historical growth trend.")

            # Forecast risk insight
            risk_companies = [c for c, g in zip(selected_companies, forecast_growth_list) if g < 0]
            if risk_companies:
                insights.append(f"⚠️ Companies with negative forecast growth: {', '.join(risk_companies)}")

            st.info("\n".join(insights))

        else:
            st.info("Please select at least one company to see the comparison")


# ---------------------------
# Page: Company Analysis & Forecasting (Enhanced Vertical Layout)
# ---------------------------
elif page == "Company Analysis & Forecasting":
    st.header("🏢 Company Analysis & Forecasting")

    # ---- Base CSS ----
    st.markdown("""
        <style>
            .main-title { font-size: 24px; font-weight: 700; }
            .sub-title { font-size: 20px; font-weight: 600; margin-bottom: 6px; }
            .glass-tile {
                background: rgba(255, 255, 255, 0.55);
                backdrop-filter: blur(8px);
                border-radius: 14px;
                padding: 18px;
                border: 1px solid rgba(200,200,200,0.40);
                box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
                margin-bottom: 12px;
            }
            .metric-title { font-size: 15px; font-weight: 600; color: #444; }
            .metric-value { font-size: 30px; font-weight: 700; color: #111; margin-top: -4px; }
            .metric-sub { font-size: 13px; color: #666; margin-top: 2px; }
        </style>
    """, unsafe_allow_html=True)

    if df_extended is None or df_forecast is None:
        st.info("Extended historical dataset or forecast dataset not available.")
    else:
         # Only allow companies with forecast data
        forecast_companies = df_forecast['BDC'].unique()
        df_extended_filtered = df_extended[df_extended['name'].isin(forecast_companies)]
        companies = sorted(df_extended_filtered['name'].unique())
        if not companies:
            st.info("No companies with forecast data available.")
        else:
            sel = st.selectbox("Choose company", companies)

            st.markdown(f"<div class='main-title'>📈 {sel} — NII Analysis & Forecast</div>", unsafe_allow_html=True)

            tmp = df_extended[df_extended['name'] == sel].copy()
            latest_data = get_latest_company_data(tmp)

            # -----------------------------------------
            # MAIN CHART (TOP)
            # -----------------------------------------
            st.markdown("<div class='sub-title'>NII Trend & Forecast Visualization</div>", unsafe_allow_html=True)
            forecast_chart = create_forecast_timeseries(df_extended, df_forecast, sel)
            st.plotly_chart(forecast_chart, use_container_width=True)

            # ====================================================================
            # *** KEY METRICS SECTION MOVED BENEATH VISUALIZATION (Vertical Layout)
            # ====================================================================
            st.markdown("### 🔍 Key Metrics")

            if latest_data is not None and not latest_data.empty:
                latest_val = latest_data['value'].iloc[0]

                # ROW 1
                colA, colB, colC = st.columns(3)
                colA.metric("Latest NII", f"${latest_val:,.0f}")

                company_full_data = tmp.sort_values('date')
                if len(company_full_data) > 1:
                    growth = ((latest_val - company_full_data['value'].iloc[0]) /
                              company_full_data['value'].iloc[0]) * 100
                    colB.metric("Historical Growth", f"{growth:.1f}%")
                else:
                    colB.metric("Historical Growth", "N/A")

                colC.metric("Historical Periods", len(tmp))

                # ROW 2
                if "date" in latest_data.columns:
                    date_val = latest_data["date"].iloc[0].strftime("%Y-%m-%d")
                    st.metric("As of", date_val)

            # ====================================================================
            # FORECAST SUMMARY (Under Key Metrics)
            # ====================================================================
            if df_summary is not None and 'BDC' in df_summary.columns:
                company_summary = df_summary[df_summary['BDC'] == sel]
                if not company_summary.empty:

                    st.markdown("### 🔮 Forecast Summary")

                    q3_pred = company_summary['Q3_2025_Pred'].iloc[0]
                    q4_pred = company_summary['Q4_2025_Pred'].iloc[0]
                    growth_pct = company_summary['Growth_Pct'].iloc[0]

                    colF1, colF2, colF3 = st.columns(3)
                    colF1.metric("Q3 2025 Forecast", f"${q3_pred:,.0f}")
                    colF2.metric("Q4 2025 Forecast", f"${q4_pred:,.0f}")
                    colF3.metric("Q3 → Q4 Growth", f"{growth_pct:.2f}%")

                    if latest_data is not None and not latest_data.empty:
                        hist_to_q3_growth = ((q3_pred - latest_val) / latest_val) * 100
                        st.metric("Latest → Q3 Growth", f"{hist_to_q3_growth:.2f}%")

            # ====================================================================
            # INSIGHT TILES (Glass UI)
            # ====================================================================
            if df_summary is not None and 'BDC' in df_summary.columns:
                company_summary = df_summary[df_summary['BDC'] == sel]
                if not company_summary.empty:

                    q4_rank_df = df_summary.sort_values('Q4_2025_Pred', ascending=False)
                    rank = q4_rank_df[q4_rank_df['BDC'] == sel].index[0] + 1
                    total_companies = len(q4_rank_df)

                    company_growth = company_summary['Growth_Pct'].iloc[0]
                    avg_growth = df_summary['Growth_Pct'].mean()
                    diff_from_avg = company_growth - avg_growth
                    growth_percentile = (df_summary['Growth_Pct'] < company_growth).mean() * 100

                    capsule_color = "#19A34A" if diff_from_avg > 0 else "#E64B4B"
                    capsule_text = f"{diff_from_avg:+.2f}%"
                    growth_category = "Outperforming Average" if diff_from_avg > 0 else "Below Sector Average"
                    growth_icon = "📈" if diff_from_avg > 0 else "📉"

                    capsule_html = f"""
                        <span style="
                            padding:6px 14px;
                            border-radius:20px;
                            background:{capsule_color};
                            font-weight:600;
                            color:white;
                            font-size:14px;">
                            {capsule_text}
                        </span>
                    """

                    st.markdown("---")
                    st.markdown("### 📊 Forecast Insights")

                    colA, colB, colC = st.columns(3)

                    with colA:
                        st.markdown(f"""
                            <div class="glass-tile">
                                <div class="metric-title">⭐ Q4 Forecast Rank</div>
                                <div class="metric-value">{rank}/{total_companies}</div>
                                <div class="metric-sub">Among all BDCs</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with colB:
                        icon = "🚀" if growth_percentile > 70 else "📊"
                        st.markdown(f"""
                            <div class="glass-tile">
                                <div class="metric-title">{icon} Growth Percentile</div>
                                <div class="metric-value">{growth_percentile:.1f}%</div>
                                <div class="metric-sub">Higher than {growth_percentile:.1f}% peers</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with colC:
                        st.markdown(f"""
                            <div class="glass-tile">
                                <div class="metric-title">{growth_icon} vs BDC market Avg</div>
                                <div style="margin-top:4px;">{capsule_html}</div>
                                <div class="metric-sub">{growth_category}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    # Summary card
                    summary = (
                        f"📈 <b>{sel} is outperforming the overall BDC market</b> heading into Q4."
                        if diff_from_avg > 0 else
                        f"📉 <b>{sel} is trending below the overall BDC market</b> with softer forecast momentum."
                    )

                    st.markdown(f"""
                        <div style="
                            margin-top:15px;
                            padding:14px 18px;
                            border-radius:12px;
                            background:#f8f9fa;
                            border-left:5px solid {capsule_color};
                            font-size:15px;
                            color:#333;">
                            {summary}
                        </div>
                    """, unsafe_allow_html=True)

            # ====================================================================
            # Detailed Forecast Table
            # ====================================================================
            if df_forecast is not None and 'BDC' in df_forecast.columns:
                fore_data = df_forecast[df_forecast['BDC'] == sel]
                if not fore_data.empty:
                    st.markdown("### Detailed Forecast Data")

                    display_forecast = fore_data.copy()
                    if 'Quarter' in display_forecast.columns:
                        display_forecast['Quarter'] = display_forecast['Quarter'].astype(str)

                    numeric_cols = ['Predicted_NII', 'Lower_85CI', 'Upper_85CI', 'Most_Recent_NII']
                    for col in numeric_cols:
                        if col in display_forecast.columns:
                            display_forecast[col] = display_forecast[col].apply(
                                lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                            )

                    st.dataframe(display_forecast)



# ---------------------------
# Page: Insights & Risk (Enhanced Insights)
# ---------------------------
elif page == "Insights & Risk":
    st.header("💡 Insights & Risk Analytics")
    
    # Growth Analysis from Summary Data
    if df_summary is not None and 'BDC' in df_summary.columns and 'Growth_Pct' in df_summary.columns:
        st.subheader("📊 Growth Forecast Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_growth = df_summary['Growth_Pct'].mean()
            st.metric("Average Growth", f"{avg_growth:.2f}%")
        with col2:
            positive_count = (df_summary['Growth_Pct'] > 0).sum()
            total_count = len(df_summary)
            st.metric("Positive Growth", f"{positive_count}/{total_count}")
        with col3:
            top_growth = df_summary['Growth_Pct'].max()
            st.metric("Highest Growth", f"{top_growth:.2f}%")
        
        # Growth comparison chart
        growth_chart = create_growth_comparison_chart(df_summary)
        if growth_chart:
            st.plotly_chart(growth_chart, use_container_width=True)
        
        # Top and bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 5 Performers")
            top_performers = df_summary.nlargest(5, 'Growth_Pct')[['BDC', 'Growth_Pct']]
            st.dataframe(top_performers.style.format({'Growth_Pct': '{:.2f}%'}))
        
        with col2:
            st.subheader("📉 Bottom 5 Performers")
            bottom_performers = df_summary.nsmallest(5, 'Growth_Pct')[['BDC', 'Growth_Pct']]
            st.dataframe(bottom_performers.style.format({'Growth_Pct': '{:.2f}%'}))
    
    # ---------------------------
    # Portfolio Concentration Analysis with Insights
    # ---------------------------
    if df_extended is not None and df_forecast is not None and 'BDC' in df_forecast.columns:
        st.markdown("---")
        st.subheader("📈 Portfolio Concentration (Forecasted Companies Only)")

        # Filter extended data to only those companies that have forecast data
        forecasted_companies = df_forecast['BDC'].unique()
        df_extended_filtered = df_extended[df_extended['name'].isin(forecasted_companies)]

        latest_data = get_latest_company_data(df_extended_filtered)
        if latest_data is not None and not latest_data.empty:
            # Calculate HHI, market shares, and top concentrations
            portfolio_values = latest_data['value']
            total_value = portfolio_values.sum()
            market_shares = (portfolio_values / total_value) * 100
            hhi = (market_shares ** 2).sum()
            top5_conc = market_shares.nlargest(5).sum()
            num_companies = len(portfolio_values)

            # Top contributor details
            top_company = latest_data.loc[latest_data['value'].idxmax(), 'name']
            top_company_val = latest_data['value'].max()
            top_company_pct = (top_company_val / total_value) * 100

            # Concentration chart (Top 10 forecasted companies)
            top_10 = latest_data.nlargest(10, 'value')
            fig = px.bar(
                top_10,
                x='name',
                y='value',
                title="Top 10 Forecasted BDCs by Latest NII",
                color='value',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics as tiles
            col1, col2, col3 = st.columns(3)
            col1.metric("HHI Concentration", f"{hhi:.0f}")
            col2.metric("Top 5 Concentration", f"{top5_conc:.1f}%")
            col3.metric("Companies in Portfolio", num_companies)

            # Determine risk level and description
            if hhi < 1500:
                risk_level = "Low"
                risk_color = "#19A34A"
                risk_comment = "The portfolio is well diversified with minimal concentration risk."
            elif hhi <= 2500:
                risk_level = "Moderate"
                risk_color = "#FFA500"
                risk_comment = "The portfolio has moderate concentration; a few companies contribute significantly to NII."
            else:
                risk_level = "High"
                risk_color = "#E64B4B"
                risk_comment = "The portfolio is highly concentrated, dominated by a few companies; risk exposure is elevated."

            # Dynamic insight text
            insight_text = f"""
            <div style="
                margin-top:12px;
                padding:14px 18px;
                border-radius:12px;
                background:#f8f9fa;
                border-left:5px solid {risk_color};
                font-size:15px;
                color:#333;
            ">
                <b>💡 Portfolio Insight:</b><br>
                The portfolio currently includes <b>{num_companies}</b> forecasted BDCs. <br>
                The HHI concentration is <b>{hhi:.0f}</b>, indicating a <b>{risk_level} concentration</b>. <br>
                The top 5 forecasted BDCs contribute <b>{top5_conc:.1f}%</b> of total NII, with the highest contributor being <b>{top_company}</b> at <b>{top_company_pct:.1f}%</b> of the portfolio.<br><br>
                <b>Interpretation:</b> {risk_comment} Monitoring top contributors is recommended, as fluctuations in these BDCs will impact overall portfolio stability.
            </div>
            """
            st.markdown(insight_text, unsafe_allow_html=True)


# --------------------------------
# Page: Project Conclusion
# --------------------------------
elif page == "Conclusion":
    st.header("📋 Conclusion")
    
    st.markdown("""
    ## 🎯 Project Summary
    This BDC Net Investment Income (NII) Analytics Platform provides comprehensive financial analysis and 
    forecasting capabilities for Business Development Companies. The dashboard enables portfolio managers, 
    financial analysts, and investors to monitor historical performance, identify trends, and make data-driven 
    decisions using machine learning-powered forecasts.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Visualization Components & Use Cases")
        
        with st.expander("🏠 **Overview Dashboard**", expanded=True):
            st.markdown("""
            - **Portfolio KPIs**: Quick summary metrics showing total companies, NII values, and growth rates for at-a-glance portfolio health assessment
            - **Performance Small Multiples**: Grid of line charts displaying NII trends for top 6 BDCs, enabling rapid comparison across major holdings
            - **Portfolio Composition Pie Chart**: Visualizes allocation concentration among top 10 BDCs to identify overexposure risks
            - **Portfolio Forecast Timeline**: Shows cumulative NII projections across all forecasted companies for strategic planning
            """)
        
        with st.expander("🔍 **Portfolio Analytics**"):
            st.markdown("""
            - **Correlation Heatmap**: Identifies relationship patterns between BDCs using color-coded matrix to detect diversification opportunities
            - **Multi-Company Comparison**: Side-by-side trend analysis with historical data and forecast points for selected companies
            - **Growth Comparison Bars**: Direct visual comparison of historical vs forecasted growth rates across multiple BDCs
            - **Contribution Analysis**: Pie charts showing relative weight of selected companies in the portfolio
            """)
        
        with st.expander("🏢 **Company Analysis & Forecasting**"):
            st.markdown("""
            - **Historical vs Forecast Timeline**: Continuous visualization blending actual performance with ML predictions and confidence intervals
            - **Glass UI Metric Tiles**: Modern dashboard elements displaying key performance indicators with visual hierarchy
            - **Forecast Rank & Percentile**: Competitive positioning analysis showing company performance relative to peers
            - **Growth Trend Analysis**: Comparative metrics highlighting outperformance or underperformance vs market averages
            """)
        
        with st.expander("💡 **Insights & Risk Analytics**"):
            st.markdown("""
            - **Growth Distribution Chart**: Horizontal bar chart ranking BDCs by forecasted growth for quick winner/loser identification
            - **Portfolio Concentration Metrics**: HHI and top-5 concentration calculations with risk-level color coding
            - **Performance Leaderboards**: Data tables highlighting top and bottom performers for focused attention
            - **Risk Insight Cards**: Contextual explanations of concentration metrics with practical implications
            """)
    
    with col2:
        st.subheader("🚀 Key Analytical Capabilities")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
        <h4 style="color: white; margin-top: 0;">Advanced Features</h4>
        <ul style="color: white;">
        <li><b>Multi-Timeframe Analysis</b>: Historical + Forecast integration</li>
        <li><b>Comparative Analytics</b>: Company & market benchmarking</li>
        <li><b>Risk Assessment</b>: Diversification planning tools</li>
        <li><b>Trend Identification</b>: Pattern recognition capabilities</li>
        <li><b>Decision Support</b>: Visual evidence for strategy</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("🎯 Target Users")
        st.markdown("""
        - 📈 **Portfolio Managers**
        - 💼 **Financial Analysts** 
        - 🤝 **Investment Committees**
        - 📊 **Risk Management Teams**
        - 🎯 **BDC Research Professionals**
        """)
    
    st.markdown("---")
    
    st.subheader("🔮 Strategic Value Proposition")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f0f8ff; border-radius: 10px;">
        <h4 style="color: #1f77b4;">📈 Performance Monitoring</h4>
        <p style="color: #1f77b4;">Real-time tracking of NII trends and portfolio health across multiple time horizons</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f0fff0; border-radius: 10px;">
        <h4 style="color: #2ca02c;">🎯 Predictive Insights</h4>
        <p style="color: #2ca02c;">ML-powered forecasts with confidence intervals for proactive decision making</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #fff0f5; border-radius: 10px;">
        <h4 style="color: #d62728;">🛡️ Risk Management</h4>
        <p style="color: #d62728;">Comprehensive concentration analysis and correlation mapping for diversification</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ---
    
    **💡 Platform Impact**: This platform transforms complex financial data into actionable insights through 
    an intuitive, visually-driven interface designed for financial professionals requiring both granular 
    detail and big-picture portfolio perspectives.
    
    *Built with Streamlit, Plotly, and Machine Learning forecasting models for BDC financial analytics.*
    """)
# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("BDC Net Investment Income Analytics Platform | Professional Financial Analytics")