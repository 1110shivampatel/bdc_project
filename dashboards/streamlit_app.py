"""
BDC Analytics & Forecasting Platform - Streamlit Dashboard

A comprehensive dashboard for analyzing Business Development Companies (BDCs),
viewing portfolio compositions, comparing performance, and accessing forecasts.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboards.utils import (
    format_currency,
    format_percentage,
    get_bdc_data,
    get_bdc_forecast,
    get_bdc_list,
    get_latest_period,
    load_backtest_predictions,
    load_entity_reference,
    load_feature_panel,
    load_forecasts,
    load_model_comparison,
    load_quarterly_panel,
)

# Page configuration
st.set_page_config(
    page_title="BDC Analytics & Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling and readability
st.markdown(
    """
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    /* Ensure all text is readable */
    .stMarkdown, .stText, p, div, span, label {
        color: #1f2937 !important;
    }
    
    /* Metric cards - high contrast */
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 0.75rem !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox label, .stMultiselect label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Dropdown menu styling - high contrast */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    
    [data-baseweb="select"] input {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* Dropdown value display */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* Dropdown popup menu - force white background */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] > div > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Dropdown list container - force white */
    [data-baseweb="menu"],
    [data-baseweb="menu"] > ul,
    [data-baseweb="menu"] ul {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* Dropdown list items - force white background and dark text */
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] ul li,
    [data-baseweb="menu"] > ul > li,
    [data-baseweb="menu"] li[role="option"],
    [data-baseweb="menu"] li[role="menuitem"] {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* All text inside menu items - force dark */
    [data-baseweb="menu"] li *,
    [data-baseweb="menu"] ul li *,
    [data-baseweb="menu"] > ul > li * {
        color: #111827 !important;
    }
    
    /* Hover state - light gray background, dark text */
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] ul li:hover,
    [data-baseweb="menu"] > ul > li:hover,
    [data-baseweb="menu"] li[role="option"]:hover,
    [data-baseweb="menu"] li[role="menuitem"]:hover {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
    }
    
    [data-baseweb="menu"] li:hover *,
    [data-baseweb="menu"] ul li:hover *,
    [data-baseweb="menu"] > ul > li:hover * {
        color: #111827 !important;
    }
    
    /* Selected state - blue background, dark blue text */
    [data-baseweb="menu"] li[aria-selected="true"],
    [data-baseweb="menu"] ul li[aria-selected="true"],
    [data-baseweb="menu"] > ul > li[aria-selected="true"],
    [data-baseweb="menu"] li[aria-selected="true"] * {
        background-color: #eff6ff !important;
        color: #1e40af !important;
    }
    
    /* Dropdown option text - force dark */
    [data-baseweb="menu"] li span,
    [data-baseweb="menu"] ul li span,
    [data-baseweb="menu"] > ul > li span,
    [data-baseweb="menu"] li div,
    [data-baseweb="menu"] ul li div,
    [data-baseweb="menu"] > ul > li div {
        color: #111827 !important;
    }
    
    /* Multiselect styling */
    [data-baseweb="select"] [role="combobox"] {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* Multiselect tags/chips */
    [data-baseweb="tag"],
    [data-baseweb="tag"] span {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
    
    /* Ensure dropdown arrow is visible */
    [data-baseweb="select"] svg {
        color: #6b7280 !important;
    }
    
    /* Streamlit top bar styling */
    header[data-testid="stHeader"] {
        background-color: #1f2937 !important;
        border-bottom: 1px solid #374151 !important;
    }
    
    header[data-testid="stHeader"] .stAppToolbar {
        background-color: #1f2937 !important;
    }
    
    /* Top bar text and buttons */
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    header[data-testid="stHeader"] span,
    header[data-testid="stHeader"] p,
    header[data-testid="stHeader"] div {
        color: #ffffff !important;
    }
    
    /* Main menu button */
    [data-testid="stMainMenuButton"] button {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMainMenuButton"] button:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Deploy button */
    [data-testid="stDeployButton"] button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stDeployButton"] button:hover {
        background-color: #2563eb !important;
    }
    
    /* Top bar menu items */
    [data-baseweb="menu"] [role="menuitem"] {
        color: #111827 !important;
    }
    
    /* Additional dropdown styling for better visibility */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    /* Ensure text in select input is dark */
    [data-baseweb="select"] input::placeholder {
        color: #9ca3af !important;
    }
    
    /* Override any light text in dropdown popovers - comprehensive */
    [data-baseweb="popover"],
    [data-baseweb="popover"] * {
        color: #111827 !important;
    }
    
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] li > div,
    [data-baseweb="popover"] li > span,
    [data-baseweb="popover"] li * {
        color: #111827 !important;
        background-color: #ffffff !important;
    }
    
    /* Ensure selected items in dropdown are visible */
    [data-baseweb="popover"] li[aria-selected="true"],
    [data-baseweb="popover"] li[aria-selected="true"] > div,
    [data-baseweb="popover"] li[aria-selected="true"] > span,
    [data-baseweb="popover"] li[aria-selected="true"] * {
        background-color: #eff6ff !important;
        color: #1e40af !important;
        font-weight: 600 !important;
    }
    
    /* Hover states in popover */
    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] li:hover * {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
    }
    
    /* Additional specificity for nested elements */
    div[data-baseweb="popover"] ul li,
    div[data-baseweb="popover"] ul li * {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* Force all dropdown menu elements to have proper contrast */
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] [data-baseweb="menu"] ul,
    [data-baseweb="popover"] [data-baseweb="menu"] li,
    [data-baseweb="popover"] [data-baseweb="menu"] li > * {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* Override any inline styles or computed styles */
    [data-baseweb="popover"] [style*="background"],
    [data-baseweb="popover"] li[style*="background"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] [style*="color"],
    [data-baseweb="popover"] li[style*="color"] {
        color: #111827 !important;
    }
    
    /* Target specific Streamlit selectbox dropdown */
    .stSelectbox [data-baseweb="popover"],
    .stSelectbox [data-baseweb="popover"] *,
    .stMultiselect [data-baseweb="popover"],
    .stMultiselect [data-baseweb="popover"] * {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* Ensure list items in Streamlit dropdowns are visible */
    .stSelectbox [data-baseweb="popover"] li,
    .stSelectbox [data-baseweb="popover"] li *,
    .stMultiselect [data-baseweb="popover"] li,
    .stMultiselect [data-baseweb="popover"] li * {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    
    /* Hover states for Streamlit dropdowns */
    .stSelectbox [data-baseweb="popover"] li:hover,
    .stSelectbox [data-baseweb="popover"] li:hover *,
    .stMultiselect [data-baseweb="popover"] li:hover,
    .stMultiselect [data-baseweb="popover"] li:hover * {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    /* Info boxes */
    [data-baseweb="notification"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #ffffff !important;
    }
    
    .dataframe th {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    .dataframe td {
        color: #374151 !important;
    }
    
    /* Warning/Info boxes */
    .stAlert {
        background-color: #ffffff !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    /* Main container background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Better spacing */
    .element-container {
        margin-bottom: 1.5rem;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
    
    /* Table styling */
    table {
        background-color: #ffffff !important;
    }
    
    /* Ensure all Streamlit text elements are readable */
    .stApp {
        background-color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_overview_tab():
    """Render the Overview tab with BDC selector and key metrics."""
    st.header("📈 BDC Overview & Forecasts")
    
    # Load data
    feature_panel = load_feature_panel()
    forecasts = load_forecasts()
    entity_ref = load_entity_reference()
    
    if feature_panel.empty:
        st.error("No data available. Please run the data pipeline first.")
        return
    
    # BDC Selector
    bdc_list = get_bdc_list(feature_panel)
    if not bdc_list:
        st.error("No BDCs found in data.")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_bdc = st.selectbox(
            "Select a BDC:",
            options=bdc_list,
            index=0,
            help="Choose a Business Development Company to analyze",
        )
    
    with col2:
        latest_period = get_latest_period(feature_panel)
        st.info(f"**Latest Period:** {latest_period}")
    
    # Get BDC data
    bdc_data = get_bdc_data(feature_panel, selected_bdc)
    if bdc_data.empty:
        st.warning(f"No data available for {selected_bdc}")
        return
    
    latest_data = bdc_data[bdc_data["period"] == latest_period]
    if latest_data.empty:
        latest_data = bdc_data.iloc[-1:].copy()
    
    # Key Metrics Row
    st.subheader("Key Metrics")
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        total_assets = latest_data["total_assets"].iloc[0] if "total_assets" in latest_data.columns else None
        st.metric(
            "Total Assets",
            format_currency(total_assets) if total_assets and pd.notna(total_assets) else "N/A",
        )
    
    with metric_cols[1]:
        net_assets = latest_data["net_assets"].iloc[0] if "net_assets" in latest_data.columns else None
        st.metric(
            "Net Assets",
            format_currency(net_assets) if net_assets and pd.notna(net_assets) else "N/A",
        )
    
    with metric_cols[2]:
        nii = latest_data["net_investment_income"].iloc[0] if "net_investment_income" in latest_data.columns else None
        st.metric(
            "Net Investment Income",
            format_currency(nii) if nii and pd.notna(nii) else "N/A",
        )
    
    with metric_cols[3]:
        forecast = get_bdc_forecast(forecasts, selected_bdc)
        forecast_label = "Next Qtr Forecast"
        if forecast is not None:
            forecast_color = "normal" if forecast >= 0 else "inverse"
            st.metric(
                forecast_label,
                format_percentage(forecast),
                delta=f"{forecast:+.2f}%" if forecast is not None else None,
                delta_color=forecast_color,
            )
        else:
            st.metric(forecast_label, "N/A")
    
    # Historical Trends
    st.subheader("Historical Trends")
    
    trend_cols = st.columns(2)
    
    with trend_cols[0]:
        # Total Assets Trend
        if "total_assets" in bdc_data.columns:
            fig_assets = px.line(
                bdc_data,
                x="period",
                y="total_assets",
                title="Total Assets Over Time",
                labels={"total_assets": "Total Assets ($)", "period": "Period"},
                markers=True,
                color_discrete_sequence=["#2563eb"],
            )
            fig_assets.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=350,
                font=dict(color="#111827", size=12),
                title_font=dict(size=16, color="#111827", family="Arial, sans-serif"),
                xaxis=dict(
                    title_font=dict(color="#374151", size=12),
                    tickfont=dict(color="#6b7280", size=11),
                    gridcolor="#e5e7eb",
                ),
                yaxis=dict(
                    title_font=dict(color="#374151", size=12),
                    tickfont=dict(color="#6b7280", size=11),
                    gridcolor="#e5e7eb",
                ),
            )
            fig_assets.update_traces(
                line=dict(width=3),
                marker=dict(size=8, color="#2563eb"),
            )
            st.plotly_chart(fig_assets, use_container_width=True)
    
    with trend_cols[1]:
        # Net Investment Income Trend
        if "net_investment_income" in bdc_data.columns:
            fig_nii = px.line(
                bdc_data,
                x="period",
                y="net_investment_income",
                title="Net Investment Income Over Time",
                labels={"net_investment_income": "NII ($)", "period": "Period"},
                markers=True,
                color_discrete_sequence=["#10b981"],
            )
            fig_nii.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=350,
                font=dict(color="#111827", size=12),
                title_font=dict(size=16, color="#111827", family="Arial, sans-serif"),
                xaxis=dict(
                    title_font=dict(color="#374151", size=12),
                    tickfont=dict(color="#6b7280", size=11),
                    gridcolor="#e5e7eb",
                ),
                yaxis=dict(
                    title_font=dict(color="#374151", size=12),
                    tickfont=dict(color="#6b7280", size=11),
                    gridcolor="#e5e7eb",
                ),
            )
            fig_nii.update_traces(
                line=dict(width=3),
                marker=dict(size=8, color="#10b981"),
            )
            st.plotly_chart(fig_nii, use_container_width=True)
    
    # Leverage and Efficiency Metrics
    st.subheader("Leverage & Efficiency")
    
    eff_cols = st.columns(3)
    
    with eff_cols[0]:
        if "debt_to_assets" in latest_data.columns:
            debt_ratio = latest_data["debt_to_assets"].iloc[0]
            if pd.notna(debt_ratio):
                st.metric("Debt to Assets", format_percentage(debt_ratio * 100))
            else:
                st.metric("Debt to Assets", "N/A")
    
    with eff_cols[1]:
        if "nii_to_assets" in latest_data.columns:
            nii_ratio = latest_data["nii_to_assets"].iloc[0]
            if pd.notna(nii_ratio):
                st.metric("NII to Assets", format_percentage(nii_ratio * 100))
            else:
                st.metric("NII to Assets", "N/A")
    
    with eff_cols[2]:
        if "debt_to_equity" in latest_data.columns:
            debt_equity = latest_data["debt_to_equity"].iloc[0]
            if pd.notna(debt_equity):
                st.metric("Debt to Equity", format_percentage(debt_equity * 100))
            else:
                st.metric("Debt to Equity", "N/A")
    
    # Forecast Details
    if forecast is not None:
        st.subheader("Forecast Details")
        forecast_cols = st.columns([2, 1])
        
        with forecast_cols[0]:
            st.info(
                f"**Forecast for Next Quarter:** {format_percentage(forecast)}\n\n"
                f"This represents the predicted percentage change in Net Assets for the next quarter, "
                f"based on historical patterns and current portfolio characteristics."
            )
        
        with forecast_cols[1]:
            if net_assets and pd.notna(net_assets) and forecast:
                projected_na = net_assets * (1 + forecast / 100)
                st.metric(
                    "Projected Net Assets",
                    format_currency(projected_na),
                    delta=format_currency(projected_na - net_assets),
                )


def render_portfolio_tab():
    """Render the Portfolio tab with SOI breakdowns."""
    st.header("💼 Portfolio Composition")
    
    feature_panel = load_feature_panel()
    forecasts = load_forecasts()
    
    if feature_panel.empty:
        st.error("No data available.")
        return
    
    bdc_list = get_bdc_list(feature_panel)
    if not bdc_list:
        st.error("No BDCs found.")
        return
    
    selected_bdc = st.selectbox("Select a BDC:", options=bdc_list, key="portfolio_bdc")
    
    bdc_data = get_bdc_data(feature_panel, selected_bdc)
    latest_period = get_latest_period(feature_panel)
    latest_data = bdc_data[bdc_data["period"] == latest_period]
    
    if latest_data.empty:
        st.warning(f"No data for {selected_bdc} in latest period.")
        return
    
    latest_data = latest_data.iloc[0]
    
    # Portfolio Features
    st.subheader("Portfolio Characteristics")
    
    portfolio_cols = st.columns(2)
    
    with portfolio_cols[0]:
        # Fair Value vs Cost
        if "portfolio_fv" in latest_data.index and "portfolio_cost" in latest_data.index:
            fv = latest_data["portfolio_fv"]
            cost = latest_data["portfolio_cost"]
            
            if pd.notna(fv) and pd.notna(cost):
                fig_fv = go.Figure()
                fig_fv.add_trace(go.Bar(
                    x=["Fair Value", "Cost"],
                    y=[fv, cost],
                    marker_color=["#2563eb", "#f59e0b"],
                    text=[format_currency(fv), format_currency(cost)],
                    textposition="auto",
                    textfont=dict(color="#111827", size=12),
                ))
                fig_fv.update_layout(
                    title="Portfolio Fair Value vs Cost",
                    yaxis_title="Value ($)",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=350,
                    font=dict(color="#111827", size=12),
                    title_font=dict(size=16, color="#111827", family="Arial, sans-serif"),
                    xaxis=dict(
                        title_font=dict(color="#374151", size=12),
                        tickfont=dict(color="#6b7280", size=11),
                        gridcolor="#e5e7eb",
                    ),
                    yaxis=dict(
                        title_font=dict(color="#374151", size=12),
                        tickfont=dict(color="#6b7280", size=11),
                        gridcolor="#e5e7eb",
                    ),
                )
                st.plotly_chart(fig_fv, use_container_width=True)
        
        # Concentration Metrics
        if "top10_concentration_pct" in latest_data.index:
            concentration = latest_data["top10_concentration_pct"]
            if pd.notna(concentration):
                st.metric("Top 10 Concentration", format_percentage(concentration))
    
    with portfolio_cols[1]:
        # Floating vs Fixed Rate
        if "floating_rate_pct" in latest_data.index and "fixed_rate_pct" in latest_data.index:
            floating = latest_data["floating_rate_pct"]
            fixed = latest_data["fixed_rate_pct"]
            
            if pd.notna(floating) or pd.notna(fixed):
                fig_rates = go.Figure(data=[
                    go.Pie(
                        labels=["Floating Rate", "Fixed Rate"],
                        values=[floating if pd.notna(floating) else 0, fixed if pd.notna(fixed) else 0],
                        hole=0.4,
                        marker_colors=["#10b981", "#ef4444"],
                        textfont=dict(color="#111827", size=12),
                    )
                ])
                fig_rates.update_layout(
                    title="Rate Type Distribution",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=350,
                    font=dict(color="#111827", size=12),
                    title_font=dict(size=16, color="#111827", family="Arial, sans-serif"),
                    legend=dict(
                        font=dict(color="#374151", size=11),
                    ),
                )
                st.plotly_chart(fig_rates, use_container_width=True)
        
        # HHI
        if "portfolio_hhi" in latest_data.index:
            hhi = latest_data["portfolio_hhi"]
            if pd.notna(hhi):
                st.metric("Portfolio HHI", f"{hhi:.0f}")
    
    # Sector Exposure
    if "top_sector_pct" in latest_data.index:
        top_sector = latest_data["top_sector_pct"]
        if pd.notna(top_sector):
            st.subheader("Sector Exposure")
            st.metric("Top Sector Weight", format_percentage(top_sector))
    
    # Fair Value Gap
    if "fv_cost_gap_pct" in latest_data.index:
        gap = latest_data["fv_cost_gap_pct"]
        if pd.notna(gap):
            st.subheader("Valuation Gap")
            gap_color = "normal" if gap >= 0 else "inverse"
            st.metric(
                "Fair Value - Cost Gap",
                format_percentage(gap),
                delta=f"{gap:+.2f}%",
                delta_color=gap_color,
            )


def render_compare_tab():
    """Render the Compare tab for side-by-side BDC comparison."""
    st.header("⚖️ Compare BDCs")
    
    feature_panel = load_feature_panel()
    forecasts = load_forecasts()
    
    if feature_panel.empty:
        st.error("No data available.")
        return
    
    bdc_list = get_bdc_list(feature_panel)
    if not bdc_list:
        st.error("No BDCs found.")
        return
    
    # Multi-select for comparison
    selected_bdcs = st.multiselect(
        "Select BDCs to compare (2-3 recommended):",
        options=bdc_list,
        default=bdc_list[:2] if len(bdc_list) >= 2 else bdc_list[:1],
        max_selections=5,
    )
    
    if not selected_bdcs:
        st.info("Please select at least one BDC to compare.")
        return
    
    latest_period = get_latest_period(feature_panel)
    
    # Comparison Table
    st.subheader(f"Key Metrics Comparison ({latest_period})")
    
    comparison_data = []
    for bdc in selected_bdcs:
        bdc_data = get_bdc_data(feature_panel, bdc)
        latest = bdc_data[bdc_data["period"] == latest_period]
        if latest.empty:
            continue
        
        latest = latest.iloc[0]
        forecast = get_bdc_forecast(forecasts, bdc)
        
        row = {
            "BDC": bdc,
            "Total Assets": format_currency(latest.get("total_assets")) if pd.notna(latest.get("total_assets")) else "N/A",
            "Net Assets": format_currency(latest.get("net_assets")) if pd.notna(latest.get("net_assets")) else "N/A",
            "NII": format_currency(latest.get("net_investment_income")) if pd.notna(latest.get("net_investment_income")) else "N/A",
            "Debt/Assets": format_percentage(latest.get("debt_to_assets", 0) * 100) if pd.notna(latest.get("debt_to_assets")) else "N/A",
            "Forecast": format_percentage(forecast) if forecast is not None else "N/A",
        }
        comparison_data.append(row)
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Comparison Charts
    if len(selected_bdcs) >= 2:
        st.subheader("Historical Comparison")
        
        # Prepare data for comparison
        compare_data = []
        for bdc in selected_bdcs:
            bdc_data = get_bdc_data(feature_panel, bdc)
            if not bdc_data.empty:
                compare_data.append(bdc_data)
        
        if compare_data:
            combined = pd.concat(compare_data, ignore_index=True)
            
            chart_cols = st.columns(2)
            
            with chart_cols[0]:
                if "total_assets" in combined.columns:
                    fig = px.line(
                        combined,
                        x="period",
                        y="total_assets",
                        color="name",
                        title="Total Assets Comparison",
                        labels={"total_assets": "Total Assets ($)", "period": "Period"},
                        markers=True,
                    )
                    fig.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        height=400,
                        font=dict(color="#111827", size=12),
                        title_font=dict(size=16, color="#111827", family="Arial, sans-serif"),
                        xaxis=dict(
                            title_font=dict(color="#374151", size=12),
                            tickfont=dict(color="#6b7280", size=11),
                            gridcolor="#e5e7eb",
                        ),
                        yaxis=dict(
                            title_font=dict(color="#374151", size=12),
                            tickfont=dict(color="#6b7280", size=11),
                            gridcolor="#e5e7eb",
                        ),
                        legend=dict(
                            font=dict(color="#374151", size=11),
                            bgcolor="white",
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with chart_cols[1]:
                if "net_investment_income" in combined.columns:
                    fig = px.line(
                        combined,
                        x="period",
                        y="net_investment_income",
                        color="name",
                        title="Net Investment Income Comparison",
                        labels={"net_investment_income": "NII ($)", "period": "Period"},
                        markers=True,
                    )
                    fig.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        height=400,
                        font=dict(color="#111827", size=12),
                        title_font=dict(size=16, color="#111827", family="Arial, sans-serif"),
                        xaxis=dict(
                            title_font=dict(color="#374151", size=12),
                            tickfont=dict(color="#6b7280", size=11),
                            gridcolor="#e5e7eb",
                        ),
                        yaxis=dict(
                            title_font=dict(color="#374151", size=12),
                            tickfont=dict(color="#6b7280", size=11),
                            gridcolor="#e5e7eb",
                        ),
                        legend=dict(
                            font=dict(color="#374151", size=11),
                            bgcolor="white",
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)


def render_models_tab():
    """Render the Model Performance tab."""
    st.header("🤖 Model Performance")
    
    model_comparison = load_model_comparison()
    backtest_preds = load_backtest_predictions()
    
    if model_comparison.empty:
        st.warning("No model comparison data available. Please train models first.")
        return
    
    st.subheader("Model Comparison")
    
    # Sort by MAE
    model_comparison_sorted = model_comparison.sort_values("mae")
    
    # Display metrics
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        best_model = model_comparison_sorted.index[0]
        st.metric("Best Model", best_model)
    
    with metric_cols[1]:
        best_mae = model_comparison_sorted["mae"].iloc[0]
        st.metric("Best MAE", f"{best_mae:.2f}")
    
    with metric_cols[2]:
        best_rmse = model_comparison_sorted["rmse"].iloc[0]
        st.metric("Best RMSE", f"{best_rmse:.2f}")
    
    with metric_cols[3]:
        best_r2 = model_comparison_sorted["r2"].iloc[0]
        st.metric("Best R²", f"{best_r2:.3f}")
    
    # Model Comparison Table
    st.subheader("All Models Performance")
    
    # Format the comparison table
    display_comparison = model_comparison_sorted.copy()
    display_comparison["mae"] = display_comparison["mae"].apply(lambda x: f"{x:.2f}")
    display_comparison["rmse"] = display_comparison["rmse"].apply(lambda x: f"{x:.2f}")
    display_comparison["mape"] = display_comparison["mape"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_comparison["r2"] = display_comparison["r2"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_comparison, use_container_width=True)
    
    # Performance Visualization
    st.subheader("Model Performance Visualization")
    
    viz_cols = st.columns(2)
    
    with viz_cols[0]:
        # MAE Comparison
        fig_mae = px.bar(
            model_comparison_sorted.head(10),
            x=model_comparison_sorted.head(10).index,
            y="mae",
            title="MAE by Model (Top 10)",
            labels={"mae": "Mean Absolute Error", "index": "Model"},
            color_discrete_sequence=["#2563eb"],
        )
        fig_mae.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400,
            xaxis_tickangle=-45,
            font=dict(color="#111827", size=12),
            title_font=dict(size=16, color="#111827", family="Arial, sans-serif"),
            xaxis=dict(
                title_font=dict(color="#374151", size=12),
                tickfont=dict(color="#6b7280", size=11),
                gridcolor="#e5e7eb",
            ),
            yaxis=dict(
                title_font=dict(color="#374151", size=12),
                tickfont=dict(color="#6b7280", size=11),
                gridcolor="#e5e7eb",
            ),
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with viz_cols[1]:
        # R² Comparison
        fig_r2 = px.bar(
            model_comparison_sorted.head(10),
            x=model_comparison_sorted.head(10).index,
            y="r2",
            title="R² by Model (Top 10)",
            labels={"r2": "R² Score", "index": "Model"},
            color="r2",
            color_continuous_scale="Viridis",
        )
        fig_r2.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400,
            xaxis_tickangle=-45,
            font=dict(color="#111827", size=12),
            title_font=dict(size=16, color="#111827", family="Arial, sans-serif"),
            xaxis=dict(
                title_font=dict(color="#374151", size=12),
                tickfont=dict(color="#6b7280", size=11),
                gridcolor="#e5e7eb",
            ),
            yaxis=dict(
                title_font=dict(color="#374151", size=12),
                tickfont=dict(color="#6b7280", size=11),
                gridcolor="#e5e7eb",
            ),
            coloraxis_colorbar=dict(
                title_font=dict(color="#374151", size=11),
                tickfont=dict(color="#6b7280", size=10),
            ),
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Backtest Results
    if not backtest_preds.empty:
        st.subheader("Backtest Results")
        
        # Aggregate predictions by model
        if "model_name" in backtest_preds.columns:
            model_performance = backtest_preds.groupby("model_name").agg({
                "y_true": ["mean", "std"],
                "y_pred": ["mean", "std"],
            }).round(2)
            
            st.write("Prediction Statistics by Model:")
            st.dataframe(model_performance, use_container_width=True)


def main():
    """Main dashboard application."""
    # Header with better styling
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
            <h1 style="color: #1e3a8a; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                📊 BDC Analytics & Forecasting Platform
            </h1>
            <p style="color: #6b7280; font-size: 1.1rem; margin: 0;">
                Comprehensive analysis and forecasting for Business Development Companies
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Disclaimer with better contrast
    st.markdown(
        """
        <div style="background-color: #fef3c7; padding: 1.25rem; border-radius: 0.75rem; 
                   border-left: 5px solid #f59e0b; margin-bottom: 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="color: #92400e; font-weight: 600; margin: 0 0 0.5rem 0; font-size: 1rem;">
                ⚠️ <strong>Disclaimer:</strong>
            </p>
            <p style="color: #78350f; margin: 0; font-size: 0.95rem; line-height: 1.6;">
                This dashboard is for analytical and educational purposes only. 
                It does not constitute investment advice. All forecasts are based on historical data and statistical models, 
                which may not accurately predict future performance. Please consult with a qualified financial advisor 
                before making any investment decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💼 Portfolio", "⚖️ Compare", "🤖 Model Performance"])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_portfolio_tab()
    
    with tab3:
        render_compare_tab()
    
    with tab4:
        render_models_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "BDC Analytics & Forecasting Platform | Data Source: SEC BDC Data Sets"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

