"""
Interactive Streamlit Dashboard for Prophet Supermarket Sales Forecasting
Enhanced with Sales Analytics Dashboard and AI Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import warnings
import google.generativeai as genai
import base64
warnings.filterwarnings('ignore')

# Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyCaHr_JDJeympltKqvVax4gT19sNYflWV0"
genai.configure(api_key=GEMINI_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="� ProfitLens - Sales Analytics Platform",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Force light theme - Override dark mode */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Force light theme for all major containers */
    .main .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Force light sidebar */
    .css-1d391kg, .css-1aumxhk, section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        color: #000000 !important;
    }
    
    /* Force light mode for text elements */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, div {
        color: #000000 !important;
    }
    
    /* Force light mode for inputs and widgets */
    .stSelectbox, .stSlider, .stNumberInput, .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Force light mode for dataframes and tables */
    .stDataFrame, .element-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Global font and spacing improvements */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header container styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
   .logo-container {
       margin-right: 1.5rem;  /* MODIFIED: Was 2rem, reduced the gap */
       display: flex;
       align-items: center;
       justify-content: center;
   }
    
    .logo-img {
        width: 80px;
        height: 80px;
        object-fit: contain;
        filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.1));
        border-radius: 8px;
    }
    
    .brand-container {
        text-align: left;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Main dashboard styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        padding-top: 0 !important;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 0.9;
        letter-spacing: -0.02em;
        
    }
    
   .tagline {
       font-size: 1.1rem;
       color: #6b7280;
       margin: 0;
       font-style: italic;
       font-weight: 500;
       line-height: 1.1;
       opacity: 0.9;
   }
    
    /* Tab content styling */
    .tab-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.3rem;
    }
    
    .tab-description {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    
    /* Section styling */
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 2rem 0;
    }
    
    /* Chart insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
        font-style: italic;
        color: #0c4a6e;
    }
    
    /* AI recommendation styling */
    .recommendation-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.15);
    }
    
    .recommendation-content {
        color: #14532d;
        font-size: 16px;
        line-height: 1.6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        white-space: pre-line;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* AI recommendation card */
    .ai-card {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Filter section */
    .filter-section {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Improve sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Responsive design for header */
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column;
            text-align: center;
            padding: 1.5rem 1rem;
        }
        
        .logo-container {
            margin-right: 0;
            margin-bottom: 1rem;
        }
        
        .brand-container {
            text-align: center;
        }
        
        .main-title {
            font-size: 2rem;
        }
        
        .tagline {
            font-size: 1rem;
        }
    }
    
    /* Enhanced visual polish */
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #1e40af, #3b82f6);
        border-radius: 15px 15px 0 0;
    }
    
    .header-container {
        position: relative;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions for performance

def get_base64_image(image_path):
    """Convert image to base64 string for HTML embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.warning(f"Could not load logo: {e}")
        return ""

@st.cache_resource
def load_models():
    """Load all trained Prophet models from saved_model directory"""
    models = {}
    model_dir = "saved_model"

    if not os.path.exists(model_dir):
        st.error(f"Model directory '{model_dir}' not found!")
        return models

    # Load all model files
    for filename in os.listdir(model_dir):
        if filename.startswith("prophet_") and filename.endswith(".pkl"):
            category = filename.replace("prophet_", "").replace(".pkl", "")
            try:
                model_path = os.path.join(model_dir, filename)
                model = joblib.load(model_path)
                models[category] = model
            except Exception as e:
                st.warning(f"Failed to load model for {category}: {str(e)}")

    return models


@st.cache_data
def load_data():
    """Load historical sales data"""
    try:
        df = pd.read_csv("dummy_forecasting_data_100weeks.csv")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None


def prepare_prophet_data(category_data):
    """Prepare data for Prophet modeling"""
    category_data = category_data.sort_values('week').reset_index(drop=True)

    # Create datetime column starting from 2020-01-01
    start_date = datetime(2020, 1, 1)
    category_data['ds'] = pd.to_datetime(
        [start_date + timedelta(weeks=w - 1) for w in category_data['week']])

    # Create target variable
    category_data['y'] = category_data['weekly_sales']

    # Calculate discount rate
    category_data['discount_rate'] = (
        category_data['normal_price'] - category_data['promo_price']) / category_data['normal_price']

    # Handle missing values
    category_data = category_data.fillna(0)

    return category_data[['ds', 'y', 'promo_active',
                          'promo_price', 'normal_price', 'discount_rate']]


def generate_forecast(
        model,
        historical_data,
        weekly_promo_prices,
        forecast_weeks=16):
    """Generate forecast with custom weekly promo_prices"""
    # Get the last date from historical data
    last_date = historical_data['ds'].max()

    # Create future dates
    future_dates = pd.date_range(
        start=last_date +
        timedelta(
            weeks=1),
        periods=forecast_weeks,
        freq='W')

    # Get last known values
    last_row = historical_data.iloc[-1]
    normal_price = last_row['normal_price']

    # Create lists for weekly regressor values
    promo_active_list = []
    promo_price_list = []
    discount_rate_list = []

    # Calculate values for each week
    for week in range(forecast_weeks):
        promo_price = weekly_promo_prices[week]
        discount_rate = max(0, (normal_price - promo_price) / normal_price)
        promo_active = 1 if promo_price < normal_price else 0

        promo_active_list.append(promo_active)
        promo_price_list.append(promo_price)
        discount_rate_list.append(discount_rate)

    # Create future dataframe with weekly variations
    future_df = pd.DataFrame({
        'ds': future_dates,
        'promo_active': promo_active_list,
        'promo_price': promo_price_list,
        'normal_price': normal_price,
        'discount_rate': discount_rate_list
    })

    # Generate forecast
    forecast = model.predict(future_df)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Analytics Functions for Sales Dashboard


@st.cache_data
def calculate_revenue_analysis(df):
    """Calculate revenue-based metrics for profitability analysis"""
    # Calculate revenue for each record
    df_revenue = df.copy()
    df_revenue['revenue'] = df_revenue['weekly_sales'] * df_revenue.apply(
        lambda row: row['promo_price'] if row['promo_active'] == 1 else row['normal_price'],
        axis=1
    )

    # Revenue by category
    category_revenue = df_revenue.groupby('category').agg({
        'revenue': 'sum',
        'weekly_sales': 'sum',
        'promo_active': 'mean'
    }).sort_values('revenue', ascending=False)

    # Revenue during promo vs non-promo
    promo_revenue = df_revenue[df_revenue['promo_active']
                               == 1]['revenue'].mean()
    non_promo_revenue = df_revenue[df_revenue['promo_active']
                                   == 0]['revenue'].mean()

    # Weekly revenue trend
    weekly_revenue = df_revenue.groupby('week').agg({
        'revenue': 'sum',
        'weekly_sales': 'sum'
    }).reset_index()

    return category_revenue, promo_revenue, non_promo_revenue, weekly_revenue


@st.cache_data
def seasonal_analysis(df):
    """Analyze seasonal patterns in sales"""
    # Group by week number (assuming 52 weeks cycle)
    df_seasonal = df.copy()
    df_seasonal['season_week'] = df_seasonal['week'] % 52
    df_seasonal['season_week'] = df_seasonal['season_week'].replace(0, 52)

    seasonal_pattern = df_seasonal.groupby(
        'season_week')['weekly_sales'].mean().reset_index()

    # Identify peaks and valleys
    avg_sales = seasonal_pattern['weekly_sales'].mean()
    peaks = seasonal_pattern[seasonal_pattern['weekly_sales']
                             > avg_sales * 1.1]
    valleys = seasonal_pattern[seasonal_pattern['weekly_sales']
                               < avg_sales * 0.9]

    return seasonal_pattern, peaks, valleys, avg_sales


@st.cache_data
def category_performance_matrix(df):
    """Create performance matrix: sales volume vs promo effectiveness"""
    category_metrics = df.groupby('category').agg({
        'weekly_sales': ['sum', 'mean'],
        'promo_active': 'mean'
    }).round(2)

    category_metrics.columns = ['total_sales', 'avg_sales', 'promo_frequency']

    # Calculate promo effectiveness
    promo_effectiveness = {}
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        promo_sales = cat_data[cat_data['promo_active']
                               == 1]['weekly_sales'].mean()
        non_promo_sales = cat_data[cat_data['promo_active']
                                   == 0]['weekly_sales'].mean()

        if non_promo_sales > 0:
            effectiveness = (promo_sales - non_promo_sales) / \
                non_promo_sales * 100
        else:
            effectiveness = 0

        promo_effectiveness[category] = effectiveness

    category_metrics['promo_effectiveness'] = category_metrics.index.map(
        promo_effectiveness)
    return category_metrics.reset_index()


def pareto_analysis(df):
    """Generate Pareto analysis (80/20 rule) for categories"""
    # Calculate total sales by category
    category_sales = df.groupby(
        'category')['weekly_sales'].sum().sort_values(ascending=False)

    # Calculate cumulative percentage
    total_sales = category_sales.sum()
    cumulative_pct = (category_sales.cumsum() / total_sales * 100)

    # Create Pareto chart
    fig = go.Figure()

    # Bar chart for sales
    fig.add_trace(go.Bar(
        x=category_sales.index,
        y=category_sales.values,
        name='Total Sales',
        marker_color='#2E8B57',
        yaxis='y'
    ))

    # Line chart for cumulative percentage
    fig.add_trace(go.Scatter(
        x=category_sales.index,
        y=cumulative_pct.values,
        mode='lines+markers',
        name='Cumulative %',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        yaxis='y2'
    ))

    # Add 80% line
    fig.add_hline(y=80, line_dash="dash", line_color="red",
                  annotation_text="80% Line", yref='y2')

    fig.update_layout(
        title="Pareto Analysis: Category Sales Distribution (80/20 Rule)",
        xaxis_title="Product Categories",
        xaxis=dict(
            tickangle=45),
        yaxis=dict(
            title="Total Sales",
            side="left"),
        yaxis2=dict(
            title="Cumulative Percentage (%)",
            side="right",
            overlaying="y"),
        hovermode='x unified',
        height=500,
        showlegend=True)

    # Calculate insights
    categories_for_80pct = (cumulative_pct <= 80).sum()
    pct_categories_for_80pct = (
        categories_for_80pct / len(category_sales)) * 100

    return fig, categories_for_80pct, pct_categories_for_80pct, category_sales


def top_categories_chart(df, top_n=10):
    """Create bar chart for top categories by sales"""
    category_sales = df.groupby('category')['weekly_sales'].sum(
    ).sort_values(ascending=False).head(top_n)

    fig = go.Figure(data=[
        go.Bar(
            x=category_sales.values,
            y=category_sales.index,
            orientation='h',
            marker_color='#2E8B57',
            text=[f'{val:,.0f}' for val in category_sales.values],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=f"Top {top_n} Categories by Total Sales",
        xaxis_title="Total Sales",
        yaxis_title="Categories",
        height=400,
        showlegend=False
    )

    # Calculate insights
    total_sales = df['weekly_sales'].sum()
    top_contribution = category_sales.sum() / total_sales * 100

    return fig, top_contribution


def promo_impact_analysis(df):
    """Analyze promotional impact on sales"""
    # Separate promo vs non-promo data
    promo_sales = df[df['promo_active'] == 1]['weekly_sales']
    non_promo_sales = df[df['promo_active'] == 0]['weekly_sales']

    # Create box plot
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=non_promo_sales,
        name='No Promotion',
        marker_color='#FF6B6B',
        boxpoints='outliers'
    ))

    fig.add_trace(go.Box(
        y=promo_sales,
        name='With Promotion',
        marker_color='#4ECDC4',
        boxpoints='outliers'
    ))

    fig.update_layout(
        title="Promotional Impact on Weekly Sales",
        yaxis_title="Weekly Sales",
        xaxis_title="Promotion Status",
        height=400,
        showlegend=True
    )

    # Calculate statistics
    avg_promo = promo_sales.mean()
    avg_non_promo = non_promo_sales.mean()
    improvement = ((avg_promo - avg_non_promo) / avg_non_promo) * 100

    # Additional bar chart for averages
    fig_bar = go.Figure(data=[
        go.Bar(
            x=['No Promotion', 'With Promotion'],
            y=[avg_non_promo, avg_promo],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{avg_non_promo:.0f}', f'{avg_promo:.0f}'],
            textposition='outside'
        )
    ])

    fig_bar.update_layout(
        title="Average Sales: Promotion vs Non-Promotion",
        yaxis_title="Average Weekly Sales",
        height=300,
        showlegend=False
    )

    return fig, fig_bar, avg_promo, avg_non_promo, improvement


def time_series_trend(df):
    """Create aggregated time series trend"""
    # Aggregate sales by week
    weekly_trend = df.groupby('week')['weekly_sales'].sum().reset_index()

    # Create datetime for better visualization
    start_date = datetime(2020, 1, 1)
    weekly_trend['date'] = pd.to_datetime(
        [start_date + timedelta(weeks=w - 1) for w in weekly_trend['week']])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=weekly_trend['date'],
        y=weekly_trend['weekly_sales'],
        mode='lines+markers',
        name='Total Weekly Sales',
        line=dict(color='#2E8B57', width=2),
        marker=dict(size=4)
    ))

    # Add trend line
    z = np.polyfit(weekly_trend['week'], weekly_trend['weekly_sales'], 1)
    p = np.poly1d(z)

    fig.add_trace(go.Scatter(
        x=weekly_trend['date'],
        y=p(weekly_trend['week']),
        mode='lines',
        name='Trend Line',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title="Weekly Sales Trend (All Categories Combined)",
        xaxis_title="Date",
        yaxis_title="Total Weekly Sales",
        height=400,
        hovermode='x unified'
    )

    # Calculate growth
    first_half_avg = weekly_trend['weekly_sales'][:50].mean()
    second_half_avg = weekly_trend['weekly_sales'][50:].mean()
    growth_rate = ((second_half_avg - first_half_avg) / first_half_avg) * 100

    return fig, growth_rate


def create_forecasting_dashboard(models, data):
    """Create the refined forecasting dashboard content"""

    # === SIDEBAR CONTROLS ===
    with st.sidebar:
        st.markdown("### 🎛️ Forecasting Controls")
        st.markdown("---")

        # Category selection with better styling
        categories = sorted(list(models.keys()))
        default_index = categories.index(
            "Sirup") if "Sirup" in categories else 0

        selected_category = st.selectbox(
            "📦 **Product Category**",
            categories,
            index=default_index,
            help="Choose a product category for analysis"
        )

        # Store selection (mockup - doesn't affect functionality)
        store_options = [
            "All Stores",
            "Store A - Jakarta Central",
            "Store B - Jakarta South", 
            "Store C - Bandung",
            "Store D - Surabaya",
            "Store E - Medan"
        ]
        
        selected_store = st.selectbox(
            "🏪 **Store Location**",
            store_options,
            index=0,
            help="Select store location (mockup feature)"
        )
        
        # Display selected store info
        if selected_store != "All Stores":
            st.info(f"📍 Selected: {selected_store}")
        else:
            st.info("📍 Showing aggregated data from all stores")

        # Save selected category to session state for AI tab
        st.session_state.selected_category = selected_category

        st.write("")  # Spacing

        # Get data for selected category
        category_data = data[data['category'] == selected_category].copy()
        historical_data = prepare_prophet_data(category_data)

        # Get pricing information
        last_row = historical_data.iloc[-1]
        normal_price = last_row['normal_price']
        default_promo_price = last_row['promo_price']

        # Forecast horizon control
        forecast_weeks = st.slider(
            "� **Forecast Horizon (weeks)**",
            min_value=8,
            max_value=24,
            value=16,
            step=4,
            help="Number of weeks to forecast ahead"
        )

        st.write("")

        # Current pricing display
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Normal Price", f"${normal_price:,.0f}")
        with col2:
            st.metric("Current Promo", f"${default_promo_price:,.0f}")

    # === MAIN CONTENT AREA ===

    # Initialize session state for weekly prices
    if 'weekly_prices' not in st.session_state:
        st.session_state.weekly_prices = [
            int(default_promo_price)] * forecast_weeks

    # Adjust prices list if forecast weeks changed
    current_length = len(st.session_state.weekly_prices)
    if current_length != forecast_weeks:
        if current_length < forecast_weeks:
            st.session_state.weekly_prices.extend(
                [int(default_promo_price)] * (forecast_weeks - current_length))
        else:
            st.session_state.weekly_prices = st.session_state.weekly_prices[:forecast_weeks]

    # Adjust the list size if forecast_weeks changed
    current_length = len(st.session_state.weekly_prices)
    if current_length != forecast_weeks:
        if current_length < forecast_weeks:
            # Extend with default price
            st.session_state.weekly_prices.extend(
                [int(default_promo_price)] * (forecast_weeks - current_length))
        else:
            # Truncate
            st.session_state.weekly_prices = st.session_state.weekly_prices[:forecast_weeks]

    # Main panel - Split into two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Weekly promotional pricing controls in an expander
        with st.expander("🎛️ Promotional Pricing Strategy", expanded=False):
            st.markdown("**Configure promotional prices for each forecast week:**")

            # Create columns for better layout
            cols_per_row = 4
            for i in range(0, forecast_weeks, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    week_idx = i + j
                    if week_idx < forecast_weeks:
                        with col:
                            current_value = st.session_state.weekly_prices[week_idx]

                            # Ensure min_price is always less than or equal to
                            # current value
                            suggested_min = max(1, int(normal_price * 0.3))
                            min_price = min(suggested_min, current_value)

                            # Ensure max_price is always greater than or equal
                            # to current value
                            suggested_max = int(normal_price)
                            max_price = max(suggested_max, current_value)

                            st.session_state.weekly_prices[week_idx] = st.number_input(
                                f"Week {week_idx + 1}",
                                min_value=min_price,
                                max_value=max_price,
                                value=current_value,
                                step=100,
                                key=f"week_{week_idx}",
                                help=f"Week {week_idx + 1} promotional price"
                            )

            # Quick actions
            st.markdown("**Quick Actions:**")
            quick_cols = st.columns(4)

            with quick_cols[0]:
                if st.button("🔄 Reset to Default",
                             help="Reset all weeks to default promo price"):
                    for i in range(forecast_weeks):
                        st.session_state.weekly_prices[i] = int(
                            default_promo_price)
                    st.rerun()

            with quick_cols[1]:
                if st.button(
                    "� Apply 10% Discount",
                        help="Apply 10% discount to all weeks"):
                    discount_price = int(normal_price * 0.9)
                    for i in range(forecast_weeks):
                        st.session_state.weekly_prices[i] = max(
                            int(normal_price * 0.5), discount_price)
                    st.rerun()

            with quick_cols[2]:
                if st.button(
                    "🎯 Apply 20% Discount",
                        help="Apply 20% discount to all weeks"):
                    discount_price = int(normal_price * 0.8)
                    for i in range(forecast_weeks):
                        st.session_state.weekly_prices[i] = max(
                            int(normal_price * 0.5), discount_price)
                    st.rerun()

            with quick_cols[3]:
                avg_discount = (
                    normal_price - np.mean(st.session_state.weekly_prices)) / normal_price
                st.metric("Avg Discount", f"{avg_discount:.1%}")

        # Generate forecast with weekly prices
        with st.spinner("Generating forecast..."):
            model = models[selected_category]
            forecast = generate_forecast(
                model,
                historical_data,
                st.session_state.weekly_prices,
                forecast_weeks)

        # Build figure with required styles (historical: green lines, forecast:
        # red lines+markers)
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#2E8B57', width=2),
            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Sales: %{y:,.0f}<extra></extra>'
        ))

        # Confidence interval first (upper then lower with fill)
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(color='rgba(220,20,60,0)', width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(220,20,60,0.15)',
            line=dict(color='rgba(220,20,60,0)', width=0),
            name='Confidence Interval',
            hovertemplate='<b>CI</b><br>Date: %{x}<br>Lower: %{y:,.0f}<extra></extra>'
        ))
        # Forecast trace with markers
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#DC143C', width=2),
            marker=dict(color='#DC143C', size=6),
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Sales: %{y:,.0f}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Weekly Sales Forecast - {selected_category}",
            xaxis_title="Date",
            yaxis_title="Weekly Sales",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
            xaxis=dict(
                rangeslider=dict(
                    visible=True),
                type="date"))

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Forecast Summary")

        # Key metrics
        avg_forecast = forecast['yhat'].mean()
        total_forecast = forecast['yhat'].sum()
        avg_historical = historical_data['y'].tail(12).mean()

        # Display metrics
        col2a, col2b = st.columns(2)

        with col2a:
            st.metric(
                "📈 Avg Weekly Sales",
                f"{avg_forecast:,.0f}",
                delta=f"{avg_forecast - avg_historical:,.0f}"
            )

        with col2b:
            st.metric(
                "🎯 Total Forecast",
                f"{total_forecast:,.0f}",
                help=f"Sum of next {forecast_weeks} weeks"
            )

        # Weekly pricing summary
        st.subheader("💰 Weekly Pricing Summary")
        avg_promo_price = np.mean(st.session_state.weekly_prices)
        total_discount = sum(
            (normal_price - price) / normal_price for price in st.session_state.weekly_prices) / forecast_weeks

        pricing_col1, pricing_col2 = st.columns(2)
        with pricing_col1:
            st.metric(
                "Avg Promo Price",
                f"${avg_promo_price:,.0f}",
                delta=f"${avg_promo_price - default_promo_price:,.0f}"
            )

        with pricing_col2:
            st.metric(
                "Avg Discount Rate",
                f"{total_discount:.1%}",
                help="Average discount across all forecast weeks"
            )

        # Forecast table
        st.subheader("📋 Forecast Details")

        forecast_display = forecast.copy()
        forecast_display['Week'] = range(1, len(forecast_display) + 1)
        forecast_display['Date'] = forecast_display['ds'].dt.strftime(
            '%Y-%m-%d')
        forecast_display['Forecast'] = forecast_display['yhat'].round(
            0).astype(int)
        forecast_display['Lower'] = forecast_display['yhat_lower'].round(
            0).astype(int)
        forecast_display['Upper'] = forecast_display['yhat_upper'].round(
            0).astype(int)

        # Add pricing information
        forecast_display['Promo_Price'] = [
            f"${price:,}" for price in st.session_state.weekly_prices[:len(forecast_display)]]
        forecast_display['Discount'] = [f"{(normal_price - price) / normal_price:.1%}"
                                        for price in st.session_state.weekly_prices[:len(forecast_display)]]

        # Display table
        st.dataframe(
            forecast_display[['Week', 'Date', 'Forecast', 'Lower', 'Upper', 'Promo_Price', 'Discount']],
            use_container_width=True,
            height=400,
            column_config={
                "Week": st.column_config.NumberColumn("Week", width="small"),
                "Date": st.column_config.DateColumn("Date", width="medium"),
                "Forecast": st.column_config.NumberColumn("Forecast", format="%d"),
                "Lower": st.column_config.NumberColumn("Lower", format="%d"),
                "Upper": st.column_config.NumberColumn("Upper", format="%d"),
                "Promo_Price": st.column_config.TextColumn("Promo Price", width="medium"),
                "Discount": st.column_config.TextColumn("Discount %", width="small"),
            }
        )
        
        # Add insight box for forecast
        total_forecast = forecast_display['Forecast'].sum()
        max_week = forecast_display.loc[forecast_display['Forecast'].idxmax()]
        
        st.markdown(f"""
            <div class="insight-box">
                💡 <strong>Forecast Insights:</strong> The model predicts a total of {total_forecast:,} units over {forecast_weeks} weeks. 
                Peak sales are expected in Week {max_week['Week']} with {max_week['Forecast']:,} units. 
                Average weekly sales forecast: {total_forecast/forecast_weeks:.0f} units.
            </div>
        """, unsafe_allow_html=True)

    # Additional insights
    st.divider()

    col3, col4, col5 = st.columns(3)

    with col3:
        st.subheader("📊 Historical Performance")
        recent_avg = historical_data['y'].tail(4).mean()
        overall_avg = historical_data['y'].mean()

        st.metric(
            "Recent 4-week Average",
            f"{recent_avg:,.0f}",
            delta=f"{recent_avg - overall_avg:,.0f}"
        )

    with col4:
        st.subheader("🎯 Pricing Impact")
        avg_promo_price = np.mean(st.session_state.weekly_prices)
        price_change = avg_promo_price - default_promo_price

        st.metric(
            "Avg Price Change",
            f"${price_change:,.0f}",
            help="Average change from default promotional price"
        )

    with col5:
        st.subheader("📈 Forecast vs Historical")
        forecast_change = avg_forecast - avg_historical
        st.metric(
            "Forecast vs Avg Historical",
            f"{avg_forecast:,.0f}",
            delta=f"{forecast_change:,.0f}",
            help="Comparison with recent historical average"
        )


def create_enhanced_analytics_dashboard(data):
    """Create the enhanced sales analytics dashboard with better insights and organization"""

    # Key business metrics at the top
    col1, col2, col3, col4, col5 = st.columns(5)

    total_sales = data['weekly_sales'].sum()
    total_categories = data['category'].nunique()
    total_weeks = data['week'].nunique()
    promo_coverage = (data['promo_active'].sum() / len(data)) * 100
    avg_weekly_sales = data['weekly_sales'].mean()

    with col1:
        st.metric("📈 Total Sales", f"{total_sales:,.0f}")
    with col2:
        st.metric("🏷️ Categories", total_categories)
    with col3:
        st.metric("📅 Weeks", total_weeks)
    with col4:
        st.metric("🎯 Promo Coverage", f"{promo_coverage:.1f}%")
    with col5:
        st.metric("📊 Avg Weekly", f"{avg_weekly_sales:.0f}")

    st.divider()

    # Sidebar filters for the analytics
    with st.sidebar:
        st.markdown("### 🔍 Analytics Filters")

        # Category filter
        all_categories = ['All Categories'] + \
            sorted(data['category'].unique().tolist())
        selected_analytics_category = st.selectbox(
            "Filter by Category",
            all_categories,
            key="analytics_category_filter"
        )

        # Week range filter
        min_week, max_week = int(data['week'].min()), int(data['week'].max())
        week_range = st.slider(
            "Week Range",
            min_week, max_week,
            (min_week, max_week),
            key="analytics_week_range"
        )

        # Apply filters
        filtered_data = data[
            (data['week'] >= week_range[0]) &
            (data['week'] <= week_range[1])
        ].copy()

        if selected_analytics_category != 'All Categories':
            filtered_data = filtered_data[filtered_data['category']
                                          == selected_analytics_category]

        st.info(f"📊 Filtered data: {len(filtered_data):,} records")

    # Create tabs for different analytics sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Pareto Analysis",
        "📈 Performance Matrix",
        "💰 Revenue Analysis",
        "📊 Time Series",
        "🔄 Seasonal Patterns"
    ])

    # === PARETO ANALYSIS TAB ===
    with tab1:
        st.subheader("🎯 Pareto Analysis (80/20 Rule)")

        with st.spinner("Generating Pareto analysis..."):
            pareto_fig, categories_80pct, pct_categories_80pct, category_sales = pareto_analysis(
                filtered_data)

        st.plotly_chart(pareto_fig, use_container_width=True)
        
        # Insight box after chart
        st.markdown(f"""
        <div class="insight-box">
            <strong>💡 Pareto Insight:</strong> {categories_80pct} categories ({pct_categories_80pct:.1f}% of all categories) 
            generate 80% of total sales. Focus marketing efforts on these high-performers for maximum ROI.
        </div>
        """, unsafe_allow_html=True)

        # Enhanced insights
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Categories for 80%",
                categories_80pct,
                help="Number of categories needed to reach 80% of sales"
            )

        with col2:
            st.metric(
                "Efficiency Ratio",
                f"{pct_categories_80pct:.1f}%",
                help="Percentage of categories generating 80% of sales"
            )

        with col3:
            top_category_share = (
                category_sales.iloc[0] / category_sales.sum()) * 100
            st.metric(
                "Top Category Share",
                f"{top_category_share:.1f}%",
                help=f"Share of top category: {category_sales.index[0]}"
            )

        # Detailed insights
        with st.expander("📋 Detailed Pareto Insights", expanded=False):
            st.markdown("**🔍 Key Findings:**")

            # Top performers
            top_5_categories = category_sales.head(5)
            top_5_contribution = (
                top_5_categories.sum() / category_sales.sum()) * 100

            st.markdown(f"""
            - **Top 5 Categories** contribute **{top_5_contribution:.1f}%** of total sales
            - **Most Important Category:** {category_sales.index[0]} ({top_category_share:.1f}% of sales)
            - **Recommendation:** Focus inventory and marketing on top {categories_80pct} categories
            - **Strategy:** Consider consolidating or improving bottom performers
            """)

            # Show detailed table
            pareto_table = pd.DataFrame({
                'Category': category_sales.index,
                'Total Sales': category_sales.values,
                'Cumulative %': (category_sales.cumsum() / category_sales.sum() * 100).round(1),
                'Individual %': (category_sales / category_sales.sum() * 100).round(1)
            })

            st.dataframe(pareto_table, use_container_width=True, height=300)

    # === PERFORMANCE MATRIX TAB ===
    with tab2:
        st.subheader("📈 Category Performance Matrix")

        with st.spinner("Analyzing category performance..."):
            performance_data = category_performance_matrix(filtered_data)

        # Create scatter plot: Sales Volume vs Promo Effectiveness
        fig_matrix = px.scatter(
            performance_data,
            x='total_sales',
            y='promo_effectiveness',
            size='avg_sales',
            color='promo_frequency',
            hover_name='category',
            title="Category Performance Matrix: Sales Volume vs Promo Effectiveness",
            labels={
                'total_sales': 'Total Sales Volume',
                'promo_effectiveness': 'Promo Effectiveness (%)',
                'promo_frequency': 'Promo Frequency',
                'avg_sales': 'Avg Weekly Sales'},
            color_continuous_scale='Viridis')

        # Add quadrant lines
        avg_sales = performance_data['total_sales'].median()
        avg_effectiveness = performance_data['promo_effectiveness'].median()

        fig_matrix.add_hline(
            y=avg_effectiveness,
            line_dash="dash",
            line_color="gray",
            annotation_text="Avg Effectiveness")
        fig_matrix.add_vline(x=avg_sales, line_dash="dash", line_color="gray",
                             annotation_text="Avg Sales")

        fig_matrix.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Quadrant analysis
        st.markdown("#### 🎯 Strategic Quadrants")

        quad_col1, quad_col2 = st.columns(2)

        with quad_col1:
            # High sales, high effectiveness (Stars)
            stars = performance_data[
                (performance_data['total_sales'] > avg_sales) &
                (performance_data['promo_effectiveness'] > avg_effectiveness)
            ]

            st.markdown("**⭐ Stars (High Sales + High Promo Effectiveness):**")
            if not stars.empty:
                for _, row in stars.iterrows():
                    st.markdown(
                        f"• {
                            row['category']} ({
                            row['promo_effectiveness']:.1f}% boost)")
            else:
                st.markdown("• No categories in this quadrant")

        with quad_col2:
            # Low sales, high effectiveness (Opportunities)
            opportunities = performance_data[
                (performance_data['total_sales'] <= avg_sales) &
                (performance_data['promo_effectiveness'] > avg_effectiveness)
            ]

            st.markdown(
                "**🚀 Opportunities (Low Sales + High Promo Effectiveness):**")
            if not opportunities.empty:
                for _, row in opportunities.iterrows():
                    st.markdown(
                        f"• {
                            row['category']} ({
                            row['promo_effectiveness']:.1f}% boost)")
            else:
                st.markdown("• No categories in this quadrant")

        # Performance table
        with st.expander("📊 Detailed Performance Metrics", expanded=False):
            st.dataframe(
                performance_data.sort_values(
                    'total_sales', ascending=False), use_container_width=True, column_config={
                    "total_sales": st.column_config.NumberColumn(
                        "Total Sales", format="%.0f"), "avg_sales": st.column_config.NumberColumn(
                        "Avg Weekly Sales", format="%.0f"), "promo_frequency": st.column_config.NumberColumn(
                        "Promo Frequency", format="%.2f"), "promo_effectiveness": st.column_config.NumberColumn(
                            "Promo Effectiveness (%)", format="%.1f")})

    # === REVENUE ANALYSIS TAB ===
    with tab3:
        st.subheader("💰 Revenue & Profitability Analysis")

        with st.spinner("Calculating revenue metrics..."):
            category_revenue, promo_rev, non_promo_rev, weekly_revenue = calculate_revenue_analysis(
                filtered_data)

        # Revenue comparison: Promo vs Non-Promo
        col1, col2 = st.columns(2)

        with col1:
            revenue_impact = (
                (promo_rev - non_promo_rev) / non_promo_rev) * 100

            fig_rev_comparison = go.Figure(data=[
                go.Bar(
                    x=['Non-Promo Weeks', 'Promo Weeks'],
                    y=[non_promo_rev, promo_rev],
                    marker_color=['#FF6B6B', '#4ECDC4'],
                    text=[f'${non_promo_rev:,.0f}', f'${promo_rev:,.0f}'],
                    textposition='outside'
                )
            ])

            fig_rev_comparison.update_layout(
                title="Average Revenue: Promo vs Non-Promo Weeks",
                yaxis_title="Average Revenue per Week",
                showlegend=False,
                height=400
            )

            st.plotly_chart(fig_rev_comparison, use_container_width=True)

            if revenue_impact > 0:
                st.success(
                    f"💰 **Revenue Impact:** Promos increase avg revenue by **{revenue_impact:.1f}%**")
            else:
                st.warning(
                    f"⚠️ **Revenue Impact:** Promos decrease avg revenue by **{abs(revenue_impact):.1f}%**")

        with col2:
            # Top revenue categories
            top_revenue_categories = category_revenue.head(8)

            fig_rev_categories = go.Figure(data=[
                go.Bar(
                    x=top_revenue_categories['revenue'],
                    y=top_revenue_categories.index,
                    orientation='h',
                    marker_color='#2E8B57',
                    text=[f'${val:,.0f}' for val in top_revenue_categories['revenue']],
                    textposition='outside'
                )
            ])

            fig_rev_categories.update_layout(
                title="Top 8 Categories by Total Revenue",
                xaxis_title="Total Revenue",
                yaxis_title="Categories",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_rev_categories, use_container_width=True)

        # Weekly revenue trend
        st.markdown("#### 📈 Weekly Revenue Trend")

        # Create datetime for better x-axis
        start_date = datetime(2020, 1, 1)
        weekly_revenue['date'] = pd.to_datetime(
            [start_date + timedelta(weeks=w - 1) for w in weekly_revenue['week']])

        fig_weekly_rev = go.Figure()

        fig_weekly_rev.add_trace(go.Scatter(
            x=weekly_revenue['date'],
            y=weekly_revenue['revenue'],
            mode='lines+markers',
            name='Weekly Revenue',
            line=dict(color='#2E8B57', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Week %{text}</b><br>Date: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>',
            text=weekly_revenue['week']
        ))

        fig_weekly_rev.update_layout(
            title="Weekly Total Revenue Trend",
            xaxis_title="Date",
            yaxis_title="Weekly Revenue ($)",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig_weekly_rev, use_container_width=True)

        # Revenue insights
        with st.expander("💡 Revenue Insights & Recommendations", expanded=False):
            total_revenue = weekly_revenue['revenue'].sum()
            avg_revenue = weekly_revenue['revenue'].mean()
            peak_revenue_week = weekly_revenue.loc[weekly_revenue['revenue'].idxmax(
            )]
            low_revenue_week = weekly_revenue.loc[weekly_revenue['revenue'].idxmin(
            )]

            st.markdown(f"""
            **📊 Revenue Summary:**
            - **Total Revenue:** ${total_revenue:,.0f}
            - **Average Weekly Revenue:** ${avg_revenue:,.0f}
            - **Peak Revenue Week:** Week {peak_revenue_week['week']} (${peak_revenue_week['revenue']:,.0f})
            - **Lowest Revenue Week:** Week {low_revenue_week['week']} (${low_revenue_week['revenue']:,.0f})

            **💡 Strategic Recommendations:**
            - {'Focus on maintaining promo strategy' if revenue_impact > 0 else 'Review promotional pricing strategy'}
            - Investigate factors behind peak week {peak_revenue_week['week']} performance
            - Address potential issues during low-performing week {low_revenue_week['week']}
            """)

    # === TIME SERIES TAB ===
    with tab4:
        st.subheader("📊 Time Series Analysis")

        # Enhanced time series with multiple metrics
        time_metrics = filtered_data.groupby('week').agg({
            'weekly_sales': ['sum', 'mean'],
            'promo_active': 'mean',
            'category': 'nunique'
        }).round(2)

        time_metrics.columns = [
            'total_sales',
            'avg_sales',
            'promo_rate',
            'active_categories']
        time_metrics = time_metrics.reset_index()

        # Create datetime
        start_date = datetime(2020, 1, 1)
        time_metrics['date'] = pd.to_datetime(
            [start_date + timedelta(weeks=w - 1) for w in time_metrics['week']])

        # Multi-metric time series
        fig_ts = go.Figure()

        # Sales trend
        fig_ts.add_trace(go.Scatter(
            x=time_metrics['date'],
            y=time_metrics['total_sales'],
            mode='lines+markers',
            name='Total Weekly Sales',
            line=dict(color='#2E8B57', width=2),
            yaxis='y1'
        ))

        # Promo rate overlay
        fig_ts.add_trace(go.Scatter(
            x=time_metrics['date'],
            # Scale for visibility
            y=time_metrics['promo_rate'] * max(time_metrics['total_sales']),
            mode='lines',
            name='Promo Activity (Scaled)',
            line=dict(color='#FF6B6B', width=1, dash='dot'),
            yaxis='y1',
            opacity=0.7
        ))

        fig_ts.update_layout(
            title="Weekly Sales Trend with Promotional Activity",
            xaxis_title="Date",
            yaxis_title="Weekly Sales",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig_ts, use_container_width=True)

        # Trend analysis
        col1, col2, col3 = st.columns(3)

        # Calculate trend metrics
        first_quarter = time_metrics['total_sales'][:len(
            time_metrics) // 4].mean()
        last_quarter = time_metrics['total_sales'][-len(
            time_metrics) // 4:].mean()
        growth_rate = ((last_quarter - first_quarter) / first_quarter) * 100

        peak_week = time_metrics.loc[time_metrics['total_sales'].idxmax()]
        valley_week = time_metrics.loc[time_metrics['total_sales'].idxmin()]

        with col1:
            st.metric(
                "📈 Growth Rate",
                f"{growth_rate:+.1f}%",
                help="First quarter vs last quarter comparison"
            )

        with col2:
            st.metric(
                "🏔️ Peak Week",
                f"Week {peak_week['week']}",
                f"{peak_week['total_sales']:,.0f} sales"
            )

        with col3:
            st.metric(
                "📉 Valley Week",
                f"Week {valley_week['week']}",
                f"{valley_week['total_sales']:,.0f} sales"
            )

        # Correlation analysis
        with st.expander("📊 Correlation Analysis", expanded=False):
            correlation_data = time_metrics[[
                'total_sales', 'avg_sales', 'promo_rate', 'active_categories']].corr()

            fig_corr = px.imshow(
                correlation_data,
                title="Correlation Matrix: Sales vs Other Factors",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            fig_corr.update_layout(height=400)

            st.plotly_chart(fig_corr, use_container_width=True)

            promo_correlation = correlation_data.loc['total_sales', 'promo_rate']
            st.info(
                f"📊 **Key Finding:** Promotion rate has a {
                    promo_correlation:.3f} correlation with total sales")

    # === SEASONAL PATTERNS TAB ===
    with tab5:
        st.subheader("🔄 Seasonal & Cyclical Patterns")

        with st.spinner("Analyzing seasonal patterns..."):
            seasonal_data, peaks, valleys, avg_sales = seasonal_analysis(
                filtered_data)

        # Seasonal pattern chart
        fig_seasonal = go.Figure()

        fig_seasonal.add_trace(go.Scatter(
            x=seasonal_data['season_week'],
            y=seasonal_data['weekly_sales'],
            mode='lines+markers',
            name='Seasonal Pattern',
            line=dict(color='#2E8B57', width=2),
            marker=dict(size=6)
        ))

        # Add average line
        fig_seasonal.add_hline(
            y=avg_sales,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Average: {avg_sales:.0f}"
        )

        # Highlight peaks
        if not peaks.empty:
            fig_seasonal.add_trace(go.Scatter(
                x=peaks['season_week'],
                y=peaks['weekly_sales'],
                mode='markers',
                name='Peak Weeks',
                marker=dict(color='red', size=10, symbol='triangle-up')
            ))

        # Highlight valleys
        if not valleys.empty:
            fig_seasonal.add_trace(go.Scatter(
                x=valleys['season_week'],
                y=valleys['weekly_sales'],
                mode='markers',
                name='Valley Weeks',
                marker=dict(color='blue', size=10, symbol='triangle-down')
            ))

        fig_seasonal.update_layout(
            title="Seasonal Sales Pattern (52-Week Cycle)",
            xaxis_title="Week of Year",
            yaxis_title="Average Weekly Sales",
            height=500
        )

        st.plotly_chart(fig_seasonal, use_container_width=True)

        # Seasonal insights
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🏔️ Peak Seasons:**")
            if not peaks.empty:
                for _, peak in peaks.iterrows():
                    st.markdown(
                        f"• Week {int(peak['season_week'])}: {peak['weekly_sales']:.0f} sales")
            else:
                st.markdown("• No significant peaks identified")

        with col2:
            st.markdown("**📉 Low Seasons:**")
            if not valleys.empty:
                for _, valley in valleys.iterrows():
                    st.markdown(
                        f"• Week {int(valley['season_week'])}: {valley['weekly_sales']:.0f} sales")
            else:
                st.markdown("• No significant valleys identified")

        with col3:
            volatility = seasonal_data['weekly_sales'].std()
            cv = (volatility / avg_sales) * 100
            st.metric(
                "📊 Sales Volatility",
                f"{cv:.1f}%",
                help="Coefficient of variation across weeks"
            )

        # Detailed seasonal recommendations
        with st.expander("🎯 Seasonal Strategy Recommendations", expanded=False):
            st.markdown("""
            **📈 Seasonal Optimization Strategy:**

            **Peak Season Management:**
            - Increase inventory levels during identified peak weeks
            - Reduce promotional intensity when demand is naturally high
            - Focus on premium products and higher margins

            **Valley Season Recovery:**
            - Implement aggressive promotional campaigns during low-demand periods
            - Consider introducing new products or seasonal items
            - Optimize supply chain to reduce costs during slow periods

            **Overall Recommendations:**
            - Plan promotional calendar around seasonal patterns
            - Adjust staffing levels based on expected demand fluctuations
            - Develop category-specific seasonal strategies
            """)

    # === EXECUTIVE SUMMARY ===
    st.divider()
    st.markdown("## 📋 Executive Summary")
    
    # Calculate key business insights
    total_revenue = filtered_data['weekly_sales'].sum()
    avg_weekly_revenue = total_revenue / len(filtered_data['week'].unique())
    top_category = filtered_data.groupby('category')['weekly_sales'].sum().idxmax()
    top_category_sales = filtered_data.groupby('category')['weekly_sales'].sum().max()
    
    # Seasonal insights - fix column reference
    seasonal_peak = seasonal_data.loc[seasonal_data['weekly_sales'].idxmax(), 'season_week']
    seasonal_valley = seasonal_data.loc[seasonal_data['weekly_sales'].idxmin(), 'season_week']
    
    st.markdown(f"""
        <div class="ai-card">
            <h4 style="margin-top: 0; color: #92400e;">🎯 Key Business Insights</h4>
            <ul style="margin: 0; padding-left: 20px; color: #451a03;">
                <li><strong>Revenue Performance:</strong> Total revenue of ${total_revenue:,.0f} with average weekly sales of ${avg_weekly_revenue:,.0f}</li>
                <li><strong>Top Category:</strong> {top_category} dominates with ${top_category_sales:,.0f} in total sales</li>
                <li><strong>Seasonal Pattern:</strong> Peak demand in Week {seasonal_peak}, lowest in Week {seasonal_valley}</li>
                <li><strong>Growth Trend:</strong> {growth_rate:+.1f}% growth rate from first to last quarter</li>
                <li><strong>Market Dynamics:</strong> Promotional activities show {promo_correlation:.3f} correlation with sales</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


def create_analytics_dashboard(data):
    """Create the enhanced sales analytics dashboard content"""
    create_enhanced_analytics_dashboard(data)


@st.cache_data
def generate_scenario_forecasts(_model, historical_data, normal_price, forecast_weeks=16):
    """
    Generate forecasts for different discount scenarios (baseline, 10%, 20%)
    Note: _model parameter starts with underscore to avoid Streamlit hashing issues
    """
    try:
        scenarios = {}
        
        # Baseline scenario (current planned prices from session state)
        baseline_prices = st.session_state.get('weekly_prices', [normal_price] * forecast_weeks)
        baseline_forecast = generate_forecast(_model, historical_data, baseline_prices, forecast_weeks)
        scenarios['baseline'] = {
            'forecast': baseline_forecast[['ds', 'yhat']].copy(),
            'discount_rate': ((normal_price - baseline_prices[0]) / normal_price) if baseline_prices[0] != normal_price else 0.0,
            'prices': baseline_prices,
            'description': 'Current Planned Prices',
            'total_forecast': baseline_forecast['yhat'].sum(),
            'avg_weekly': baseline_forecast['yhat'].mean()
        }
        
        # 10% discount scenario
        discount_10_prices = [normal_price * 0.9] * forecast_weeks
        forecast_10 = generate_forecast(_model, historical_data, discount_10_prices, forecast_weeks)
        scenarios['discount_10'] = {
            'forecast': forecast_10[['ds', 'yhat']].copy(),
            'discount_rate': 0.10,
            'prices': discount_10_prices,
            'description': '10% Discount',
            'total_forecast': forecast_10['yhat'].sum(),
            'avg_weekly': forecast_10['yhat'].mean()
        }
        
        # 20% discount scenario
        discount_20_prices = [normal_price * 0.8] * forecast_weeks
        forecast_20 = generate_forecast(_model, historical_data, discount_20_prices, forecast_weeks)
        scenarios['discount_20'] = {
            'forecast': forecast_20[['ds', 'yhat']].copy(),
            'discount_rate': 0.20,
            'prices': discount_20_prices,
            'description': '20% Discount',
            'total_forecast': forecast_20['yhat'].sum(),
            'avg_weekly': forecast_20['yhat'].mean()
        }
        
        return scenarios
        
    except Exception as e:
        st.error(f"Error generating scenario forecasts: {str(e)}")
        return {}


@st.cache_data
def build_scenario_context_for_ai(
        selected_category,
        data,
        scenarios,
        normal_price,
        forecast_weeks=16
    ):
    """Build enhanced context with scenario comparisons for AI analysis"""
    try:
        # Get historical data for the category
        category_data = data[data['category'] == selected_category].copy()
        historical_avg = category_data['weekly_sales'].mean()
        recent_4week_avg = category_data['weekly_sales'].tail(4).mean()
        
        # Build scenario comparison
        scenario_summary = []
        for key, scenario in scenarios.items():
            discount_pct = scenario['discount_rate'] * 100
            scenario_summary.append(
                f"- {scenario['description']} ({discount_pct:.0f}% discount): "
                f"Next week forecast = {scenario['forecast']['yhat'].iloc[0]:.0f} units, "
                f"Total {forecast_weeks}-week forecast = {scenario['total_forecast']:.0f} units"
            )
        
        # Calculate uplift comparisons
        baseline_total = scenarios['baseline']['total_forecast']
        uplift_10 = ((scenarios['discount_10']['total_forecast'] - baseline_total) / baseline_total * 100) if baseline_total > 0 else 0
        uplift_20 = ((scenarios['discount_20']['total_forecast'] - baseline_total) / baseline_total * 100) if baseline_total > 0 else 0
        
        # Build comprehensive context string
        context = f"""Context Data:
Category: {selected_category}
Normal price: {normal_price:,}
Current planned promo prices: {scenarios['baseline']['prices'][:4]}... (showing first 4 weeks)

Scenario forecasts for next {forecast_weeks} weeks:
{chr(10).join(scenario_summary)}

Sales uplift analysis:
- 10% discount scenario: {uplift_10:+.1f}% increase vs baseline
- 20% discount scenario: {uplift_20:+.1f}% increase vs baseline

Historical performance:
- Historical average weekly sales: {historical_avg:.0f} units
- Recent 4-week average: {recent_4week_avg:.0f} units
- Total historical weeks: {len(category_data)}

Margin considerations:
- Normal price: ${normal_price:,}
- 10% discount price: ${normal_price * 0.9:,.0f} (margin reduction: 10%)
- 20% discount price: ${normal_price * 0.8:,.0f} (margin reduction: 20%)"""

        return context
        
    except Exception as e:
        return f"Error building scenario context: {str(e)}"


@st.cache_data
def build_context_for_ai(
        selected_category,
        data,
        forecast,
        weekly_prices,
        normal_price):
    """Build context string for AI recommendations"""
    try:
        # Get historical data for the category
        category_data = data[data['category'] == selected_category].copy()
        historical_avg = category_data['weekly_sales'].mean()

        # Calculate discount rates
        discount_rates = [
            (normal_price -
             price) /
            normal_price *
            100 for price in weekly_prices]

        # Format forecast data
        forecast_dict = {}
        for idx, row in forecast.iterrows():
            forecast_dict[row['ds'].strftime('%Y-%m-%d')] = int(row['yhat'])

        # Build context string
        context = f"""Context Data:
Category: {selected_category}
Normal price: {normal_price:,}
Promo prices: {weekly_prices}
Discount rates: {[f"{rate:.1f}%" for rate in discount_rates]}
Historical average sales: {historical_avg:.0f} units/week
Forecast next {len(forecast)} weeks: {forecast_dict}
Total historical weeks: {len(category_data)}
Recent 4-week average: {category_data['weekly_sales'].tail(4).mean():.0f} units/week"""

        return context
    except Exception as e:
        return f"Error building context: {str(e)}"


def clean_ai_text(text: str) -> str:
    """
    Clean AI text with minimal processing - just basic whitespace cleanup.
    Preserve all markdown formatting for proper Streamlit rendering.
    """
    if not text or text.startswith("Error"):
        return text
    
    import re
    
    try:
        # Just basic cleanup, preserve all markdown
        cleaned = text.strip()
        
        # Only fix excessive whitespace
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Max 2 consecutive newlines
        
        return cleaned
        
    except Exception as e:
        return text.strip()


def post_process_ai_response(raw_text):
    """
    Post-process AI recommendation output using the advanced cleaning function.
    This is maintained for backward compatibility.
    """
    return clean_ai_text(raw_text)


def get_scenario_ai_recommendation(context):
    """Get AI recommendation with scenario comparison from Google Gemini"""
    try:
        # Enhanced guardrails for scenario comparison
        guardrails = """You are a sales forecasting assistant.
Compare the baseline forecast with the 10% and 20% discount scenarios.
Recommend the most effective discount strategy, balancing sales uplift and margin considerations.
Answer ONLY using the provided context data below.

Analyze and provide insights on:
1. Which discount scenario offers the best balance of volume increase vs margin loss
2. Whether the forecasted sales uplift justifies the discount level
3. Risk assessment for each scenario (market response, competitor reaction)
4. Specific recommendations for pricing strategy implementation

If certain information is missing from the context, acknowledge it rather than inventing numbers.
Format your response in clear sections for easy reading."""

        # Combine guardrails + context
        full_prompt = f"{guardrails}\n\n{context}"

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Generate response
        response = model.generate_content(full_prompt)
        return response.text

    except ImportError:
        # Fallback when Gemini is not available
        return """## Scenario Analysis Results

**1. Discount Strategy Comparison:**
Based on the provided scenarios, all discount levels show varying impacts on forecasted sales volume.

**2. Volume vs Margin Analysis:**
Each discount scenario represents a trade-off between increased sales volume and reduced profit margins per unit.

**3. Risk Assessment:**
- Baseline scenario maintains current margin structure
- 10% discount may provide moderate volume uplift with acceptable margin reduction  
- 20% discount offers higher volume potential but significant margin impact

**4. Strategic Recommendations:**
Consider testing the 10% discount scenario initially to gauge market response before implementing deeper discounts.
Monitor competitor pricing and customer response to optimize discount strategy.

*Note: Full scenario analysis requires Google Gemini API configuration and complete context data.*"""
        
    except Exception as e:
        return f"**Error generating scenario recommendations:** {
            str(e)}\n\nPlease check your Gemini API configuration."


def get_ai_recommendation(context):
    """Get automatic AI recommendation from Google Gemini"""
    try:
        # Guardrails prefix
        guardrails = """You are a sales forecasting assistant.
Provide actionable recommendations based ONLY on the Context Data provided below.
If certain information is missing, acknowledge it and do not invent numbers.

Generate insights covering:
1. Whether the planned promo prices are effective
2. Which weeks might see strong or weak sales
3. Suggestions for adjusting promo prices to maximize sales/profit
4. Highlight risks (e.g., if discount is too low to impact sales)

Format your response in clear sections with bullet points for easy reading.
"""

        # Combine guardrails + context
        full_prompt = f"{guardrails}\n{context}"

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Generate response
        response = model.generate_content(full_prompt)
        return response.text

    except ImportError:
        # Fallback when Gemini is not available
        return """## Sales Forecasting Insights

**1. Effectiveness of Planned Promo Prices:**
* The promotional strategy appears to be in development phase
* Consider implementing A/B testing for optimal pricing

**2. Weekly Sales Prediction:**
* Sales patterns may vary based on seasonal factors
* Monitor competitor pricing and market conditions

**3. Promo Price Optimization:**
* **Recommendation:** Test different discount levels (10-25% range)
* Focus on volume-driven categories for deeper discounts

**4. Risk Assessment:**
* Insufficient historical promotion data may limit accuracy
* Consider gradual rollout of new pricing strategies

*Note: Full AI recommendations require Google Gemini API configuration.*"""
        
    except Exception as e:
        return f"**Error generating AI recommendations:** {
            str(e)}\n\nPlease check your Gemini API configuration."


def create_ai_recommendation_tab(models, data):
    """Create the automatic AI recommendation interface"""
    
    # Get current selected category from session state (from forecasting tab)
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = "Sirup"  # Default

    if 'weekly_prices' not in st.session_state:
        st.session_state.weekly_prices = [9999] * 16  # Default prices

    # Category selector for AI analysis
    categories = sorted(list(models.keys()))
    selected_category = st.selectbox(
        "🏷️ Select category for AI analysis:",
        categories,
        index=categories.index(
            st.session_state.selected_category) if st.session_state.selected_category in categories else 0,
        key="ai_category_select")

    # Update session state
    st.session_state.selected_category = selected_category

    st.write("")
    st.divider()

    # Generate AI recommendations automatically with scenario analysis
    with st.spinner("🤖 AI is analyzing multiple discount scenarios and generating recommendations..."):
        try:
            # Get data for selected category
            category_data = data[data['category'] == selected_category].copy()
            historical_data = prepare_prophet_data(category_data)

            # Get pricing info
            last_row = historical_data.iloc[-1]
            normal_price = last_row['normal_price']

            # Generate scenario forecasts (baseline, 10%, 20% discount)
            model = models[selected_category]
            scenarios = generate_scenario_forecasts(model, historical_data, normal_price, 16)
            
            if not scenarios:
                st.error("Unable to generate scenario forecasts. Please try again.")
                return

            # Build enhanced context for AI with scenario comparisons
            context = build_scenario_context_for_ai(
                selected_category,
                data,
                scenarios,
                normal_price,
                16
            )

            # Get scenario-based AI recommendations
            raw_ai_recommendations = get_scenario_ai_recommendation(context)
            
            # Post-process the recommendations to clean markdown
            clean_ai_recommendations = post_process_ai_response(raw_ai_recommendations)

            # Display scenario comparison first
            st.markdown("## 📊 Discount Scenario Analysis")
            
            # Create metrics columns for scenario comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📈 Baseline Scenario")
                baseline = scenarios['baseline']
                st.metric(
                    "Total 16-Week Forecast",
                    f"{baseline['total_forecast']:,.0f} units"
                )
                st.metric(
                    "Average Weekly Sales",
                    f"{baseline['avg_weekly']:.0f} units"
                )
                st.metric(
                    "Discount Rate",
                    f"{baseline['discount_rate']*100:.0f}%"
                )
            
            with col2:
                st.markdown("### 🎯 10% Discount Scenario")
                discount_10 = scenarios['discount_10']
                uplift_10 = ((discount_10['total_forecast'] - baseline['total_forecast']) / baseline['total_forecast'] * 100) if baseline['total_forecast'] > 0 else 0
                st.metric(
                    "Total 16-Week Forecast",
                    f"{discount_10['total_forecast']:,.0f} units",
                    delta=f"{uplift_10:+.1f}%"
                )
                st.metric(
                    "Average Weekly Sales",
                    f"{discount_10['avg_weekly']:.0f} units"
                )
                st.metric(
                    "Price per Unit",
                    f"${normal_price * 0.9:,.0f}"
                )
            
            with col3:
                st.markdown("### 🚀 20% Discount Scenario") 
                discount_20 = scenarios['discount_20']
                uplift_20 = ((discount_20['total_forecast'] - baseline['total_forecast']) / baseline['total_forecast'] * 100) if baseline['total_forecast'] > 0 else 0
                st.metric(
                    "Total 16-Week Forecast",
                    f"{discount_20['total_forecast']:,.0f} units",
                    delta=f"{uplift_20:+.1f}%"
                )
                st.metric(
                    "Average Weekly Sales",
                    f"{discount_20['avg_weekly']:.0f} units"
                )
                st.metric(
                    "Price per Unit",
                    f"${normal_price * 0.8:,.0f}"
                )

            # Display AI recommendations  
            st.markdown("## 🎯 AI-Generated Strategic Recommendations")
            
            # Apply minimal text cleaning (preserving markdown)
            clean_text = clean_ai_text(clean_ai_recommendations)
            
            # Simply render the markdown content without the problematic container
            st.markdown(clean_text, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"**Error generating scenario recommendations:** {str(e)}")
            st.info("Please ensure all required data is available and try again.")
            
            # Fallback to simple analysis if scenario generation fails
            try:
                # Get data for selected category
                category_data = data[data['category'] == selected_category].copy()
                historical_data = prepare_prophet_data(category_data)

                # Get pricing info
                last_row = historical_data.iloc[-1]
                normal_price = last_row['normal_price']

                # Generate basic forecast with current pricing strategy
                model = models[selected_category]
                forecast = generate_forecast(
                    model, historical_data, st.session_state.weekly_prices, 16)

                # Build basic context for AI
                context = build_context_for_ai(
                    selected_category,
                    data,
                    forecast,
                    st.session_state.weekly_prices,
                    normal_price
                )

                # Get basic AI recommendations
                raw_ai_recommendations = get_ai_recommendation(context)
                clean_ai_recommendations = post_process_ai_response(raw_ai_recommendations)

                st.markdown("## 🎯 Basic AI Recommendations")
                
                # Apply advanced text cleaning
                super_clean_text = clean_ai_text(clean_ai_recommendations)
                
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                        border: 2px solid #22c55e; 
                        border-radius: 12px; 
                        padding: 1.5rem; 
                        margin: 1.5rem 0; 
                        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.15);
                        word-wrap: break-word; 
                        white-space: pre-wrap;
                        overflow-wrap: break-word;
                        max-width: 100%;
                        overflow: hidden;
                        line-height: 1.6;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    ">
                        {super_clean_text}
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as fallback_error:
                st.error(f"Fallback analysis also failed: {str(fallback_error)}")

    st.write("")
    st.divider()

    # Enhanced insights section with scenario data
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Scenario Performance Summary")
        try:
            if 'scenarios' in locals() and scenarios:
                baseline = scenarios['baseline']
                discount_10 = scenarios['discount_10'] 
                discount_20 = scenarios['discount_20']
                
                # Calculate revenue projections (simplified)
                baseline_revenue = baseline['total_forecast'] * normal_price
                revenue_10 = discount_10['total_forecast'] * (normal_price * 0.9)
                revenue_20 = discount_20['total_forecast'] * (normal_price * 0.8)
                
                st.metric("Baseline Revenue (16 weeks)", f"${baseline_revenue:,.0f}")
                st.metric("10% Discount Revenue", f"${revenue_10:,.0f}", 
                         delta=f"${revenue_10 - baseline_revenue:,.0f}")
                st.metric("20% Discount Revenue", f"${revenue_20:,.0f}",
                         delta=f"${revenue_20 - baseline_revenue:,.0f}")
            else:
                st.info("Scenario data not available. Using current strategy metrics.")
                try:
                    avg_discount = (normal_price - np.mean(st.session_state.weekly_prices)) / normal_price * 100
                    st.metric("Current Average Discount", f"{avg_discount:.1f}%")
                except:
                    st.metric("Current Average Discount", "N/A")
        except Exception as e:
            st.error(f"Error calculating scenario metrics: {str(e)}")

    with col2:
        st.markdown("### 🔄 Quick Actions")

        if st.button("🔄 Refresh Recommendations", key="refresh_ai"):
            st.rerun()

        if st.button("� View Data Context", key="show_context"):
            st.session_state.show_ai_context = not st.session_state.get(
                'show_ai_context', False)

    # Context display (conditional)
    if st.session_state.get('show_ai_context', False):
        st.write("")
        st.markdown("### 🔍 Data Context Used by AI")
        with st.expander("View detailed context data", expanded=True):
            try:
                category_data = data[data['category']
                                     == selected_category].copy()
                historical_data = prepare_prophet_data(category_data)
                last_row = historical_data.iloc[-1]
                normal_price = last_row['normal_price']

                model = models[selected_category]
                forecast = generate_forecast(
                    model, historical_data, st.session_state.weekly_prices, 16)

                context = build_context_for_ai(
                    selected_category,
                    data,
                    forecast,
                    st.session_state.weekly_prices,
                    normal_price
                )
                st.code(context, language="text")
            except Exception as e:
                st.error(f"Error generating context display: {str(e)}")

    # Recommendation refresh note
    st.write("")
    st.info("💡 **Tip:** Recommendations are automatically generated based on your current promotional pricing strategy. Change prices in the Forecasting Dashboard tab and return here to see updated recommendations.")

# Main Dashboard


def main():
    # Professional Header with Logo and Branding
    st.markdown("""
    <div class="header-container">
        <div class="logo-container">
            <img src="data:image/png;base64,{}" class="logo-img" alt="ProfitLens Logo">
        </div>
        <div class="brand-container">
            <h1 class="main-title">ProfitLens</h1>
            <p class="tagline">Advanced Sales Analytics & Forecasting Platform</p>
        </div>
    </div>
    """.format(get_base64_image("logo.png")), unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner("🔄 Loading models and data..."):
        models = load_models()
        data = load_data()

    if not models or data is None:
        st.error("❌ Failed to load models or data. Please check your files.")
        return

    # Key metrics in a clean row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Models", f"{len(models)}", help="Prophet forecasting models")
    with col2:
        st.metric("📈 Records", f"{len(data):,}", help="Historical sales data points")  
    with col3:
        st.metric("🏷️ Categories", f"{data['category'].nunique()}", help="Product categories")
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Clean tabs without redundant descriptions
    tab1, tab2, tab3 = st.tabs(["📈 Forecasting", "📊 Analytics", "🤖 AI Insights"])

    with tab1:
        st.markdown('<h2 class="tab-title">📈 Forecasting Dashboard</h2>', unsafe_allow_html=True)
        st.markdown('<p class="tab-description">Interactive forecasting with dynamic promotional price adjustments</p>', unsafe_allow_html=True)
        create_forecasting_dashboard(models, data)

    with tab2:
        st.markdown('<h2 class="tab-title">📊 Sales Analytics</h2>', unsafe_allow_html=True)
        st.markdown('<p class="tab-description">Business insights with Pareto analysis, trends, and promotional impact</p>', unsafe_allow_html=True)
        create_analytics_dashboard(data)

    with tab3:
        st.markdown('<h2 class="tab-title">🤖 AI Recommendations</h2>', unsafe_allow_html=True)
        st.markdown('<p class="tab-description">AI-powered insights based on forecast and promotional strategies</p>', unsafe_allow_html=True)
        create_ai_recommendation_tab(models, data)

    # Clean footer
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; text-align: center; color: #6b7280; font-size: 0.9rem;">
        <strong>📊 Chart Legend:</strong>
        🟢 Historical Data • 🔴 Forecasts • 🔵 Trends & Analysis • 
        Interactive charts support zoom, pan, and hover for detailed insights
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
