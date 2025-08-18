"""
Interactive Streamlit Dashboard for Prophet Supermarket Sales Forecasting
Enhanced with Sales Analytics Dashboard
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
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🛒 Supermarket Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chart-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: white;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .legend-box {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions for performance
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
    category_data['ds'] = pd.to_datetime([start_date + timedelta(weeks=w-1) for w in category_data['week']])
    
    # Create target variable
    category_data['y'] = category_data['weekly_sales']
    
    # Calculate discount rate
    category_data['discount_rate'] = (category_data['normal_price'] - category_data['promo_price']) / category_data['normal_price']
    
    # Handle missing values
    category_data = category_data.fillna(0)
    
    return category_data[['ds', 'y', 'promo_active', 'promo_price', 'normal_price', 'discount_rate']]

def generate_forecast(model, historical_data, weekly_promo_prices, forecast_weeks=16):
    """Generate forecast with custom weekly promo_prices"""
    # Get the last date from historical data
    last_date = historical_data['ds'].max()
    
    # Create future dates
    future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=forecast_weeks, freq='W')
    
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
def pareto_analysis(df):
    """Generate Pareto analysis (80/20 rule) for categories"""
    # Calculate total sales by category
    category_sales = df.groupby('category')['weekly_sales'].sum().sort_values(ascending=False)
    
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
        xaxis=dict(tickangle=45),
        yaxis=dict(title="Total Sales", side="left"),
        yaxis2=dict(title="Cumulative Percentage (%)", side="right", overlaying="y"),
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    # Calculate insights
    categories_for_80pct = (cumulative_pct <= 80).sum()
    pct_categories_for_80pct = (categories_for_80pct / len(category_sales)) * 100
    
    return fig, categories_for_80pct, pct_categories_for_80pct, category_sales

def top_categories_chart(df, top_n=10):
    """Create bar chart for top categories by sales"""
    category_sales = df.groupby('category')['weekly_sales'].sum().sort_values(ascending=False).head(top_n)
    
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
    weekly_trend['date'] = pd.to_datetime([start_date + timedelta(weeks=w-1) for w in weekly_trend['week']])
    
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
        default_index = categories.index("Sirup") if "Sirup" in categories else 0
        
        selected_category = st.selectbox(
            "📦 **Product Category**",
            categories,
            index=default_index,
            help="Choose a product category for analysis"
        )
        
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
        st.session_state.weekly_prices = [int(default_promo_price)] * forecast_weeks
    
    # Adjust prices list if forecast weeks changed
    current_length = len(st.session_state.weekly_prices)
    if current_length != forecast_weeks:
        if current_length < forecast_weeks:
            st.session_state.weekly_prices.extend([int(default_promo_price)] * (forecast_weeks - current_length))
        else:
            st.session_state.weekly_prices = st.session_state.weekly_prices[:forecast_weeks]
    
    # Adjust the list size if forecast_weeks changed
    current_length = len(st.session_state.weekly_prices)
    if current_length != forecast_weeks:
        if current_length < forecast_weeks:
            # Extend with default price
            st.session_state.weekly_prices.extend([int(default_promo_price)] * (forecast_weeks - current_length))
        else:
            # Truncate
            st.session_state.weekly_prices = st.session_state.weekly_prices[:forecast_weeks]
    
    # Main panel - Split into two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 Sales Forecast: {selected_category}")
        
        # Weekly promotional pricing controls in an expander
        with st.expander("🎛️ Adjust promo prices for forecast weeks", expanded=False):
            st.markdown("**Set promotional price for each forecast week:**")
            
            # Create columns for better layout
            cols_per_row = 4
            for i in range(0, forecast_weeks, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    week_idx = i + j
                    if week_idx < forecast_weeks:
                        with col:
                            min_price = max(1, int(normal_price * 0.5))
                            max_price = int(normal_price)
                            
                            st.session_state.weekly_prices[week_idx] = st.number_input(
                                f"Week {week_idx + 1}",
                                min_value=min_price,
                                max_value=max_price,
                                value=st.session_state.weekly_prices[week_idx],
                                step=100,
                                key=f"week_{week_idx}",
                                help=f"Week {week_idx + 1} promotional price"
                            )
            
            # Quick actions
            st.markdown("**Quick Actions:**")
            quick_cols = st.columns(4)
            
            with quick_cols[0]:
                if st.button("🔄 Reset to Default", help="Reset all weeks to default promo price"):
                    for i in range(forecast_weeks):
                        st.session_state.weekly_prices[i] = int(default_promo_price)
                    st.rerun()
            
            with quick_cols[1]:
                if st.button("� Apply 10% Discount", help="Apply 10% discount to all weeks"):
                    discount_price = int(normal_price * 0.9)
                    for i in range(forecast_weeks):
                        st.session_state.weekly_prices[i] = max(int(normal_price * 0.5), discount_price)
                    st.rerun()
            
            with quick_cols[2]:
                if st.button("🎯 Apply 20% Discount", help="Apply 20% discount to all weeks"):
                    discount_price = int(normal_price * 0.8)
                    for i in range(forecast_weeks):
                        st.session_state.weekly_prices[i] = max(int(normal_price * 0.5), discount_price)
                    st.rerun()
                    
            with quick_cols[3]:
                avg_discount = (normal_price - np.mean(st.session_state.weekly_prices)) / normal_price
                st.metric("Avg Discount", f"{avg_discount:.1%}")
        
        # Generate forecast with weekly prices
        with st.spinner("Generating forecast..."):
            model = models[selected_category]
            forecast = generate_forecast(model, historical_data, st.session_state.weekly_prices, forecast_weeks)
        
        # Build figure with required styles (historical: green lines, forecast: red lines+markers)
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
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
        total_discount = sum((normal_price - price) / normal_price for price in st.session_state.weekly_prices) / forecast_weeks
        
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
        forecast_display['Date'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
        forecast_display['Forecast'] = forecast_display['yhat'].round(0).astype(int)
        forecast_display['Lower'] = forecast_display['yhat_lower'].round(0).astype(int)
        forecast_display['Upper'] = forecast_display['yhat_upper'].round(0).astype(int)
        
        # Add pricing information
        forecast_display['Promo_Price'] = [f"${price:,}" for price in st.session_state.weekly_prices[:len(forecast_display)]]
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

def create_analytics_dashboard(data):
    """Create the sales analytics dashboard content"""
    st.header("📊 Sales Analytics Dashboard")
    st.markdown("### Business insights and performance analysis")
    st.divider()
    
    # Pareto Analysis
    st.subheader("🎯 Pareto Analysis (80/20 Rule)")
    
    with st.spinner("Generating Pareto analysis..."):
        pareto_fig, categories_80pct, pct_categories_80pct, category_sales = pareto_analysis(data)
    
    st.plotly_chart(pareto_fig, use_container_width=True)
    
    # Insights text
    st.info(f"""
    **📈 Pareto Insights:** 
    • **{categories_80pct}** categories ({pct_categories_80pct:.1f}% of all categories) generate 80% of total sales
    • This follows the classic 80/20 rule, indicating a concentrated sales distribution
    • Focus marketing and inventory efforts on these top-performing categories
    """)
    
    st.divider()
    
    # Two columns for next analyses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Categories by Sales")
        
        with st.spinner("Analyzing top categories..."):
            top_fig, top_contribution = top_categories_chart(data)
        
        st.plotly_chart(top_fig, use_container_width=True)
        
        st.success(f"""
        **� Key Insight:** 
        Top 10 categories contribute **{top_contribution:.1f}%** of total sales
        """)
    
    with col2:
        st.subheader("💰 Promotional Impact Analysis")
        
        with st.spinner("Analyzing promotional impact..."):
            promo_box_fig, promo_bar_fig, avg_promo, avg_non_promo, improvement = promo_impact_analysis(data)
        
        st.plotly_chart(promo_bar_fig, use_container_width=True)
        st.plotly_chart(promo_box_fig, use_container_width=True)
        
        if improvement > 0:
            st.success(f"""
            **📊 Promotional Impact:** 
            Promotions increase average sales by **{improvement:.1f}%**
            (${avg_promo:,.0f} vs ${avg_non_promo:,.0f})
            """)
        else:
            st.warning(f"""
            **📊 Promotional Impact:** 
            Promotions show **{improvement:.1f}%** change in average sales
            """)
    
    st.divider()
    
    # Time Series Trend
    st.subheader("📈 Overall Sales Trend Over Time")
    
    with st.spinner("Analyzing sales trends..."):
        trend_fig, growth_rate = time_series_trend(data)
    
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # Growth insight
    if growth_rate > 0:
        st.success(f"""
        **📈 Growth Insight:** 
        Sales show a **{growth_rate:.1f}%** improvement from first half to second half of the period
        """)
    else:
        st.warning(f"""
        **📉 Trend Alert:** 
        Sales show a **{abs(growth_rate):.1f}%** decline from first half to second half
        """)
    
    # Summary statistics
    st.divider()
    st.subheader("📋 Business Summary")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.metric(
            "Total Categories",
            data['category'].nunique(),
            help="Number of product categories"
        )
    
    with col_s2:
        st.metric(
            "Total Sales",
            f"{data['weekly_sales'].sum():,.0f}",
            help="Total sales across all categories and weeks"
        )
    
    with col_s3:
        st.metric(
            "Avg Weekly Sales",
            f"{data['weekly_sales'].mean():,.0f}",
            help="Average weekly sales per category"
        )
    
    with col_s4:
        promo_weeks = (data['promo_active'] == 1).sum()
        total_weeks = len(data)
        promo_percentage = (promo_weeks / total_weeks) * 100
        st.metric(
            "Promotion Coverage",
            f"{promo_percentage:.1f}%",
            help="Percentage of weeks with active promotions"
        )

# Main Dashboard
def main():
    # Professional Header
    st.markdown('<h1 class="main-header">🛒 Supermarket Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive Business Intelligence & Predictive Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner("🔄 Loading models and data..."):
        models = load_models()
        data = load_data()
    
    if not models or data is None:
        st.error("❌ Failed to load models or data. Please check your files.")
        return
    
    # Success message with metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Models Loaded", f"{len(models)}", help="Prophet forecasting models")
    with col2:
        st.metric("📈 Data Records", f"{len(data):,}", help="Historical sales records")
    with col3:
        st.metric("🏷️ Categories", f"{data['category'].nunique()}", help="Product categories")
    
    st.write("")  # Spacing
    
    # Professional Tabs with clear descriptions
    tab1, tab2 = st.tabs(["� Forecasting Dashboard", "📊 Sales Analytics Dashboard"])
    
    with tab1:
        st.markdown("### 🔮 Sales Forecasting & Promotional Planning")
        st.markdown("*Interactive Prophet-based forecasting with dynamic pricing controls and scenario planning*")
        st.write("")
        create_forecasting_dashboard(models, data)
    
    with tab2:
        st.markdown("### 📊 Business Intelligence & Performance Analytics")
        st.markdown("*Comprehensive sales analytics with Pareto analysis, trend insights, and promotional impact*")
        st.write("")
        create_analytics_dashboard(data)
    
    # Professional Footer
    st.write("")
    st.write("")
    st.markdown("---")
    st.markdown("""
    <div class="legend-box">
        <strong>🎯 Dashboard Legend:</strong><br>
        🟢 <strong>Green Lines:</strong> Historical sales data<br>
        🔴 <strong>Red Lines + Dots:</strong> Prophet forecasts with confidence intervals<br>
        🔵 <strong>Blue Lines:</strong> Cumulative metrics and trend analysis<br>
        📊 <strong>Interactive Charts:</strong> Zoom, pan, and hover for detailed insights
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
