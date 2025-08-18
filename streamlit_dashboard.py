"""
Interactive Streamlit Dashboard for Prophet Supermarket Sales Forecasting
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
    page_title="🛒 Supermarket Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Main Dashboard
def main():
    # Header
    st.title("🛒 Supermarket Sales Forecasting Dashboard")
    st.markdown("### Interactive Prophet-based forecasting with promotional pricing controls")
    st.divider()
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        models = load_models()
        data = load_data()
    
    if not models or data is None:
        st.error("Failed to load models or data. Please check your files.")
        return
    
    # Success message
    st.success(f"✅ Loaded {len(models)} Prophet models for forecasting")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    st.sidebar.divider()
    
    # Category selection - Default to "Sirup"
    categories = sorted(list(models.keys()))
    default_index = 0
    if "Sirup" in categories:
        default_index = categories.index("Sirup")
    
    selected_category = st.sidebar.selectbox(
        "📦 Select Product Category:",
        categories,
        index=default_index,
        help="Choose a product category to view forecasts"
    )
    
    # Get historical data for selected category
    category_data = data[data['category'] == selected_category].copy()
    historical_data = prepare_prophet_data(category_data)
    
    # Get pricing information
    last_row = historical_data.iloc[-1]
    normal_price = last_row['normal_price']
    default_promo_price = last_row['promo_price']
    
    # Promotional pricing controls
    st.sidebar.subheader("💰 Promotional Pricing")
    
    # Display current pricing info
    st.sidebar.info(f"**Normal Price:** ${normal_price:,.0f}")
    st.sidebar.info(f"**Default Promo Price:** ${default_promo_price:,.0f}")
    
    # Forecast weeks control (moved up for better UX)
    forecast_weeks = st.sidebar.slider(
        "📅 Forecast Horizon (weeks):",
        min_value=8,
        max_value=24,
        value=16,
        step=4,
        help="Number of weeks to forecast ahead"
    )
    
    # Initialize session state for weekly prices
    if 'weekly_prices' not in st.session_state:
        st.session_state.weekly_prices = [int(default_promo_price)] * forecast_weeks
    
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
        
        # Create interactive plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#2E8B57', width=2),
            hovertemplate='<b>Historical</b><br>' +
                         'Date: %{x}<br>' +
                         'Sales: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#DC143C', width=2, dash='dash'),
            hovertemplate='<b>Forecast</b><br>' +
                         'Date: %{x}<br>' +
                         'Sales: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(color='rgba(220, 20, 60, 0)', width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(220, 20, 60, 0.2)',
            line=dict(color='rgba(220, 20, 60, 0)', width=0),
            name='Confidence Interval',
            hovertemplate='<b>Confidence Interval</b><br>' +
                         'Date: %{x}<br>' +
                         'Upper: ' + str(forecast['yhat_upper'].iloc[0] if len(forecast) > 0 else 0) + '<br>' +
                         'Lower: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout
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
                x=1
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
    
    # Footer
    st.divider()
    st.markdown("""
    **Enhanced Dashboard Features:**
    - 🟢 **Green Line**: Historical actual sales data
    - 🔴 **Red Line**: Prophet model forecasts (16-week horizon)
    - 🎛️ **Weekly Pricing Controls**: Set individual promo prices for each forecast week
    - 📊 **Confidence Intervals**: Shaded area shows forecast uncertainty
    - 🎯 **Quick Actions**: Apply common discount scenarios instantly
    - 📈 **Real-time Updates**: Forecasts update automatically as you adjust prices
    """)

if __name__ == "__main__":
    main()
