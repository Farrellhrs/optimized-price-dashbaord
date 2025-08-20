# 🚀 Interactive Streamlit Dashboard

A complete interactive dashboard has been created for your Prophet forecasting models!

## 📁 Files Created:
- **`streamlit_dashboard.py`** - Main dashboard application
- **`requirements_dashboard.txt`** - Required Python packages

## 🛠️ Dashboard Features:
- **📦 Category Selection**: Dropdown to choose from 49 product categories
- **💰 Interactive Pricing**: Slider to adjust promotional prices in real-time
- **📈 Visual Forecasting**: Interactive charts with historical vs forecast data
- **📊 Detailed Tables**: 12-week forecast data with confidence intervals
- **🎯 Key Metrics**: Performance indicators and comparisons

## 🏃‍♂️ How to Run:

1. **Install Dependencies** (run in terminal):
   ```bash
   pip install -r requirements_dashboard.txt
   ```

2. **Launch Dashboard**:
   ```bash
   streamlit run streamlit_dashboard.py
   ```

3. **Access Dashboard**: Open your browser to `http://localhost:8501`

## 🎯 Dashboard Controls:
- **Green Line**: Historical sales data
- **Red Line**: Prophet forecasts
- **Pricing Slider**: Adjust promo_price to see real-time forecast updates
- **Confidence Intervals**: Shaded areas show prediction uncertainty

The dashboard automatically loads all your trained Prophet models and provides an intuitive interface for business users to explore forecasts and pricing scenarios!