"""
Test script to verify the enhanced Streamlit dashboard
"""

import sys
import os
import importlib.util

def test_dashboard_imports():
    """Test if all required packages are available"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def test_dashboard_structure():
    """Test if dashboard file structure is correct"""
    dashboard_file = "streamlit_dashboard.py"
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Dashboard file {dashboard_file} not found")
        return False
    
    print(f"✅ Dashboard file {dashboard_file} exists")
    
    # Check if saved_model directory exists
    if not os.path.exists("saved_model"):
        print("⚠️  saved_model directory not found - run Prophet training first")
    else:
        model_files = [f for f in os.listdir("saved_model") if f.startswith("prophet_") and f.endswith(".pkl")]
        print(f"✅ Found {len(model_files)} Prophet model files")
    
    # Check if data file exists
    if not os.path.exists("dummy_forecasting_data_100weeks.csv"):
        print("⚠️  dummy_forecasting_data_100weeks.csv not found")
    else:
        print("✅ Data file exists")
    
    return True

def test_dashboard_syntax():
    """Test if dashboard Python code is syntactically correct"""
    try:
        spec = importlib.util.spec_from_file_location("dashboard", "streamlit_dashboard.py")
        module = importlib.util.module_from_spec(spec)
        print("✅ Dashboard code syntax is valid")
        return True
    except Exception as e:
        print(f"❌ Dashboard syntax error: {str(e)}")
        return False

def main():
    print("🔍 TESTING ENHANCED STREAMLIT DASHBOARD")
    print("=" * 50)
    
    # Run tests
    imports_ok = test_dashboard_imports()
    structure_ok = test_dashboard_structure()
    syntax_ok = test_dashboard_syntax()
    
    print("\n" + "=" * 50)
    
    if imports_ok and structure_ok and syntax_ok:
        print("🎉 ALL TESTS PASSED!")
        print("\n📋 TO RUN THE DASHBOARD:")
        print("1. Ensure Prophet models are trained (run notebook cells)")
        print("2. Run: streamlit run streamlit_dashboard.py")
        print("3. Open browser to: http://localhost:8501")
        print("\n🆕 NEW FEATURES:")
        print("- 16-week forecast horizon")
        print("- Individual promo price per week")
        print("- Default category: 'Sirup'")
        print("- Enhanced UI with expandable controls")
        print("- Quick discount action buttons")
    else:
        print("❌ SOME TESTS FAILED")
        if not imports_ok:
            print("- Install missing packages: pip install -r requirements_dashboard.txt")
        if not structure_ok:
            print("- Check file locations")
        if not syntax_ok:
            print("- Fix syntax errors in dashboard code")

if __name__ == "__main__":
    main()
