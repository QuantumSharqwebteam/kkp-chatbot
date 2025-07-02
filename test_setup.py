#!/usr/bin/env python3
"""
Test script to verify the Dress Sales Chatbot setup
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'pandas',
        'numpy', 
        'streamlit',
        'plotly',
        'sklearn',
        'google.generativeai',
        'joblib'
    ]
    
    print("🔍 Testing package imports...")
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            return False
    
    return True

def test_data_file():
    """Test if the data file exists"""
    print("\n📁 Testing data file...")
    
    try:
        import pandas as pd
        df = pd.read_csv('sales_data_1000_records.csv')
        print(f"✅ Data file loaded successfully")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        return True
    except FileNotFoundError:
        print("❌ sales_data_1000_records.csv not found")
        return False
    except Exception as e:
        print(f"❌ Error loading data file: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config import Config
        print("✅ Configuration loaded successfully")
        api_key = Config.get_gemini_api_key()
        print(f"   - API Key configured: {api_key != 'AIzaSyBwZR5MJm4r5NE1AX5FbzOveBJhuYYbJCQ'}")
        return True
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def test_chatbot_class():
    """Test if the chatbot class can be instantiated"""
    print("\n🤖 Testing chatbot class...")
    
    try:
        from sales_chatbot import DressSalesChatbot
        print("✅ Chatbot class imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing chatbot class: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Dress Sales Chatbot Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_data_file,
        test_config,
        test_chatbot_class
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n🚀 To run the chatbot:")
        print("   streamlit run sales_chatbot.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Ensure sales_data_1000_records.csv is in the current directory")
        print("   3. Configure your Gemini API key in .streamlit/secrets.toml")

if __name__ == "__main__":
    main() 