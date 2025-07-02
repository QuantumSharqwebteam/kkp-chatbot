# 🚀 Quick Start Guide

Get your Dress Sales Monitoring Chatbot running in 5 minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection (for Gemini API)

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### 3. Configure API Key
Open `.streamlit/secrets.toml` and replace the placeholder:
```toml
GEMINI_API_KEY = "AIzaSyCYKuwXYcxW8u0y23UgYTKd9dx240bsm6U"
```

### 4. Test Setup (Optional)
```bash
python test_setup.py
```

### 5. Run the Chatbot
```bash
streamlit run sales_chatbot.py
```

The app will open at `http://localhost:8501`

## 🎯 What You Can Do

### Ask Questions
- Click predefined question buttons
- Type custom questions in the text box
- Get AI-powered responses about your sales data

### Predict Sales
- Select dress attributes (quality, weave, composition)
- Set pricing
- Get ML-powered sales predictions

### View Analytics
- Real-time sales metrics
- Interactive charts and graphs
- Performance insights

## 🔧 Troubleshooting

### "API Key Not Configured"
- Check `.streamlit/secrets.toml` file
- Ensure API key is properly set

### "Data File Not Found"
- Ensure `sales_data_1000_records.csv` is in the same folder
- Check file permissions

### "Package Import Error"
- Run `pip install -r requirements.txt`
- Check Python version (3.8+)

## 📞 Need Help?

1. Run the test script: `python test_setup.py`
2. Check the full README.md for detailed instructions
3. Verify all files are in the correct locations

---

**Happy Chatting! 🤖👗** 