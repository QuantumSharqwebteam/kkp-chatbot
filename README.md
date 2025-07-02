# 👗 Dress Sales Monitoring Chatbot

A comprehensive **admin-only** dress sales monitoring and prediction system powered by Machine Learning and Gemini AI. This chatbot provides detailed analytics, sales predictions, and intelligent responses to dress sales queries.

## 🔐 Admin Access Only

This system is restricted to authorized sales company administrators. Default credentials:
- **Username:** `admin`
- **Password:** `admin123`

## 🚀 Key Features

### 📊 **Real-time Analytics**
- Total revenue and order tracking
- Agent performance analysis
- Customer purchase patterns
- Quality and composition breakdowns
- Order status monitoring

### 🤖 **Intelligent Q&A System**
- **Admin-specific questions** answered instantly without API calls
- Gemini AI integration for complex queries
- Fallback responses when API is unavailable
- Predefined common questions for quick access

### 🔮 **Advanced Predictions**
- Sales quantity predictions based on dress attributes
- Future revenue forecasting (3-month projections)
- Trend analysis and growth rate calculations
- Confidence level assessments

### 📈 **Interactive Visualizations**
- Real-time charts and graphs
- Sales trends over time
- Agent performance comparisons
- Quality and status distributions

## 🎯 Admin Questions (Instant Answers)

The system provides precise, data-driven answers to these common admin questions **without requiring API calls**:

### 💰 **Revenue Analysis**
- What is the total revenue from all orders?
- What is the average order value?
- How much revenue did we generate this month?
- What is the trend in sales revenue over the past 6 months?

### 📦 **Order Analysis**
- What is the total quantity sold this year?
- How many dress orders are confirmed?
- How many pending or cancelled orders do we have?
- What is the status breakdown of all orders?

### 💎 **Product & Pricing**
- What is the average rate per meter?
- How many premium quality orders have been made?
- Which composition material sells best?
- What is the average quantity per order?

### 👥 **Performance Metrics**
- Show me top 5 performing agents
- What is the conversion rate of orders?
- Predict sales for premium cotton dresses

## 🛠️ Technical Architecture

### **Data Processing**
- Automated data cleaning and preprocessing
- Feature engineering for ML models
- Real-time analytics calculations

### **Machine Learning**
- Random Forest regression for sales prediction
- Label encoding for categorical variables
- Model persistence and loading

### **AI Integration**
- Gemini API for natural language processing
- Intelligent fallback responses
- Context-aware question handling

### **Web Interface**
- Streamlit-based dashboard
- Interactive visualizations with Plotly
- Session management and chat history

## 📋 Prerequisites

- Python 3.8+
- Gemini API key (optional - system works without it)
- Required Python packages (see `requirements.txt`)

## 🚀 Quick Start

1. **Clone and Setup:**
   ```bash
   git clone <repository-url>
   cd dress-sales-chatbot
   pip install -r requirements.txt
   ```

2. **Configure API (Optional):**
   - Edit `config.py` and add your Gemini API key
   - Or leave as default for fallback-only mode

3. **Run the Application:**
   ```bash
   streamlit run sales_chatbot.py
   ```

4. **Login as Admin:**
   - Username: `admin`
   - Password: `admin123`

## 📁 Project Structure

```
dress-sales-chatbot/
├── sales_chatbot.py          # Main application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── sales_data_1000_records.csv  # Sample dataset
├── README.md                 # This file
├── QUICK_START.md           # Quick start guide
├── run_chatbot.bat          # Windows run script
├── run_chatbot.sh           # Linux/Mac run script
└── test_setup.py            # Setup testing script
```

## 🔧 Configuration

### **API Settings**
- `GEMINI_API_KEY`: Your Gemini API key
- `API_ENDPOINT`: Gemini API endpoint

### **Model Parameters**
- `n_estimators`: Number of trees in Random Forest
- `test_size`: Test split ratio
- `random_state`: Random seed for reproducibility

### **UI Settings**
- `PAGE_TITLE`: Application title
- `CHART_HEIGHT`: Chart display height
- `MAX_CHAT_HISTORY`: Maximum chat history entries

## 📊 Dataset Fields

The system works with dress sales data containing:
- `date`: Order date
- `quality`: Dress quality (premium, standard)
- `weave`: Weave type (linen, spandex)
- `quantity`: Order quantity
- `composition`: Material composition
- `status`: Order status
- `rate`: Price per unit
- `agentName`: Sales agent
- `customerName`: Customer name

## 🎯 Use Cases

### **For Sales Managers**
- Monitor team performance
- Track revenue trends
- Analyze customer preferences
- Forecast future sales

### **For Business Analysts**
- Generate detailed reports
- Identify growth opportunities
- Analyze market trends
- Optimize pricing strategies

### **For Operations Teams**
- Track order status
- Monitor inventory needs
- Analyze production efficiency
- Plan resource allocation

## 🔒 Security Features

- **Admin-only access** with authentication
- **Secure API key handling**
- **Session management**
- **Input validation**

## 🚀 Performance Features

- **Lazy loading** for faster startup
- **Cached analytics** for quick responses
- **Optimized ML models** for accurate predictions
- **Responsive UI** for better user experience

## 📞 Support

For technical support or questions:
1. Check the `QUICK_START.md` guide
2. Review the configuration in `config.py`
3. Test the setup with `test_setup.py`

## 🔄 Updates & Maintenance

- Regular model retraining with new data
- API key rotation for security
- Performance monitoring and optimization
- Feature updates based on admin feedback

---

**🤖 Powered by Machine Learning & Gemini AI | 👗 Dress Sales Monitoring System** 