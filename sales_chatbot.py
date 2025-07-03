import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Import configuration
from config import Config

class DressSalesChatbot:
    def __init__(self):
        self.df = None
        self.model = None
        self.label_encoders = {}
        self._initialized = False
        
    def initialize(self):
        """Initialize the chatbot (called lazily)"""
        if not self._initialized:
            # Configure Gemini API lazily
            api_key = Config.get_gemini_api_key()
            genai.configure(api_key=api_key)
            
            self.load_data()
            self.train_model()
            self._initialized = True
        
    def load_data(self):
        """Load and preprocess the sales data"""
        try:
            self.df = pd.read_csv(Config.DATA_FILE)
            
            # Convert date column
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Clean quantity column (remove 'm' and convert to numeric)
            self.df['quantity'] = self.df['quantity'].astype(str).str.replace('m', '').astype(float)
            
            # Clean quality column
            self.df['quality'] = self.df['quality'].str.strip().str.lower()
            self.df['quality'] = self.df['quality'].replace({
                'primium': 'premium',
                'stand': 'standard',
                'premium ': 'premium',
                'stand ': 'standard'
            })
            
            # Clean weave column
            self.df['weave'] = self.df['weave'].str.strip().str.lower()
            self.df['weave'] = self.df['weave'].replace({
                'spandex ': 'spandex',
                'linen': 'linen',
                'spandex': 'spandex'
            })
            
            # Clean composition column
            self.df['composition'] = self.df['composition'].str.strip()
            
            # Add derived features
            self.df['month'] = self.df['date'].dt.month
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            self.df['total_value'] = self.df['quantity'] * self.df['rate']
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def train_model(self):
        """Train the sales prediction model"""
        try:
            # Prepare features for ML
            features = ['quality', 'weave', 'composition', 'month', 'day_of_week', 'rate']
            X = self.df[features].copy()
            y = self.df['quantity']
            
            # Encode categorical variables
            for col in ['quality', 'weave', 'composition']:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=Config.MODEL_PARAMS['test_size'], 
                random_state=Config.MODEL_PARAMS['random_state']
            )
            
            self.model = RandomForestRegressor(
                n_estimators=Config.MODEL_PARAMS['n_estimators'], 
                random_state=Config.MODEL_PARAMS['random_state']
            )
            self.model.fit(X_train, y_train)
            
            # Save model
            joblib.dump(self.model, Config.MODEL_FILE)
            joblib.dump(self.label_encoders, Config.ENCODERS_FILE)
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def predict_sales(self, quality, weave, composition, rate, month=None, day_of_week=None):
        """Predict sales quantity"""
        try:
            self.initialize()  # Ensure initialization
            
            if month is None:
                month = datetime.now().month
            if day_of_week is None:
                day_of_week = datetime.now().weekday()
            
            # Prepare input
            input_data = pd.DataFrame({
                'quality': [quality],
                'weave': [weave],
                'composition': [composition],
                'month': [month],
                'day_of_week': [day_of_week],
                'rate': [rate]
            })
            
            # Encode categorical variables
            for col in ['quality', 'weave', 'composition']:
                if col in self.label_encoders:
                    input_data[col] = self.label_encoders[col].transform(input_data[col])
            
            # Predict
            prediction = self.model.predict(input_data)[0]
            return max(0, prediction)  # Ensure non-negative
            
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")
            return 0
    
    def get_sales_analytics(self):
        """Get comprehensive sales analytics"""
        self.initialize()  # Ensure initialization
        
        analytics = {}
        
        # Basic statistics
        analytics['total_sales'] = len(self.df)
        analytics['total_revenue'] = self.df['total_value'].sum()
        analytics['avg_order_value'] = self.df['total_value'].mean()
        analytics['total_quantity'] = self.df['quantity'].sum()
        
        # Status breakdown
        analytics['status_breakdown'] = self.df['status'].value_counts().to_dict()
        
        # Quality performance
        analytics['quality_performance'] = self.df.groupby('quality')['total_value'].sum().to_dict()
        
        # Top agents
        analytics['top_agents'] = self.df.groupby('agentName')['total_value'].sum().sort_values(ascending=False).head(5).to_dict()
        
        # Top customers
        analytics['top_customers'] = self.df.groupby('customerName')['total_value'].sum().sort_values(ascending=False).head(5).to_dict()
        
        # Monthly trends
        analytics['monthly_trends'] = self.df.groupby(self.df['date'].dt.to_period('M'))['total_value'].sum().to_dict()
        
        return analytics
    
    def get_predefined_questions(self):
        """Get list of predefined questions"""
        return [
            "What is the total revenue from all orders?",
            "What is the average order value?",
            "How much revenue did we generate this month?",
            "What is the trend in sales revenue over the past 6 months?",
            "What is the total quantity sold this year?",
            "How many dress orders are confirmed?",
            "How many pending or cancelled orders do we have?",
            "What is the status breakdown of all orders?",
            "What is the average rate per meter?",
            "How many premium quality orders have been made?",
            "Predict sales for premium cotton dresses",
            "Show me top 5 performing agents",
            "What is the conversion rate of orders?",
            "Which composition material sells best?",
            "What is the average quantity per order?"
        ]
    
    def get_admin_question_answer(self, question):
        """Provide precise answers for admin questions without API calls"""
        self.initialize()  # Ensure initialization
        
        analytics = self.get_sales_analytics()
        question_lower = question.lower()
        
        # Revenue Analysis
        if "total revenue" in question_lower and "all orders" in question_lower:
            return f"**Total Revenue from All Orders:** ₹{analytics['total_revenue']:,.2f}\n\nThis represents the cumulative revenue from all {analytics['total_sales']} orders in our system."
        
        elif "average order value" in question_lower:
            return f"**Average Order Value:** ₹{analytics['avg_order_value']:,.2f}\n\nThis is calculated by dividing total revenue (₹{analytics['total_revenue']:,.2f}) by total orders ({analytics['total_sales']:,})."
        
        elif "revenue" in question_lower and "this month" in question_lower:
            current_month = datetime.now().month
            current_year = datetime.now().year
            month_data = self.df[(self.df['date'].dt.month == current_month) & (self.df['date'].dt.year == current_year)]
            month_revenue = month_data['total_value'].sum()
            month_orders = len(month_data)
            return f"**Revenue Generated This Month ({current_month}/{current_year}):** ₹{month_revenue:,.2f}\n\nFrom {month_orders} orders this month."
        
        elif "trend" in question_lower and "sales revenue" in question_lower and "6 months" in question_lower:
            # Get last 6 months data
            six_months_ago = datetime.now() - timedelta(days=180)
            recent_data = self.df[self.df['date'] >= six_months_ago]
            monthly_trend = recent_data.groupby(recent_data['date'].dt.to_period('M'))['total_value'].sum()
            
            if len(monthly_trend) > 0:
                trend_text = "**Sales Revenue Trend (Last 6 Months):**\n\n"
                for month, revenue in monthly_trend.items():
                    trend_text += f"• {month}: ₹{revenue:,.2f}\n"
                
                # Calculate trend direction
                if len(monthly_trend) >= 2:
                    first_month = monthly_trend.values[0]
                    last_month = monthly_trend.values[-1]
                    if last_month > first_month:
                        trend_text += f"\n📈 **Trend:** Revenue is **increasing** (₹{last_month - first_month:+,.2f} change)"
                    else:
                        trend_text += f"\n📉 **Trend:** Revenue is **decreasing** (₹{last_month - first_month:+,.2f} change)"
                
                return trend_text
            else:
                return "**Sales Revenue Trend:** No data available for the last 6 months."
        
        elif "total quantity" in question_lower and "this year" in question_lower:
            current_year = datetime.now().year
            year_data = self.df[self.df['date'].dt.year == current_year]
            year_quantity = year_data['quantity'].sum()
            year_orders = len(year_data)
            return f"**Total Quantity Sold This Year ({current_year}):** {year_quantity:,.0f} units\n\nFrom {year_orders} orders this year."
        
        # Order Analysis
        elif "dress orders" in question_lower and "confirmed" in question_lower:
            confirmed_orders = analytics['status_breakdown'].get('Confirmed', 0)
            total_orders = analytics['total_sales']
            confirmation_rate = (confirmed_orders / total_orders * 100) if total_orders > 0 else 0
            return f"**Confirmed Dress Orders:** {confirmed_orders:,}\n\nOut of {total_orders:,} total orders ({confirmation_rate:.1f}% confirmation rate)."
        
        elif "pending" in question_lower and "cancelled" in question_lower:
            pending_orders = analytics['status_breakdown'].get('Pending', 0)
            cancelled_orders = analytics['status_breakdown'].get('Cancelled', 0)
            total_pending_cancelled = pending_orders + cancelled_orders
            return f"**Pending or Cancelled Orders:** {total_pending_cancelled:,}\n\n• Pending: {pending_orders:,} orders\n• Cancelled: {cancelled_orders:,} orders"
        
        elif "status breakdown" in question_lower and "all orders" in question_lower:
            breakdown_text = "**Status Breakdown of All Orders:**\n\n"
            total_orders = analytics['total_sales']
            
            for status, count in analytics['status_breakdown'].items():
                percentage = (count / total_orders * 100) if total_orders > 0 else 0
                breakdown_text += f"• **{status}:** {count:,} orders ({percentage:.1f}%)\n"
            
            return breakdown_text
        
        elif "average rate" in question_lower and "per meter" in question_lower:
            # Calculate average rate per meter
            avg_rate = self.df['rate'].mean()
            return f"**Average Rate Per Meter:** ₹{avg_rate:.2f}\n\nThis is the average price per unit across all dress orders."
        
        elif "premium quality" in question_lower and "orders" in question_lower:
            premium_orders = len(self.df[self.df['quality'] == 'premium'])
            premium_revenue = self.df[self.df['quality'] == 'premium']['total_value'].sum()
            total_orders = analytics['total_sales']
            premium_percentage = (premium_orders / total_orders * 100) if total_orders > 0 else 0
            
            return f"**Premium Quality Orders:** {premium_orders:,}\n\n• Revenue from Premium: ₹{premium_revenue:,.2f}\n• Percentage of Total: {premium_percentage:.1f}%"
        
        # Most/Least sold weave, quality, composition, customer, agent
        elif ("most sold" in question_lower or "top selling" in question_lower or "highest selling" in question_lower) and "weave" in question_lower:
            weave_sales = self.df.groupby('weave')['quantity'].sum().sort_values(ascending=False)
            top_weave = weave_sales.index[0]
            top_qty = weave_sales.iloc[0]
            return f"**Most Sold Weave Type:** {top_weave}\n\n• Quantity Sold: {top_qty:,.0f} units"
        elif ("least sold" in question_lower or "lowest selling" in question_lower) and "weave" in question_lower:
            weave_sales = self.df.groupby('weave')['quantity'].sum().sort_values(ascending=True)
            least_weave = weave_sales.index[0]
            least_qty = weave_sales.iloc[0]
            return f"**Least Sold Weave Type:** {least_weave}\n\n• Quantity Sold: {least_qty:,.0f} units"
        elif ("most sold" in question_lower or "top selling" in question_lower or "highest selling" in question_lower) and "quality" in question_lower:
            quality_sales = self.df.groupby('quality')['quantity'].sum().sort_values(ascending=False)
            top_quality = quality_sales.index[0]
            top_qty = quality_sales.iloc[0]
            return f"**Most Sold Quality:** {top_quality}\n\n• Quantity Sold: {top_qty:,.0f} units"
        elif ("least sold" in question_lower or "lowest selling" in question_lower) and "quality" in question_lower:
            quality_sales = self.df.groupby('quality')['quantity'].sum().sort_values(ascending=True)
            least_quality = quality_sales.index[0]
            least_qty = quality_sales.iloc[0]
            return f"**Least Sold Quality:** {least_quality}\n\n• Quantity Sold: {least_qty:,.0f} units"
        elif ("most sold" in question_lower or "top selling" in question_lower or "highest selling" in question_lower) and "composition" in question_lower:
            comp_sales = self.df.groupby('composition')['quantity'].sum().sort_values(ascending=False)
            top_comp = comp_sales.index[0]
            top_qty = comp_sales.iloc[0]
            return f"**Most Sold Composition:** {top_comp}\n\n• Quantity Sold: {top_qty:,.0f} units"
        elif ("least sold" in question_lower or "lowest selling" in question_lower) and "composition" in question_lower:
            comp_sales = self.df.groupby('composition')['quantity'].sum().sort_values(ascending=True)
            least_comp = comp_sales.index[0]
            least_qty = comp_sales.iloc[0]
            return f"**Least Sold Composition:** {least_comp}\n\n• Quantity Sold: {least_qty:,.0f} units"
        elif ("most sold" in question_lower or "top selling" in question_lower or "highest selling" in question_lower) and ("customer" in question_lower or "customer name" in question_lower):
            cust_sales = self.df.groupby('customerName')['quantity'].sum().sort_values(ascending=False)
            top_cust = cust_sales.index[0]
            top_qty = cust_sales.iloc[0]
            return f"**Customer with Most Purchases:** {top_cust}\n\n• Quantity Purchased: {top_qty:,.0f} units"
        elif ("least sold" in question_lower or "lowest selling" in question_lower) and ("customer" in question_lower or "customer name" in question_lower):
            cust_sales = self.df.groupby('customerName')['quantity'].sum().sort_values(ascending=True)
            least_cust = cust_sales.index[0]
            least_qty = cust_sales.iloc[0]
            return f"**Customer with Least Purchases:** {least_cust}\n\n• Quantity Purchased: {least_qty:,.0f} units"
        elif ("most sold" in question_lower or "top selling" in question_lower or "highest selling" in question_lower) and ("agent" in question_lower or "agent name" in question_lower):
            agent_sales = self.df.groupby('agentName')['quantity'].sum().sort_values(ascending=False)
            top_agent = agent_sales.index[0]
            top_qty = agent_sales.iloc[0]
            return f"**Agent with Most Sales:** {top_agent}\n\n• Quantity Sold: {top_qty:,.0f} units"
        elif ("least sold" in question_lower or "lowest selling" in question_lower) and ("agent" in question_lower or "agent name" in question_lower):
            agent_sales = self.df.groupby('agentName')['quantity'].sum().sort_values(ascending=True)
            least_agent = agent_sales.index[0]
            least_qty = agent_sales.iloc[0]
            return f"**Agent with Least Sales:** {least_agent}\n\n• Quantity Sold: {least_qty:,.0f} units"
        
        # Enhanced predictions
        elif "predict" in question_lower and "premium cotton" in question_lower:
            # Enhanced prediction with detailed analysis
            prediction = self.predict_sales('premium', 'linen', 'Cotton 100%', 400)
            avg_premium_rate = self.df[self.df['quality'] == 'premium']['rate'].mean()
            estimated_revenue = prediction * avg_premium_rate
            
            # Get historical premium cotton data
            premium_cotton_data = self.df[
                (self.df['quality'] == 'premium') & 
                (self.df['composition'].str.contains('Cotton', case=False))
            ]
            
            historical_avg = premium_cotton_data['quantity'].mean() if len(premium_cotton_data) > 0 else 0
            
            return f"**Sales Prediction for Premium Cotton Dresses:**\n\n" \
                   f"• **Predicted Quantity:** {prediction:.0f} units\n" \
                   f"• **Estimated Revenue:** ₹{estimated_revenue:,.2f}\n" \
                   f"• **Historical Average:** {historical_avg:.0f} units\n" \
                   f"• **Confidence Level:** {'High' if abs(prediction - historical_avg) < historical_avg * 0.3 else 'Medium'}"
        
        # Additional admin insights
        elif "top" in question_lower and "performing agents" in question_lower:
            top_agents_text = "**Top 5 Performing Agents:**\n\n"
            for i, (agent, revenue) in enumerate(analytics['top_agents'].items(), 1):
                top_agents_text += f"{i}. **{agent}:** ₹{revenue:,.2f}\n"
            return top_agents_text
        
        elif "conversion rate" in question_lower:
            confirmed = analytics['status_breakdown'].get('Confirmed', 0)
            total = analytics['total_sales']
            conversion_rate = (confirmed / total * 100) if total > 0 else 0
            return f"**Order Conversion Rate:** {conversion_rate:.1f}%\n\n" \
                   f"• Confirmed Orders: {confirmed:,}\n" \
                   f"• Total Orders: {total:,}\n" \
                   f"• Pending: {analytics['status_breakdown'].get('Pending', 0):,}\n" \
                   f"• Cancelled: {analytics['status_breakdown'].get('Cancelled', 0):,}"
        
        elif "composition" in question_lower and "best" in question_lower:
            composition_performance = self.df.groupby('composition')['total_value'].sum().sort_values(ascending=False)
            best_composition = composition_performance.index[0]
            best_revenue = composition_performance.iloc[0]
            
            return f"**Best-Selling Composition Material:** {best_composition}\n\n" \
                   f"• Revenue: ₹{best_revenue:,.2f}\n" \
                   f"• Orders: {len(self.df[self.df['composition'] == best_composition]):,}"
        
        elif "average quantity" in question_lower and "per order" in question_lower:
            avg_quantity = analytics['total_quantity'] / analytics['total_sales']
            return f"**Average Quantity Per Order:** {avg_quantity:.1f} units\n\n" \
                   f"• Total Quantity: {analytics['total_quantity']:,.0f} units\n" \
                   f"• Total Orders: {analytics['total_sales']:,}"
        
        # If no specific pattern matches, return None to use API or fallback
        return None
    
    def predict_future_sales(self, months_ahead=3):
        """Enhanced future sales prediction based on historical data"""
        self.initialize()
        
        # Get historical monthly data
        monthly_data = self.df.groupby(self.df['date'].dt.to_period('M')).agg({
            'total_value': 'sum',
            'quantity': 'sum',
            'rate': 'mean'
        }).reset_index()
        
        if len(monthly_data) < 2:
            return "Insufficient historical data for prediction."
        
        # Calculate growth rate
        recent_months = monthly_data.tail(3)
        if len(recent_months) >= 2:
            growth_rate = (recent_months['total_value'].iloc[-1] - recent_months['total_value'].iloc[0]) / recent_months['total_value'].iloc[0]
        else:
            growth_rate = 0
        
        # Predict future months
        current_month = datetime.now()
        predictions = []
        
        for i in range(1, months_ahead + 1):
            future_month = current_month + timedelta(days=30*i)
            predicted_revenue = monthly_data['total_value'].mean() * (1 + growth_rate * i)
            predicted_quantity = monthly_data['quantity'].mean() * (1 + growth_rate * i)
            
            predictions.append({
                'month': future_month.strftime('%B %Y'),
                'revenue': predicted_revenue,
                'quantity': predicted_quantity
            })
        
        return predictions
    
    def test_gemini_api(self):
        """Test if Gemini API is working properly"""
        try:
            api_key = Config.get_gemini_api_key()
            if api_key == "AIzaSyBwZR5MJm4r5NE1AX5FbzOveBJhuYYbJCQ":
                return False, "API key not configured"
            
            # Test with a simple prompt
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content("Hello, this is a test. Please respond with 'API is working' if you can see this message.")
            
            if response and response.text:
                return True, "API is working"
            else:
                return False, "Empty response received"
                
        except Exception as e:
            return False, f"API Error: {str(e)}"
    
    def extract_date_from_question(self, question):
        """Extract a date in YYYY-MM-DD format from the question, or return None."""
        import re
        match = re.search(r"(\d{4}-\d{2}-\d{2})", question)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d")
            except Exception:
                return None
        return None

    def find_nearest_date(self, target_date):
        """Find the nearest available date in the dataset to the target_date."""
        available_dates = self.df['date'].dt.date.unique()
        if not available_dates.size:
            return None
        nearest = min(available_dates, key=lambda d: abs(d - target_date.date()))
        return nearest

    def get_most_relevant_data(self, question):
        """Return the most relevant data from the dataset based on the question if exact data is not available."""
        # Example: If question mentions a product, quality, or agent, return top results for that
        q = question.lower()
        if "premium" in q:
            df = self.df[self.df['quality'] == 'premium']
            if not df.empty:
                return df
        if "agent" in q:
            top_agent = self.df.groupby('agentName')['total_value'].sum().idxmax()
            df = self.df[self.df['agentName'] == top_agent]
            if not df.empty:
                return df
        # Fallback: return most recent day's data
        latest_date = self.df['date'].max().date()
        return self.df[self.df['date'].dt.date == latest_date]

    def extract_keywords(self, question):
        """Extracts relevant keywords for routing the question to the right data logic."""
        keywords = []
        q = question.lower()
        # Add more as needed
        if any(k in q for k in ["trend", "sales revenue", "over the past", "last", "recent"]):
            keywords.append("trend")
        if any(k in q for k in ["predict", "forecast", "future sales"]):
            keywords.append("predict")
        if any(k in q for k in ["most sold", "top selling", "highest selling"]):
            keywords.append("most_sold")
        if any(k in q for k in ["least sold", "lowest selling"]):
            keywords.append("least_sold")
        if "weave" in q:
            keywords.append("weave")
        if "quality" in q:
            keywords.append("quality")
        if "composition" in q:
            keywords.append("composition")
        if "customer" in q:
            keywords.append("customer")
        if "agent" in q:
            keywords.append("agent")
        if "date" in q or any(char.isdigit() for char in q):
            keywords.append("date")
        if "revenue" in q:
            keywords.append("revenue")
        if "quantity" in q:
            keywords.append("quantity")
        if "order" in q:
            keywords.append("order")
        return keywords

    def answer_question(self, question):
        """Answer user questions using data and always enhance with Gemini API if available."""
        try:
            self.initialize()  # Ensure initialization
            keywords = self.extract_keywords(question)
            data_answer = None

            # 1. Date-specific queries
            if "date" in keywords:
                date = self.extract_date_from_question(question)
                if date:
                    day_data = self.df[self.df['date'].dt.date == date.date()]
                    if not day_data.empty:
                        summary = f"\nHere is the exact sales data for {date.strftime('%Y-%m-%d')}:\n"
                        for idx, row in day_data.iterrows():
                            summary += (
                                f"- Agent: {row['agentName']}, Customer: {row['customerName']}, "
                                f"Quality: {row['quality']}, Weave: {row['weave']}, Quantity: {row['quantity']}, "
                                f"Composition: {row['composition']}, Status: {row['status']}, Rate: {row['rate']}, Revenue: ₹{row['total_value']:.2f}\n"
                            )
                        data_answer = summary
                    else:
                        nearest = self.find_nearest_date(date)
                        if nearest:
                            nearest_data = self.df[self.df['date'].dt.date == nearest]
                            summary = f"\nNo sales data found for {date.strftime('%Y-%m-%d')}. Showing nearest available data for {nearest.strftime('%Y-%m-%d')}:\n"
                            for idx, row in nearest_data.iterrows():
                                summary += (
                                    f"- Agent: {row['agentName']}, Customer: {row['customerName']}, "
                                    f"Quality: {row['quality']}, Weave: {row['weave']}, Quantity: {row['quantity']}, "
                                    f"Composition: {row['composition']}, Status: {row['status']}, Rate: {row['rate']}, Revenue: ₹{row['total_value']:.2f}\n"
                                )
                            data_answer = summary
                        else:
                            data_answer = f"Sorry, no sales data is available for the requested or any nearby date."

            # 2. Trend, prediction, most/least sold, and admin questions
            if data_answer is None:
                admin_answer = self.get_admin_question_answer(question)
                if admin_answer:
                    data_answer = admin_answer

            # 3. Fallback: most relevant data
            if data_answer is None:
                relevant_data = self.get_most_relevant_data(question)
                if relevant_data is not None and not relevant_data.empty:
                    summary = "Here is the most relevant sales data I could find for your query:\n"
                    for idx, row in relevant_data.iterrows():
                        summary += (
                            f"- Date: {row['date'].strftime('%Y-%m-%d')}, Agent: {row['agentName']}, Customer: {row['customerName']}, "
                            f"Quality: {row['quality']}, Weave: {row['weave']}, Quantity: {row['quantity']}, "
                            f"Composition: {row['composition']}, Status: {row['status']}, Rate: {row['rate']}, Revenue: ₹{row['total_value']:.2f}\n"
                        )
                    data_answer = summary

            # 4. If still nothing, generic fallback
            if data_answer is None:
                data_answer = "I apologize, but I could not find an exact answer in the sales data. Please try rephrasing your question or ask for a different sales insight."

            # 5. Always enhance with API if available
            enhanced_answer = ""
            if Config.validate_api_key():
                analytics = self.get_sales_analytics()
                context = f"Data-driven answer:\n{data_answer}\n\nSales Data Context:\n- Total Sales: {analytics['total_sales']} orders\n- Total Revenue: ₹{analytics['total_revenue']:,.2f}\n- Average Order Value: ₹{analytics['avg_order_value']:,.2f}\n- Total Quantity Sold: {analytics['total_quantity']:,.0f} units\n\nStatus Breakdown: {analytics['status_breakdown']}\nTop Agents: {list(analytics['top_agents'].keys())[:3]}\nTop Customers: {list(analytics['top_customers'].keys())[:3]}\n\nQuestion: {question}\nPlease provide a professional, conversational, and insightful answer based on this data."
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    response = model.generate_content(context)
                    if response and response.text:
                        enhanced_answer = response.text
                except Exception:
                    pass
            # 6. Return both answers (data first, then enhancement)
            if enhanced_answer:
                return f"{data_answer}\n\n---\n\n{enhanced_answer}"
            else:
                return data_answer
        except Exception as e:
            st.error(f"General Error: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question. Please try rephrasing or ask a different question about dress sales data."
    
    def create_visualizations(self):
        """Create interactive visualizations"""
        self.initialize()  # Ensure initialization
        
        # 1. Sales by Status
        fig_status = px.pie(
            values=list(self.df['status'].value_counts().values),
            names=list(self.df['status'].value_counts().index),
            title="Sales Distribution by Status",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_status.update_layout(height=Config.CHART_HEIGHT)
        
        # 2. Revenue by Quality
        quality_revenue = self.df.groupby('quality')['total_value'].sum().sort_values(ascending=True)
        fig_quality = px.bar(
            x=quality_revenue.values,
            y=quality_revenue.index,
            orientation='h',
            title="Revenue by Dress Quality",
            color=quality_revenue.values,
            color_continuous_scale='viridis'
        )
        fig_quality.update_layout(height=Config.CHART_HEIGHT)
        
        # 3. Monthly Sales Trend
        monthly_sales = self.df.groupby(self.df['date'].dt.to_period('M'))['total_value'].sum()
        fig_trend = px.line(
            x=[str(x) for x in monthly_sales.index],
            y=monthly_sales.values,
            title="Monthly Sales Trend",
            markers=True
        )
        fig_trend.update_layout(
            xaxis_title="Month", 
            yaxis_title="Total Revenue (₹)",
            height=Config.CHART_HEIGHT
        )
        
        # 4. Top Agents Performance
        top_agents = self.df.groupby('agentName')['total_value'].sum().sort_values(ascending=False).head(10)
        fig_agents = px.bar(
            x=top_agents.index,
            y=top_agents.values,
            title="Top 10 Agents by Revenue",
            color=top_agents.values,
            color_continuous_scale='plasma'
        )
        fig_agents.update_layout(
            xaxis_title="Agent Name", 
            yaxis_title="Total Revenue (₹)",
            height=Config.CHART_HEIGHT
        )
        
        return fig_status, fig_quality, fig_trend, fig_agents

def main():
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout=Config.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Admin Authentication
    st.sidebar.title("🔐 Admin Authentication")
    
    # Simple admin authentication
    admin_username = st.sidebar.text_input("Username:", placeholder="Enter admin username")
    admin_password = st.sidebar.text_input("Password:", type="password", placeholder="Enter admin password")
    
    # Default admin credentials (in production, use proper authentication)
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "admin123"
    
    if not admin_username or not admin_password:
        st.warning("⚠️ Please login with admin credentials to access the Dress Sales Monitoring System.")
        st.stop()
    
    if admin_username != ADMIN_USERNAME or admin_password != ADMIN_PASSWORD:
        st.error("❌ Invalid credentials. Access denied.")
        st.stop()
    
    # Show success message for admin
    st.sidebar.success(f"✅ Welcome, {admin_username}!")
    
    st.title("👗 Dress Sales Monitoring Chatbot")
    st.markdown("---")
    st.info("🔒 **Admin Access Only** - This system is restricted to authorized sales company administrators.")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot and training ML model..."):
            try:
                chatbot = DressSalesChatbot()
                chatbot.initialize()
                st.session_state.chatbot = chatbot
                st.success("✅ Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing chatbot: {str(e)}")
                st.stop()
    
    chatbot = st.session_state.chatbot
    
    # Sidebar
    st.sidebar.title("🤖 Chatbot Controls")
    
    # API Test Button
    if st.sidebar.button("🔧 Test Gemini API"):
        with st.sidebar:
            with st.spinner("Testing API connection..."):
                is_working, message = chatbot.test_gemini_api()
                if is_working:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    # Show API key status
    api_key = Config.get_gemini_api_key()
    if api_key == "AIzaSyBwZR5MJm4r5NE1AX5FbzOveBJhuYYbJCQ":
        st.sidebar.warning("⚠️ Gemini API key not configured")
    else:
        st.sidebar.success("✅ API key configured")
    
    st.sidebar.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Questions About Dress Sales")
        
        # Predefined questions
        st.markdown("**Quick Questions:**")
        predefined_questions = chatbot.get_predefined_questions()
        
        # Create 3 columns for predefined questions
        q_col1, q_col2, q_col3 = st.columns(3)
        
        with q_col1:
            for i in range(0, len(predefined_questions), 3):
                if i < len(predefined_questions):
                    if st.button(predefined_questions[i], key=f"q_{i}"):
                        st.session_state.user_question = predefined_questions[i]
        
        with q_col2:
            for i in range(1, len(predefined_questions), 3):
                if i < len(predefined_questions):
                    if st.button(predefined_questions[i], key=f"q_{i}"):
                        st.session_state.user_question = predefined_questions[i]
        
        with q_col3:
            for i in range(2, len(predefined_questions), 3):
                if i < len(predefined_questions):
                    if st.button(predefined_questions[i], key=f"q_{i}"):
                        st.session_state.user_question = predefined_questions[i]
        
        # Custom question input
        st.markdown("**Or ask your own question:**")
        user_question = st.text_input(
            "Enter your question about dress sales:",
            value=st.session_state.get('user_question', ''),
            placeholder="e.g., What is our total revenue this month?"
        )
        
        if st.button("🚀 Get Answer", type="primary"):
            if user_question:
                with st.spinner("Analyzing data and generating response..."):
                    answer = chatbot.answer_question(user_question)
                    
                    st.markdown("### 🤖 Chatbot Response:")
                    st.markdown(f"**Question:** {user_question}")
                    st.markdown(f"**Answer:** {answer}")
                    
                    # Store in session for chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Limit chat history
                    if len(st.session_state.chat_history) > Config.MAX_CHAT_HISTORY:
                        st.session_state.chat_history = st.session_state.chat_history[-Config.MAX_CHAT_HISTORY:]
            else:
                st.warning("Please enter a question!")
    
    with col2:
        st.subheader("📊 Sales Prediction")
        
        st.markdown("**Predict Future Sales:**")
        
        # Prediction form
        with st.form("prediction_form"):
            quality = st.selectbox("Dress Quality:", Config.get_available_qualities())
            weave = st.selectbox("Weave Type:", Config.get_available_weaves())
            composition = st.selectbox("Composition:", Config.get_available_compositions())
            rate = st.number_input("Rate per unit (₹):", min_value=1, value=300)
            
            if st.form_submit_button("🔮 Predict Sales"):
                prediction = chatbot.predict_sales(quality, weave, composition, rate)
                st.success(f"**Predicted Sales Quantity:** {prediction:.0f} units")
                st.info(f"**Estimated Revenue:** ₹{prediction * rate:,.2f}")
    
    # Analytics Dashboard
    st.markdown("---")
    st.subheader("📈 Sales Analytics Dashboard")
    
    analytics = chatbot.get_sales_analytics()
    
    # Key metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Orders", f"{analytics['total_sales']:,}")
    
    with metric_col2:
        st.metric("Total Revenue", f"₹{analytics['total_revenue']:,.0f}")
    
    with metric_col3:
        st.metric("Avg Order Value", f"₹{analytics['avg_order_value']:,.0f}")
    
    with metric_col4:
        st.metric("Total Quantity", f"{analytics['total_quantity']:,.0f}")
    
    # Visualizations
    fig_status, fig_quality, fig_trend, fig_agents = chatbot.create_visualizations()
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.plotly_chart(fig_status, use_container_width=True)
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with viz_col2:
        st.plotly_chart(fig_trend, use_container_width=True)
        st.plotly_chart(fig_agents, use_container_width=True)
    
    # Chat History
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💭 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})"):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Machine Learning & Gemini AI | 👗 Dress Sales Monitoring System</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

