import os

class Config:
    """Configuration class for the Dress Sales Chatbot"""
    
    # Model Configuration
    MODEL_PARAMS = {
        'n_estimators': 100,
        'random_state': 42,
        'test_size': 0.2
    }
    
    # Data Configuration
    DATA_FILE = 'sales_data_1000_records.csv'
    MODEL_FILE = 'sales_prediction_model.pkl'
    ENCODERS_FILE = 'label_encoders.pkl'
    
    # UI Configuration
    PAGE_TITLE = "Dress Sales Monitoring Chatbot"
    PAGE_ICON = "👗"
    LAYOUT = "wide"
    
    # Chart Configuration
    CHART_HEIGHT = 400
    CHART_WIDTH = 600
    
    # Chat Configuration
    MAX_CHAT_HISTORY = 10
    
    # Gemini API Configuration - Loaded lazily
    _GEMINI_API_KEY = None
    
    @classmethod
    def get_gemini_api_key(cls):
        """Get Gemini API key (loaded lazily to avoid Streamlit context issues)"""
        if cls._GEMINI_API_KEY is None:
            try:
                import streamlit as st
                cls._GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "AIzaSyCYKuwXYcxW8u0y23UgYTKd9dx240bsm6U"))
            except Exception:
                # Fallback if Streamlit context is not available
                cls._GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCYKuwXYcxW8u0y23UgYTKd9dx240bsm6U")
        return cls._GEMINI_API_KEY
    
    # Validation
    @staticmethod
    def validate_api_key():
        """Validate if Gemini API key is properly configured"""
        try:
            import streamlit as st
            api_key = Config.get_gemini_api_key()
            if api_key == "AIzaSyCYKuwXYcxW8u0y23UgYTKd9dx240bsm6U":
                st.error("""
                ❌ Gemini API key not configured!
                
                Please follow these steps:
                1. Get your free API key from: https://makersuite.google.com/app/apikey
                2. Open `.streamlit/secrets.toml`
                3. Replace 'your-gemini-api-key-here' with your actual API key
                
                Example:
                ```toml
                GEMINI_API_KEY = "AIzaSyCYKuwXYcxW8u0y23UgYTKd9dx240bsm6U"
                ```
                """)
                return False
            return True
        except Exception:
            # If Streamlit context is not available, just return True
            return True
    
    @staticmethod
    def get_available_qualities():
        """Get list of available dress qualities"""
        return ['premium', 'standard', 'economy']
    
    @staticmethod
    def get_available_weaves():
        """Get list of available weave types"""
        return ['spandex', 'linen', 'denim', 'satin', 'crepe', 'plain', 'twill']
    
    @staticmethod
    def get_available_compositions():
        """Get list of available compositions"""
        return ['Cotton 100%', 'Cotton 50%', 'Polyester 80%', 'Nylon 40%', 'Silk 60%', 'Viscose 70%', '20%'] 