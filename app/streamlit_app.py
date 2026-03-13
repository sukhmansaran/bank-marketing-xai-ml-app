import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Bank Deposit Prediction", layout="wide")

# Title
st.title("🏦 Bank Deposit Prediction with XAI")
st.markdown("Predict whether a customer will subscribe to a term deposit with XGBoost & SHAP")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained XGBoost model from JSON"""
    try:
        if os.path.exists('models/model_xgb.json'):
            model = XGBClassifier()
            model.load_model('models/model_xgb.json')
            return model
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_label_encoders():
    """Load label encoders for categorical features"""
    try:
        if os.path.exists('models/label_encoders_xgb.pkl'):
            with open('models/label_encoders_xgb.pkl', 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading label encoders: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load training and test data"""
    try:
        # Load with float_precision='round_trip' to handle scientific notation
        train_df = pd.read_csv('data/processed/train_df.csv', float_precision='round_trip')
        test_df = pd.read_csv('data/processed/test_df.csv', float_precision='round_trip')
        
        # Convert numeric columns that might have scientific notation strings
        numeric_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                       'euribor3m', 'nr.employed']
        
        for col in numeric_cols:
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        
        return train_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_resource
def get_shap_explainer(_model, _X_train):
    """Create SHAP explainer"""
    try:
        return shap.TreeExplainer(_model)
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {str(e)}")
        return None

def preprocess_input(input_data, label_encoders):
    """Preprocess input for XGBoost model using label encoders"""
    try:
        input_processed = input_data.copy()
        
        # Apply label encoders to categorical columns
        for col, encoder in label_encoders.items():
            if col in input_processed.columns:
                input_processed[col] = encoder.transform(input_processed[col])
        
        return input_processed, True, None
    except Exception as e:
        return None, False, f"Preprocessing error: {str(e)}"

def prepare_training_data(train_df, label_encoders):
    """Prepare training data for SHAP using label encoders"""
    try:
        X_train = train_df.drop('y', axis=1).copy()
        
        # Apply label encoders to categorical columns
        for col, encoder in label_encoders.items():
            if col in X_train.columns:
                X_train[col] = encoder.transform(X_train[col])
        
        # Ensure all columns are proper numeric types (not strings)
        # This handles any edge cases where values might be string representations
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                # Try to convert object columns to numeric
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            # Ensure no NaN values
            if X_train[col].isna().any():
                X_train[col] = X_train[col].fillna(0)
        
        return X_train, True, None
    except Exception as e:
        return None, False, f"Error preparing training data: {str(e)}"

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

page = st.sidebar.radio("Navigation", ["Home", "Data Analysis", "Make Prediction", "Model Interpretability", "Model Info"])

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "Home":
    st.markdown("""
    ### Welcome to the Bank Deposit Prediction System
    
    This application uses XGBoost machine learning to predict whether a bank customer will subscribe to a term deposit.
    
    **Features:**
    - 📊 Explore the dataset
    - 🔮 Make predictions for new customers
    - 📈 View model performance metrics
    - 🔍 Understand predictions with SHAP explanations
    
    **Dataset:** Portuguese bank marketing campaign data
    
    **Model:** XGBoost Classifier with 88%+ accuracy
    """)


# ============================================================================
# DATA ANALYSIS PAGE
# ============================================================================

elif page == "Data Analysis":
    st.header("📊 Data Analysis")
    
    train_df, test_df = load_data()
    
    if train_df is None or test_df is None:
        st.error("Unable to load data files")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(train_df))
        with col2:
            st.metric("Test Samples", len(test_df))
        
        st.subheader("Dataset Overview")
        st.write(train_df.head())
        
        st.subheader("Dataset Statistics")
        st.write(train_df.describe())
        
        st.subheader("Missing Values")
        missing = train_df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values")
        else:
            st.write(missing)


# ============================================================================
# MAKE PREDICTION PAGE
# ============================================================================

elif page == "Make Prediction":
    st.header("🔮 Make a Prediction")
    
    model = load_model()
    label_encoders = load_label_encoders()
    
    if model is None or label_encoders is None:
        st.error("❌ Model not trained yet. Please run: `python src/train_model_xg.py`")
    else:
        train_df, _ = load_data()
        
        if train_df is None:
            st.error("Unable to load training data")
        else:
            st.info("ℹ️ Using XGBoost model")
            
            st.subheader("Customer Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                job = st.selectbox("Job", sorted(train_df['job'].unique()))
                marital = st.selectbox("Marital Status", sorted(train_df['marital'].unique()))
                education = st.selectbox("Education", sorted(train_df['education'].unique()))
                default = st.selectbox("Credit Default", sorted(train_df['default'].unique()))
            
            with col2:
                housing = st.selectbox("Housing Loan", sorted(train_df['housing'].unique()))
                loan = st.selectbox("Personal Loan", sorted(train_df['loan'].unique()))
                contact = st.selectbox("Contact Type", sorted(train_df['contact'].unique()))
                month = st.selectbox("Month", sorted(train_df['month'].unique()))
                day_of_week = st.selectbox("Day of Week", sorted(train_df['day_of_week'].unique()))
            
            with col3:
                duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=100)
                campaign = st.number_input("Campaign Contacts", min_value=1, max_value=50, value=1)
                pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=999, value=-1)
                previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0)
                poutcome = st.selectbox("Previous Outcome", sorted(train_df['poutcome'].unique()))
            
            st.subheader("Economic Indicators")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                emp_var_rate = st.number_input("Employment Variation Rate (%)", value=1.1)
            
            with col2:
                cons_price_idx = st.number_input("Consumer Price Index", value=93.5)
            
            with col3:
                cons_conf_idx = st.number_input("Consumer Confidence Index", value=-46.2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                euribor3m = st.number_input("Euribor 3 Month Rate (%)", value=1.3)
            
            with col2:
                nr_employed = st.number_input("Number Employed", value=5191.0)
            
            if st.button("Predict", type="primary"):
                # Prepare input
                input_data = pd.DataFrame({
                    'age': [age],
                    'job': [job],
                    'marital': [marital],
                    'education': [education],
                    'default': [default],
                    'housing': [housing],
                    'loan': [loan],
                    'contact': [contact],
                    'month': [month],
                    'day_of_week': [day_of_week],
                    'duration': [duration],
                    'campaign': [campaign],
                    'pdays': [pdays],
                    'previous': [previous],
                    'poutcome': [poutcome],
                    'emp.var.rate': [emp_var_rate],
                    'cons.price.idx': [cons_price_idx],
                    'cons.conf.idx': [cons_conf_idx],
                    'euribor3m': [euribor3m],
                    'nr.employed': [nr_employed]
                })
                
                try:
                    # Preprocess
                    input_processed, success, error = preprocess_input(input_data.copy(), label_encoders)
                    
                    if not success:
                        st.error(f"❌ {error}")
                    else:
                        # Make prediction
                        prediction = model.predict(input_processed)[0]
                        probability = model.predict_proba(input_processed)[0]
                        
                        st.success("✅ Prediction Complete!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction", "✅ Will Subscribe" if prediction == 1 else "❌ Won't Subscribe")
                        with col2:
                            st.metric("Confidence", f"{max(probability)*100:.1f}%")
                        
                        # SHAP explanation
                        st.subheader("📊 Why This Prediction?")
                        
                        try:
                            # Prepare training data
                            X_train, success, error = prepare_training_data(train_df, label_encoders)
                            
                            if not success:
                                st.warning(f"⚠️ Could not generate SHAP explanation: {error}")
                            else:
                                # Create SHAP explainer
                                explainer = get_shap_explainer(model, X_train)
                                
                                if explainer is None:
                                    st.warning("⚠️ Could not create SHAP explainer")
                                else:
                                    shap_values = explainer.shap_values(input_processed)
                                    
                                    # Force plot
                                    st.markdown("**SHAP Force Plot** - Shows how each feature pushes the prediction")
                                    try:
                                        if len(shap_values.shape) > 1:
                                            sample_shap = shap_values[0]
                                        else:
                                            sample_shap = shap_values
                                        
                                        sample_features = input_processed.iloc[0]
                                        base_value = explainer.expected_value
                                        
                                        plt.figure(figsize=(12, 3))
                                        shap.force_plot(base_value, sample_shap, sample_features, matplotlib=True)
                                        st.pyplot(plt.gcf(), use_container_width=True)
                                        plt.close()
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not render force plot: {str(e)}")
                                    
                                    # Feature contribution table
                                    st.markdown("**Feature Contributions**")
                                    try:
                                        sample_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                                        contributions = pd.DataFrame({
                                            'Feature': input_processed.columns,
                                            'Value': input_processed.iloc[0].values,
                                            'SHAP Value': sample_shap,
                                            'Impact': ['Positive (Subscribe)' if float(x) > 0 else 'Negative (No Subscribe)' for x in sample_shap]
                                        }).sort_values('SHAP Value', ascending=False, key=abs)
                                        st.dataframe(contributions, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not display feature contributions: {str(e)}")
                        
                        except Exception as e:
                            st.warning(f"⚠️ SHAP explanation unavailable: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")


# ============================================================================
# MODEL INTERPRETABILITY PAGE
# ============================================================================

elif page == "Model Interpretability":
    st.header("🔍 Model Interpretability (SHAP)")
    
    model = load_model()
    label_encoders = load_label_encoders()
    
    if model is None or label_encoders is None:
        st.error("❌ Model not trained yet.")
    else:
        train_df, _ = load_data()
        
        if train_df is None:
            st.error("Unable to load training data")
        else:
            st.info("ℹ️ Using XGBoost model")
            
            st.markdown("""
            ### Understanding Model Predictions with SHAP
            
            SHAP (SHapley Additive exPlanations) values explain the contribution of each feature to the model's prediction.
            - **Red values**: Push prediction towards "Will Subscribe"
            - **Blue values**: Push prediction towards "Won't Subscribe"
            """)
            
            try:
                # Prepare training data
                X_train, success, error = prepare_training_data(train_df, label_encoders)
                
                if not success:
                    st.error(f"❌ {error}")
                else:
                    # Create SHAP explainer
                    explainer = get_shap_explainer(model, X_train)
                    
                    if explainer is None:
                        st.error("❌ Could not create SHAP explainer")
                    else:
                        # Get SHAP values for a sample
                        X_sample = X_train.sample(min(100, len(X_train)))
                        shap_values = explainer.shap_values(X_sample)
                        
                        # Tabs for different visualizations
                        tab1, tab2, tab3 = st.tabs(["Summary Plot", "Feature Importance", "Dependence Plots"])
                        
                        with tab1:
                            st.subheader("SHAP Summary Plot (Bar)")
                            st.markdown("Shows the mean absolute SHAP value for each feature")
                            
                            try:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                                st.pyplot(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"⚠️ Could not render summary plot: {str(e)}")
                        
                        with tab2:
                            st.subheader("SHAP Summary Plot (Beeswarm)")
                            st.markdown("Shows how each feature value impacts the prediction")
                            
                            try:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                shap.summary_plot(shap_values, X_sample, show=False)
                                st.pyplot(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"⚠️ Could not render beeswarm plot: {str(e)}")
                        
                        with tab3:
                            st.subheader("Feature Dependence Plots")
                            st.markdown("Shows how predictions change with feature values")
                            
                            try:
                                feature_cols = X_sample.columns.tolist()
                                selected_feature = st.selectbox("Select Feature", feature_cols)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                shap.dependence_plot(selected_feature, shap_values, X_sample, show=False)
                                st.pyplot(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"⚠️ Could not render dependence plot: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error generating interpretability visualizations: {str(e)}")

# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "Model Info":
    st.header("📈 Model Information")
    
    model = load_model()
    
    if model is None:
        st.info("ℹ️ Model not trained yet.")
    else:
        st.info("ℹ️ Using XGBoost model")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write("**Status:** ✅ Ready for predictions")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        try:
            train_df, _ = load_data()
            
            if train_df is None:
                st.error("Unable to load training data")
            else:
                feature_importance = pd.DataFrame({
                    'Feature': train_df.drop('y', axis=1).columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
                ax.set_xlabel('Importance')
                ax.set_title('Top 15 Feature Importances')
                st.pyplot(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error displaying feature importance: {str(e)}")
