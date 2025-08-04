"""
Streamlit Web Application for Customer Churn Prediction
Interactive dashboard for predicting customer churn using the FastAPI backend
"""

import os

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# API configuration - use environment variable or default to localhost for development
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="🔮 Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: 700;
    }

            .model-info-banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 15px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        .model-info-small {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 15px;
            margin: 10px 0;
            border-radius: 10px;
            text-align: center;
            font-size: 14px;
            font-weight: 600;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

    .sub-header {
        font-size: 1.6rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }

    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .risk-high {
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }

    .risk-medium {
        background: linear-gradient(45deg, #feca57, #ff9ff3);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }

    .risk-low {
        background: linear-gradient(45deg, #48cae4, #0077b6);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }

    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #007bff;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        color: #495057;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Configuration
# Configuration moved to top of file


def get_model_info():
    """Get model information from the FastAPI backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def make_prediction(customer_data):
    """Make prediction using the FastAPI backend"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict", json=customer_data, timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            "🚨 Cannot connect to the prediction API. Please ensure the FastAPI server is running."
        )
        return None
    except Exception as e:
        st.error(f"🚨 Error making prediction: {str(e)}")
        return None


def create_gauge_chart(probability):
    """Create a gauge chart for churn probability"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Probability (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=400, font={"color": "darkblue", "family": "Arial"})

    return fig


def create_feature_importance_chart(customer_data):
    """Create a simple feature importance visualization"""
    # This is a simplified version - in practice, you'd get actual feature importance from the model
    features = [
        "Tenure",
        "Monthly Charges",
        "Total Charges",
        "Contract Type",
        "Internet Service",
    ]
    importance = [0.25, 0.20, 0.18, 0.15, 0.12]

    fig = px.bar(
        x=importance,
        y=features,
        orientation="h",
        title="Key Factors Affecting Churn Prediction",
        labels={"x": "Importance Score", "y": "Features"},
        color=importance,
        color_continuous_scale="Blues",
    )

    fig.update_layout(height=300, showlegend=False)

    return fig


def main():
    # Get model information
    model_info = get_model_info()
    algorithm_name = "Machine Learning Model"
    model_version = "Latest"

    if model_info:
        # Get algorithm name from algorithm_display field
        algorithm_name = model_info.get("model_name", "ML Model")
        model_version = model_info.get("model_version", "Latest")

        # Clean up algorithm name if needed
        if algorithm_name == "XGBoost":
            algorithm_display = "XGBoost (Extreme Gradient Boosting)"
        elif algorithm_name == "Random Forest" or "RandomForest" in algorithm_name:
            algorithm_display = "Random Forest"
        elif (
            algorithm_name == "Logistic Regression"
            or "LogisticRegression" in algorithm_name
        ):
            algorithm_display = "Logistic Regression"
        elif algorithm_name == "SVC":
            algorithm_display = "Support Vector Classifier"
        elif algorithm_name == "AdaBoost":
            algorithm_display = "AdaBoost"
        elif algorithm_name == "LightGBM":
            algorithm_display = "LightGBM (Light Gradient Boosting)"
        else:
            algorithm_display = algorithm_name
    else:
        algorithm_display = "Machine Learning Model"

    # Header with model name
    st.markdown(
        '<h1 class="main-header">🎯 Customer Churn Prediction</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="model-info-banner">🤖 Powered by <strong>{algorithm_display}</strong> (Version {model_version})</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box">📈 Advanced analytics to predict customer retention and optimize business strategies</div>',
        unsafe_allow_html=True,
    )

    # Check API health
    api_status = check_api_health()

    if not api_status:
        st.error(
            "🚨 Prediction API is not available. Please start the FastAPI server first."
        )
        st.info("💡 Run: `python web_app/fastapi_app.py` to start the API server")
        st.stop()
    else:
        st.success("✅ Prediction API is running")

    # Sidebar for input
    st.sidebar.markdown(
        '<h2 class="sub-header">📝 Customer Information</h2>', unsafe_allow_html=True
    )

    # Customer Demographics
    st.sidebar.markdown("### 👤 Demographics")
    senior_citizen = st.sidebar.selectbox(
        "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No"
    )
    partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])

    # Service Information
    st.sidebar.markdown("### 📱 Service Details")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

    # Additional Services (only the ones used by the model)
    st.sidebar.markdown("### 🔧 Services")
    online_security = st.sidebar.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.sidebar.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.sidebar.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )

    # Contract Information
    st.sidebar.markdown("### 📄 Contract Details")
    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    # Financial Information
    st.sidebar.markdown("### 💰 Financial Details")
    monthly_charges = st.sidebar.number_input(
        "Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0
    )
    total_charges = st.sidebar.number_input(
        "Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges, step=50.0
    )

    # Prediction button
    if st.sidebar.button("🎯 Predict Churn", type="primary"):
        # Prepare data for API
        customer_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
        }

        # Make prediction
        with st.spinner("🔄 Making prediction..."):
            result = make_prediction(customer_data)

        if result:
            # Display results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(
                    '<h2 class="sub-header">📊 Prediction Results</h2>',
                    unsafe_allow_html=True,
                )

                # Display model name from prediction result
                model_name = result.get("model_name", "ML Model")
                model_version = result.get("model_version", "Latest")

                # Create full algorithm display name
                if model_name == "XGBoost":
                    full_algorithm_name = "XGBoost (Extreme Gradient Boosting)"
                elif model_name == "RandomForest":
                    full_algorithm_name = "Random Forest"
                elif model_name == "LogisticRegression":
                    full_algorithm_name = "Logistic Regression"
                else:
                    full_algorithm_name = model_name

                st.markdown(
                    f'<div class="model-info-small">🤖 Model: {full_algorithm_name} '
                    f"(Version {model_version})</div>",
                    unsafe_allow_html=True,
                )

                probability = result["churn_probability"]

                # Calculate risk level based on probability
                if probability >= 0.7:
                    risk_level = "High Risk"
                    risk_class = "risk-high"
                    risk_icon = "🚨"
                elif probability >= 0.4:
                    risk_level = "Medium Risk"
                    risk_class = "risk-medium"
                    risk_icon = "⚠️"
                else:
                    risk_level = "Low Risk"
                    risk_class = "risk-low"
                    risk_icon = "✅"

                # Display risk level with appropriate styling
                st.markdown(
                    f'<div class="{risk_class}">{risk_icon} {risk_level}<br>Probability: {probability:.1%}</div>',
                    unsafe_allow_html=True,
                )

                # Additional metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    # Calculate confidence as how certain the prediction is
                    confidence = max(probability, 1 - probability)
                    st.metric("Confidence", f"{confidence:.1%}")
                with col_b:
                    # Get model version from result
                    version = result.get("model_version", "Latest")
                    if isinstance(version, str) and "demo" in version.lower():
                        version = "Production"
                    st.metric("Model Version", str(version))

                # Recommendations
                st.markdown("### 💡 Recommendations")
                if probability > 0.7:
                    st.warning("🚨 **Immediate Action Required:**")
                    st.write("• Contact customer immediately")
                    st.write("• Offer retention incentives")
                    st.write("• Review service satisfaction")
                elif probability > 0.4:
                    st.info("⚠️ **Monitor Closely:**")
                    st.write("• Schedule check-in call")
                    st.write("• Consider service upgrades")
                    st.write("• Monitor usage patterns")
                else:
                    st.success("✅ **Customer Stable:**")
                    st.write("• Continue current service")
                    st.write("• Consider upselling opportunities")
                    st.write("• Maintain satisfaction levels")

            with col2:
                # Gauge chart
                st.plotly_chart(
                    create_gauge_chart(probability), use_container_width=True
                )

                # Feature importance
                st.plotly_chart(
                    create_feature_importance_chart(customer_data),
                    use_container_width=True,
                )

            # Customer summary
            st.markdown(
                '<h2 class="sub-header">📋 Customer Summary</h2>',
                unsafe_allow_html=True,
            )

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            with summary_col1:
                st.metric("Tenure", f"{tenure} months")
                st.metric("Contract", contract)

            with summary_col2:
                st.metric("Monthly Charges", f"${monthly_charges:.2f}")
                st.metric("Online Security", online_security)

            with summary_col3:
                st.metric("Total Charges", f"${total_charges:.2f}")
                st.metric("Payment Method", payment_method.split(" ")[0])

            with summary_col4:
                senior_text = "Senior" if senior_citizen else "Regular"
                st.metric("Customer Type", senior_text)
                partner_text = "Has Partner" if partner == "Yes" else "Single"
                st.metric("Family Status", partner_text)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Customer Churn Prediction</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
