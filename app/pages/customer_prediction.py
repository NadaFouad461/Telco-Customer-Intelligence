import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Prediction",
    page_icon="👤",
    layout="wide",
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_churn_model.pkl"
)

CLTV_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "linear_regression_cltv_model.pkl"
)

CLUSTER_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "kmeans_customer_segmentation.pkl"
)

CLUSTER_PREPROCESSOR_PATH = (
    BASE_DIR
    / "models"
    / "clustering_preprocessor.pkl"
)


# =========================================================
# COLORS
# =========================================================

NAVY = "#172033"
TEAL = "#0F766E"
GRAY = "#94A3B8"

TEXT = "#334155"
MUTED = "#64748B"

BORDER = "#D9E2EC"
GRID = "#E2E8F0"

WHITE = "#FFFFFF"


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1200px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

h1 {
    color: #172033 !important;
    font-size: 38px !important;
    font-weight: 750 !important;
}

h2 {
    color: #172033 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-top: 38px !important;
}

h3 {
    color: #172033 !important;
}

.section-description {
    color: #475569 !important;
    font-size: 14px;
    margin-bottom: 18px;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 110px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

div[data-testid="stMetricValue"] {
    color: #172033 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-color: #D9E2EC !important;
    border-radius: 14px;
}

div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 14px;
    padding: 24px;
}

/* ---------------------------------------------------------
   Input Fields and Selectboxes Styling
--------------------------------------------------------- */
div[data-baseweb="select"] > div, 
div[data-baseweb="input"] > div,
input {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span {
    color: #0F172A !important;
}

/* ---------------------------------------------------------
   Primary Submit Button Styling
--------------------------------------------------------- */
.stButton > button, 
div[data-testid="stFormSubmitButton"] > button {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 0.75rem 1.5rem !important;
    transition: background-color 0.2s ease;
}

.stButton > button:hover, 
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #115E59 !important;
    color: #FFFFFF !important;
}

label {
    color: #334155 !important;
    font-weight: 600 !important;
}

</style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    churn_model = joblib.load(
        MODEL_PATH
    )

    cltv_model = joblib.load(
        CLTV_MODEL_PATH
    )

    cluster_model = joblib.load(
        CLUSTER_MODEL_PATH
    )

    cluster_preprocessor = joblib.load(
        CLUSTER_PREPROCESSOR_PATH
    )

    return (
        churn_model,
        cltv_model,
        cluster_model,
        cluster_preprocessor,
    )


try:

    (
        churn_model,
        cltv_model,
        cluster_model,
        cluster_preprocessor,
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "The prediction models could not be loaded."
    )

    st.caption(
        f"Error: {e}"
    )


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Customer Prediction")

st.markdown(
    '<div class="section-description">'
    "Enter customer information to generate churn, customer value, "
    "and customer segment predictions."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("Customer Information")

st.markdown(
    '<div class="section-description">'
    "Provide the customer's demographic information, services, "
    "contract, and spending details."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# FORM
# =========================================================

with st.form("customer_prediction_form"):

    # -----------------------------------------------------
    # Demographics
    # -----------------------------------------------------

    st.markdown("### Customer Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"],
        )

    with col3:

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"],
        )

    with col4:

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"],
        )


    # -----------------------------------------------------
    # Services
    # -----------------------------------------------------

    st.markdown("### Services")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service",
            ],
        )

    with col3:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No",
            ],
        )

    with col4:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

    with col3:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

    with col4:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

    with col2:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
        )

    with col3:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "No",
                "Yes",
            ],
        )


    # -----------------------------------------------------
    # Payment
    # -----------------------------------------------------

    st.markdown("### Billing & Payment")

    col1, col2, col3 = st.columns(3)

    with col1:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
        )

    with col2:

        tenure_months = st.number_input(
            "Tenure Months",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
        )

    with col3:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=18.25,
            max_value=118.75,
            value=65.00,
            step=0.01,
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=780.0,
            step=0.01,
        )

    with col2:

        latitude = st.number_input(
            "Latitude",
            min_value=0.0,
            max_value=90.0,
            value=34.0,
            step=0.0001,
            format="%.4f",
        )

    with col3:

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=0.0,
            value=-118.0,
            step=0.0001,
            format="%.4f",
        )


    st.markdown("")

    predict_button = st.form_submit_button(
        "Generate Predictions",
        use_container_width=True,
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button and models_loaded:

    # -----------------------------------------------------
    # Build customer dataframe
    # -----------------------------------------------------

    customer = pd.DataFrame(
        {
            "Latitude": [latitude],
            "Longitude": [longitude],

            "Gender": [gender],
            "Senior Citizen": [senior_citizen],

            "Partner": [partner],
            "Dependents": [dependents],

            "Tenure Months": [tenure_months],

            "Phone Service": [phone_service],
            "Multiple Lines": [multiple_lines],

            "Internet Service": [internet_service],

            "Online Security": [online_security],
            "Online Backup": [online_backup],
            "Device Protection": [device_protection],
            "Tech Support": [tech_support],

            "Streaming TV": [streaming_tv],
            "Streaming Movies": [streaming_movies],

            "Contract": [contract],

            "Paperless Billing": [paperless_billing],

            "Payment Method": [payment_method],

            "Monthly Charges": [monthly_charges],
            "Total Charges": [total_charges],
        }
    )


    # =====================================================
    # 1. CHURN PREDICTION
    # =====================================================

    churn_probability = (
        churn_model
        .predict_proba(customer)[0, 1]
    )

    churn_prediction = (
        churn_model
        .predict(customer)[0]
    )


    churn_label = (
        "Likely to Churn"
        if churn_prediction == 1
        else "Likely to Stay"
    )


    # =====================================================
    # 2. CLTV PREDICTION
    # =====================================================

    cltv_customer = customer.copy()


    # Tenure Group

    cltv_customer["Tenure Group"] = pd.cut(
        cltv_customer["Tenure Months"],
        bins=[
            -1,
            12,
            24,
            48,
            72,
        ],
        labels=[
            "0-12",
            "13-24",
            "25-48",
            "49-72",
        ],
    )


    # Charge Per Tenure Ratio

    cltv_customer["Charge_Per_Tenure_Ratio"] = np.where(
        cltv_customer["Tenure Months"] > 0,

        cltv_customer["Monthly Charges"]
        / cltv_customer["Tenure Months"],

        cltv_customer["Monthly Charges"],
    )


    predicted_cltv = (
        cltv_model
        .predict(cltv_customer)[0]
    )

    predicted_cltv = max(
        0,
        predicted_cltv,
    )


    # =====================================================
    # 3. CUSTOMER SEGMENT
    # =====================================================

    cluster_features = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges"
    ]


    cluster_customer = customer[cluster_features].copy()


    cluster_processed = cluster_preprocessor.transform(cluster_customer)


    cluster_id = cluster_model.predict(cluster_processed)[0]


    segment_names = {

        0: "Newer / Higher-Risk Internet Customers",

        1: "Basic / Non-Internet Customers",

        2: "Long-Tenure / Higher-Value Customers",
    }


    segment_name = segment_names.get(
        int(cluster_id),
        "Customer Segment",
    )


    # =====================================================
    # RESULTS
    # =====================================================

    st.subheader("Prediction Results")

    st.markdown(
        '<div class="section-description">'
        "Predictions generated from the customer information provided above."
        "</div>",
        unsafe_allow_html=True,
    )


    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # Churn
    # -----------------------------------------------------

    with col1:

        with st.container(border=True):

            st.metric(
                "Churn Risk",
                f"{churn_probability * 100:.1f}%",
            )

            st.caption(
                churn_label
            )


    # -----------------------------------------------------
    # CLTV
    # -----------------------------------------------------

    with col2:

        with st.container(border=True):

            st.metric(
                "Predicted CLTV",
                f"${predicted_cltv:,.0f}",
            )

            st.caption(
                "Estimated customer lifetime value"
            )


    # -----------------------------------------------------
    # Segment
    # -----------------------------------------------------

    with col3:

        with st.container(border=True):

            st.metric(
                "Customer Segment",
                f"Segment {int(cluster_id) + 1:02d}",
            )

            st.caption(
                segment_name
            )


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    st.subheader("Customer Summary")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        with st.container(border=True):

            st.metric(
                "Tenure",
                f"{tenure_months} months",
            )


    with col2:

        with st.container(border=True):

            st.metric(
                "Monthly Charges",
                f"${monthly_charges:,.2f}",
            )


    with col3:

        with st.container(border=True):

            st.metric(
                "Total Charges",
                f"${total_charges:,.2f}",
            )


    with col4:

        with st.container(border=True):

            st.metric(
                "Contract",
                contract,
            )


# =========================================================
# FOOTNOTE
# =========================================================

st.markdown(
    '<div style="'
    'color:#64748B;'
    'font-size:13px;'
    'padding-top:18px;'
    'margin-top:30px;'
    'border-top:1px solid #D9E2EC;'
    '">'
    "Predictions are model-based estimates and should be interpreted "
    "as decision-support information rather than guarantees."
    "</div>",
    unsafe_allow_html=True,
)