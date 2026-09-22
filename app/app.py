import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Telco Customer Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL STYLE
# =========================================================

st.markdown(
    """
<style>

/* =========================================
   MAIN APP
   ========================================= */

.stApp {
    background-color: #F7F9FC;
}


/* =========================================
   TOP STREAMLIT HEADER
   ========================================= */

header[data-testid="stHeader"] {
    background-color: #172033 !important;
}

header[data-testid="stHeader"] * {
    color: #FFFFFF !important;
}

div[data-testid="stToolbar"] {
    background-color: #172033 !important;
}


/* =========================================
   SIDEBAR
   ========================================= */

section[data-testid="stSidebar"] {
    background-color: #172033 !important;
    border-right: none !important;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0;
}


/* Selected navigation item */

section[data-testid="stSidebar"] [aria-selected="true"] {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border-radius: 8px;
}

section[data-testid="stSidebar"] [aria-selected="true"] * {
    color: #FFFFFF !important;
}


/* Sidebar width */

section[data-testid="stSidebar"] {
    min-width: 235px;
    max-width: 235px;
}


/* =========================================
   MAIN CONTENT
   ========================================= */

.block-container {
    max-width: 1200px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}


/* =========================================
   HEADINGS
   ========================================= */

h1 {
    color: #172033 !important;
}

h2 {
    color: #172033 !important;
}

h3 {
    color: #172033 !important;
}


/* =========================================
   METRIC CARDS
   ========================================= */

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


/* =========================================
   CONTAINERS / CARDS
   ========================================= */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-color: #D9E2EC !important;
    border-radius: 14px;
}


/* =========================================
   CAPTION
   ========================================= */

.stCaption {
    color: #64748B !important;
}


/* =========================================
   TEXT
   ========================================= */

p {
    color: #334155;
}

</style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# NAVIGATION
# =========================================================

pg = st.navigation(
    [
        st.Page(
            "pages/overview.py",
            title="Overview",
            icon="📡",
        ),
       st.Page("pages/customer_prediction.py", title="Customer Prediction", icon="👤"),
    ],
    position="sidebar",
)


# =========================================================
# RUN APP
# =========================================================

pg.run()