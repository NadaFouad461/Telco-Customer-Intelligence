import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Telco Customer Intelligence",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "telco_cleaned.csv"
KMEANS_PATH = BASE_DIR / "models" / "kmeans_customer_segmentation.pkl"
CLUSTER_PREPROCESSOR_PATH = BASE_DIR / "models" / "clustering_preprocessor.pkl"


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
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# =========================================================
# LOAD CLUSTERING MODEL
# =========================================================

@st.cache_resource
def load_clustering_artifacts():
    kmeans_model = joblib.load(KMEANS_PATH)
    clustering_preprocessor = joblib.load(CLUSTER_PREPROCESSOR_PATH)
    return kmeans_model, clustering_preprocessor


kmeans_model, clustering_preprocessor = load_clustering_artifacts()

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
    "Total Charges",
]


def assign_clusters(data):
    """Assign existing K-Means cluster labels without retraining."""
    if data.empty:
        result = data.copy()
        result["Cluster"] = pd.Series(dtype="int64")
        return result

    cluster_input = data[cluster_features].copy()
    cluster_processed = clustering_preprocessor.transform(cluster_input)
    cluster_labels = kmeans_model.predict(cluster_processed)

    result = data.copy()
    result["Cluster"] = cluster_labels
    return result


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
    color: #64748B;
    font-size: 14px;
    margin-bottom: 15px;
}

.segment-title {
    color: #172033 !important;
    font-size: 17px;
    font-weight: 700;
}

.segment-kpi-label {
    color: #334155;
    font-size: 14px;
    line-height: 1.35;
    margin-bottom: 8px;
}

.segment-kpi-value {
    color: #172033;
    font-size: 30px;
    font-weight: 500;
    line-height: 1.2;
    white-space: nowrap;
    margin-bottom: 14px;
}

.segment-kpi-caption {
    color: #64748B;
    font-size: 14px;
    line-height: 1.35;
}

.segment-label {
    color: #0F766E !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.dashboard-note {
    color: #64748B;
    font-size: 13px;
    padding-top: 18px;
    margin-top: 30px;
    border-top: 1px solid #D9E2EC;
}


div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D9E2EC !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    min-height: 115px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}

div[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 14px !important;
    white-space: nowrap !important;
}

div[data-testid="stMetricValue"] {
    color: #172033 !important;
    font-size: 28px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-color: #D9E2EC !important;
    border-radius: 14px;
}

</style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

with st.sidebar:

    st.markdown("## Filters")

    st.caption(
        "Use the filters to explore different customer groups."
    )

    st.markdown("---")

    # -----------------------------------------------------
    # Initialize filter state
    # -----------------------------------------------------

    if "filter_gender" not in st.session_state:
        st.session_state.filter_gender = "All"

    if "filter_senior" not in st.session_state:
        st.session_state.filter_senior = "All"

    if "filter_internet" not in st.session_state:
        st.session_state.filter_internet = "All"

    if "filter_contract" not in st.session_state:
        st.session_state.filter_contract = "All"

    if "filter_payment" not in st.session_state:
        st.session_state.filter_payment = "All"

    if "filter_status" not in st.session_state:
        st.session_state.filter_status = "All"


    # -----------------------------------------------------
    # Gender
    # -----------------------------------------------------

    gender_options = ["All"] + sorted(
        df["Gender"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_gender = st.selectbox(
        "Gender",
        gender_options,
        key="filter_gender",
    )


    # -----------------------------------------------------
    # Senior Citizen
    # -----------------------------------------------------

    senior_options = ["All", "Yes", "No"]

    selected_senior = st.selectbox(
        "Senior Citizen",
        senior_options,
        key="filter_senior",
    )


    # -----------------------------------------------------
    # Internet Service
    # -----------------------------------------------------

    internet_options = ["All"] + sorted(
        df["Internet Service"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_internet = st.selectbox(
        "Internet Service",
        internet_options,
        key="filter_internet",
    )


    # -----------------------------------------------------
    # Contract
    # -----------------------------------------------------

    contract_options = ["All"] + sorted(
        df["Contract"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_contract = st.selectbox(
        "Contract",
        contract_options,
        key="filter_contract",
    )


    # -----------------------------------------------------
    # Payment Method
    # -----------------------------------------------------

    payment_options = ["All"] + sorted(
        df["Payment Method"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_payment = st.selectbox(
        "Payment Method",
        payment_options,
        key="filter_payment",
    )


    # -----------------------------------------------------
    # Customer Status
    # -----------------------------------------------------

    status_options = [
        "All",
        "Retained",
        "Churned",
    ]

    selected_status = st.selectbox(
        "Customer Status",
        status_options,
        key="filter_status",
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Reset Filters
    # -----------------------------------------------------

    if st.button("Reset Filters", use_container_width=True):
        for key in list(st.session_state.keys()):
           del st.session_state[key]
    
        st.query_params.clear()
        st.rerun()


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df.copy()


# ---------------------------------------------------------
# 1. Gender
# ---------------------------------------------------------

if selected_gender != "All":

    filtered_df = filtered_df[
        filtered_df["Gender"]
        .astype(str)
        .str.strip()
        .eq(selected_gender)
    ]


# ---------------------------------------------------------
# 2. Senior Citizen
# ---------------------------------------------------------

if selected_senior != "All":

    filtered_df = filtered_df[
        filtered_df["Senior Citizen"]
        .astype(str)
        .str.strip()
        .eq(selected_senior)
    ]

    


# ---------------------------------------------------------
# 3. Internet Service
# ---------------------------------------------------------

if selected_internet != "All":

    filtered_df = filtered_df[
        filtered_df["Internet Service"]
        .astype(str)
        .str.strip()
        .eq(selected_internet)
    ]


# ---------------------------------------------------------
# 4. Contract
# ---------------------------------------------------------

if selected_contract != "All":

    filtered_df = filtered_df[
        filtered_df["Contract"]
        .astype(str)
        .str.strip()
        .eq(selected_contract)
    ]


# ---------------------------------------------------------
# 5. Payment Method
# ---------------------------------------------------------

if selected_payment != "All":

    filtered_df = filtered_df[
        filtered_df["Payment Method"]
        .astype(str)
        .str.strip()
        .eq(selected_payment)
    ]


# ---------------------------------------------------------
# 6. Customer Status
# ---------------------------------------------------------

if selected_status != "All":

    churn_value = (
        "Yes"
        if selected_status == "Churned"
        else "No"
    )

    filtered_df = filtered_df[
        filtered_df["Churn Label"]
        .astype(str)
        .str.strip()
        .eq(churn_value)
    ]


# =========================================================
# EMPTY FILTER RESULT
# =========================================================

if filtered_df.empty:

    st.warning(
        "No customers match the selected filter combination."
    )

else:
    filtered_df = assign_clusters(filtered_df)


# =========================================================
# PLOTLY STYLE
# =========================================================

def style_chart(fig, height=360):

    fig.update_layout(
        template="plotly_white",

        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,

        font=dict(
            family="Arial",
            color=TEXT,
            size=12,
        ),

        title=dict(
            font=dict(
                size=16,
                color=NAVY,
            ),
            x=0,
            xanchor="left",
        ),

        margin=dict(
            l=25,
            r=25,
            t=65,
            b=35,
        ),

        height=height,

        showlegend=False,

        xaxis=dict(
            title_font=dict(
                color=TEXT,
                size=12,
            ),
            tickfont=dict(
                color=TEXT,
                size=11,
            ),
            showgrid=False,
            showline=True,
            linecolor=BORDER,
            zeroline=False,
            fixedrange=True,  
        ),

        yaxis=dict(
            title_font=dict(
                color=TEXT,
                size=12,
            ),
            tickfont=dict(
                color=TEXT,
                size=11,
            ),
            showgrid=True,
            gridcolor=GRID,
            gridwidth=1,
            showline=False,
            zeroline=False,
            fixedrange=True,  
        ),
    )

    return fig


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Telco Customer Intelligence")

st.markdown(
    "Understand customer behavior, churn patterns, customer value, "
    "and customer segments."
)


# =========================================================
# ACTIVE FILTER SUMMARY
# =========================================================

active_filters = []

if selected_gender != "All":
    active_filters.append(f"Gender: {selected_gender}")

if selected_senior != "All":
    active_filters.append(
        f"Senior Citizen: {selected_senior}"
    )

if selected_internet != "All":
    active_filters.append(
        f"Internet: {selected_internet}"
    )

if selected_contract != "All":
    active_filters.append(
        f"Contract: {selected_contract}"
    )

if selected_payment != "All":
    active_filters.append(
        f"Payment: {selected_payment}"
    )

if selected_status != "All":
    active_filters.append(
        f"Status: {selected_status}"
    )


if active_filters:

    st.info(
        " | ".join(active_filters)
    )


# =========================================================
# KPI VALUES
# =========================================================

total_customers = len(filtered_df)

if total_customers > 0:

    churn_rate = (
        filtered_df["Churn Label"]
        .eq("Yes")
        .mean()
        * 100
    )

    max_tenure = int(
        filtered_df["Tenure Months"].max()
    )

else:

    churn_rate = 0
    max_tenure = 0


# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Customers",
        value=f"{total_customers:,}",
    )
    st.caption("Customer records matching filters")

with col2:
    st.metric(
        label="Observed Churn",
        value=f"{churn_rate:.1f}%",
    )
    st.caption("Customers who churned")

with col3:
    if not filtered_df.empty:
        segment_counts = filtered_df["Cluster"].value_counts()
        most_represented_cluster = int(segment_counts.idxmax())
        most_represented_segment = f"Segment {most_represented_cluster + 1}"
        most_represented_count = int(segment_counts.loc[most_represented_cluster])
    else:
        most_represented_segment = "—"
        most_represented_count = 0

    st.metric(
        label="Dominant Segment",
        value=most_represented_segment,
    )
    st.caption(f"{most_represented_count:,} customers in group")

with col4:
    st.metric(
        label="Maximum Tenure",
        value=f"{max_tenure}",
    )
    st.caption("Months")


# =========================================================
# CHURN OVERVIEW
# =========================================================

st.subheader("Churn Overview")

st.markdown(
    '<div class="section-description">'
    "A quick view of customers who stayed and customers who churned."
    "</div>",
    unsafe_allow_html=True,
)


col1, col2 = st.columns([0.85, 1.15])


# =========================================================
# DONUT
# =========================================================

churn_data = (
    filtered_df["Churn Label"]
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Customers")
)

churn_data["Status"] = churn_data["Status"].map(
    {
        "No": "Retained",
        "Yes": "Churned",
    }
)


fig_churn = px.pie(
    churn_data,
    names="Status",
    values="Customers",
    hole=0.68,
    color="Status",
    color_discrete_map={
        "Retained": GRAY,
        "Churned": TEAL,
    },
)


fig_churn.update_traces(
    textinfo="percent",
    textfont=dict(
        size=15,
        color=NAVY,
    ),
)


fig_churn.update_layout(
    template="plotly_white",
    paper_bgcolor=WHITE,
    plot_bgcolor=WHITE,
    height=360,
    margin=dict(
        l=20,
        r=20,
        t=25,
        b=20,
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        y=-0.05,
        x=0.5,
        xanchor="center",
        font=dict(
            color=TEXT,
        ),
    ),
)


with col1:

    st.plotly_chart(
        fig_churn,
        use_container_width=True,
    )


# =========================================================
# CONTRACT CHURN
# =========================================================

contract_churn = (
    filtered_df.groupby("Contract")["Churn Value"]
    .mean()
    .mul(100)
    .reset_index(name="Churn Rate")
    .sort_values("Churn Rate")
)


fig_contract = px.bar(
    contract_churn,
    x="Churn Rate",
    y="Contract",
    orientation="h",
    text="Churn Rate",
)


fig_contract.update_traces(
    marker_color=TEAL,
    texttemplate="%{text:.1f}%",
    textposition="outside",
)


fig_contract.update_layout(
    title="Churn Rate by Contract",
    xaxis_title="Observed churn rate (%)",
    yaxis_title="",
    xaxis_range=[0, 50],
)


fig_contract = style_chart(
    fig_contract,
    height=360,
)


with col2:

    st.plotly_chart(
        fig_contract,
        use_container_width=True,
    )


# =========================================================
# CUSTOMER BEHAVIOR
# =========================================================

st.subheader("Customer Behavior")

st.markdown(
    '<div class="section-description">'
    "Differences between customers who stayed and customers who churned."
    "</div>",
    unsafe_allow_html=True,
)


col1, col2 = st.columns(2)


# =========================================================
# TENURE
# =========================================================

tenure_data = (
    filtered_df.groupby("Churn Label")["Tenure Months"]
    .mean()
    .reset_index()
)

tenure_data["Customer Status"] = (
    tenure_data["Churn Label"]
    .map(
        {
            "No": "Retained",
            "Yes": "Churned",
        }
    )
)


fig_tenure = px.bar(
    tenure_data,
    x="Customer Status",
    y="Tenure Months",
    text="Tenure Months",
    color="Customer Status",
    color_discrete_map={
        "Retained": GRAY,
        "Churned": TEAL,
    },
)


fig_tenure.update_traces(
    texttemplate="%{text:.1f} months",
    textposition="outside",
)


fig_tenure.update_layout(
    title="Average Tenure",
    yaxis_title="Months",
    xaxis_title="",
)


fig_tenure = style_chart(
    fig_tenure,
    height=360,
)


with col1:

    st.plotly_chart(
        fig_tenure,
        use_container_width=True,
    )


# =========================================================
# MONTHLY CHARGES
# =========================================================

monthly_data = (
    filtered_df.groupby("Churn Label")["Monthly Charges"]
    .mean()
    .reset_index()
)

monthly_data["Customer Status"] = (
    monthly_data["Churn Label"]
    .map(
        {
            "No": "Retained",
            "Yes": "Churned",
        }
    )
)


fig_monthly = px.bar(
    monthly_data,
    x="Customer Status",
    y="Monthly Charges",
    text="Monthly Charges",
    color="Customer Status",
    color_discrete_map={
        "Retained": GRAY,
        "Churned": TEAL,
    },
)


fig_monthly.update_traces(
    texttemplate="$%{text:.1f}",
    textposition="outside",
)


fig_monthly.update_layout(
    title="Average Monthly Charges",
    yaxis_title="Monthly charges",
    xaxis_title="",
)


fig_monthly = style_chart(
    fig_monthly,
    height=360,
)


with col2:

    st.plotly_chart(
        fig_monthly,
        use_container_width=True,
    )


# =========================================================
# PAYMENT METHOD
# =========================================================

st.subheader("Payment Method")

st.markdown(
    '<div class="section-description">'
    "Observed churn varies across different payment methods."
    "</div>",
    unsafe_allow_html=True,
)


payment_data = (
    filtered_df.groupby("Payment Method")["Churn Value"]
    .mean()
    .mul(100)
    .reset_index(name="Churn Rate")
    .sort_values("Churn Rate")
)


fig_payment = px.bar(
    payment_data,
    x="Churn Rate",
    y="Payment Method",
    orientation="h",
    text="Churn Rate",
)


fig_payment.update_traces(
    marker_color=TEAL,
    texttemplate="%{text:.1f}%",
    textposition="outside",
)


fig_payment.update_layout(
    title="Churn Rate by Payment Method",
    xaxis_title="Observed churn rate (%)",
    yaxis_title="",
    xaxis_range=[0, 55],
)


fig_payment = style_chart(
    fig_payment,
    height=420,
)


st.plotly_chart(
    fig_payment,
    use_container_width=True,
)


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================

st.subheader("Customer Segments")

st.markdown(
    '<div class="section-description">'
    "Three groups identified from service usage, tenure, contracts, "
    "payment behavior, and spending."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Dynamic segment assignment
# ---------------------------------------------------------
# The saved K-Means model assigns each filtered customer to
# one of the three existing customer segments.
# The model is NOT retrained in the Overview page.
# ---------------------------------------------------------

if not filtered_df.empty:

    segment_counts = (
        filtered_df["Cluster"]
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
    )

    segment_data = pd.DataFrame({
        "Cluster": [0, 1, 2],
        "Customers": segment_counts.values,
    })

    segment_data["Segment"] = segment_data["Cluster"].map(
        {
            0: "Segment 01",
            1: "Segment 02",
            2: "Segment 03",
        }
    )

    fig_segments = px.bar(
        segment_data,
        x="Segment",
        y="Customers",
        text="Customers",
    )

    fig_segments.update_traces(
        marker_color=TEAL,
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig_segments.update_layout(
        title="Customer Distribution Across Segments",
        xaxis_title="",
        yaxis_title="Customers",
    )

    fig_segments = style_chart(
        fig_segments,
        height=360,
    )

    st.plotly_chart(
        fig_segments,
        use_container_width=True,
    )

else:

    st.info(
        "No segment distribution is available because no customers "
        "match the selected filters."
    )


# =========================================================
# SEGMENT DETAILS
# =========================================================

col1, col2, col3 = st.columns(3)


segment_titles = {
    0: "Newer / Higher-Risk Internet Customers",
    1: "Basic / Non-Internet Customers",
    2: "Long-Tenure / Higher-Value Customers",
}


for column, cluster_id, number in zip(
    [col1, col2, col3],
    [0, 1, 2],
    [1, 2, 3],
):

    if filtered_df.empty:
        profile = None
    else:
        cluster_df = filtered_df[
            filtered_df["Cluster"] == cluster_id
        ]

        if cluster_df.empty:
            profile = None
        else:
            profile = {
                "customers": len(cluster_df),
                "tenure": cluster_df["Tenure Months"].mean(),
                "monthly": cluster_df["Monthly Charges"].mean(),
                "churn": cluster_df["Churn Value"].mean() * 100,
            }

    with column:

        with st.container(border=True):

            st.markdown(
                f'<div class="segment-label">'
                f'SEGMENT {number:02d}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="segment-title">'
                f'{segment_titles[cluster_id]}'
                f'</div>',
                unsafe_allow_html=True
            )

            if profile is None:

                st.write("0 customers match the current filters")

            else:

                st.write(
                    f'{profile["customers"]:,} customers'
                )

                st.write(
                    f'{profile["tenure"]:.1f} months average tenure'
                )

                st.write(
                    f'${profile["monthly"]:.2f} average monthly charges'
                )

                st.write(
                    f'{profile["churn"]:.1f}% observed churn'
                )


# =========================================================
# FOOTNOTE
# =========================================================

st.markdown(
    '<div class="dashboard-note">'
    "Observed patterns describe relationships in the dataset "
    "and should not be interpreted as causal effects."
    "</div>",
    unsafe_allow_html=True,
)