Telco Customer Intelligence System

An end-to-end machine learning project built on the IBM Telco Customer Churn dataset. The project combines customer churn prediction, Customer Lifetime Value (CLTV) regression, customer segmentation, exploratory data analysis, and an interactive Streamlit dashboard.

Project Overview

Telecom companies need to understand which customers are at risk of leaving, estimate customer lifetime value, and identify meaningful customer segments.

This project builds a complete customer intelligence workflow:

Clean and validate customer data.

Explore churn patterns and customer behavior.

Predict customer churn probability.

Estimate Customer Lifetime Value (CLTV).

Segment customers using unsupervised learning.

Present insights and individual predictions through a Streamlit application.

Dataset

The project uses the IBM Telco Customer Churn dataset available on Kaggle:

https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data

The original dataset contains 7,043 customers and 33 columns.

The cleaned project dataset contains 7,043 customers and 30 columns after removing constant columns.

Project Objectives

1. Churn Prediction

Predict whether a customer is likely to churn using supervised classification.

2. CLTV Prediction

Estimate Customer Lifetime Value using supervised regression.

3. Customer Segmentation

Group customers into meaningful behavioral/value-based segments using K-Means clustering.

4. Interactive Customer Intelligence

Build a Streamlit application that allows users to:

Explore customer-level business insights.

Filter the customer population.

View churn and customer behavior patterns.

Enter customer information.

Generate churn, CLTV, and segment predictions.

Project Structure

Telco-Customer-Intelligence/
│
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.xlsx
│   └── processed/
│       └── telco_cleaned.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_regression.ipynb
│   └── 06_clustering.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── utils.py
│
├── models/
│   ├── xgboost_churn_model.pkl
│   ├── linear_regression_cltv_model.pkl
│   ├── kmeans_customer_segmentation.pkl
│   ├── clustering_preprocessor.pkl
│   ├── churn_features.pkl
│   └── cltv_features.pkl
│
├── app/
│   ├── app.py
│   └── pages/
│       ├── overview.py
│       └── customer_prediction.py
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md

Data Understanding & Cleaning

The data understanding stage identified:

7,043 customers.

33 original columns.

7,043 unique Customer IDs.

No fully duplicated rows.

Count, Country, and State were constant and removed.

Total Charges was stored as text and contained 11 blank values.

The 11 blank Total Charges values belonged to customers with zero months of tenure and no churn, so they were treated as zero cumulative charges.

Churn Reason is missing for non-churned customers because it represents churn-related outcome information.

Leakage and Feature Decisions

The following columns were excluded from predictive features where appropriate:

CustomerID — identifier.

Churn Value — duplicate representation of the churn target.

Churn Score — existing churn-related score that could introduce leakage into baseline churn prediction.

Churn Reason — post-outcome information.

CLTV — regression target.

City, Lat Long, and Zip Code — high-cardinality/location identifiers not used in the core predictive models.

Exploratory Data Analysis

The EDA examined:

Overall churn distribution.

Contract type vs. churn.

Internet service vs. churn.

Payment method vs. churn.

Technical support and online security vs. churn.

Tenure behavior.

Monthly charges.

Total charges.

CLTV.

Numeric correlations.

IQR-based outlier checks.

Main Observed Patterns

The dataset contains approximately 26.5% churned customers and 73.5% non-churned customers.

Month-to-month customers show higher observed churn than customers on longer contracts.

Fiber optic customers show higher observed churn than DSL and customers without internet service.

Electronic check customers show higher observed churn than the other payment methods.

Customers with shorter tenure tend to have higher observed churn.

Churned customers generally have higher monthly charges.

Total Charges are strongly associated with Tenure Months because longer-tenure customers accumulate more charges.

No IQR outliers were identified for Tenure Months, Monthly Charges, Total Charges, or CLTV, so no outlier removal or capping was applied.

These are observed relationships in the dataset and should not be interpreted as causal effects.

Machine Learning

1. Churn Classification

Target

Churn Label

Mapped to:

No  -> 0
Yes -> 1

Preprocessing

Numerical features: StandardScaler.

Categorical features: OneHotEncoder.

handle_unknown="ignore" was used for categorical encoding.

Train/test split: 80/20.

Stratification was used because the target is imbalanced.

Models Evaluated

Logistic Regression

Random Forest

XGBoost

Results

Model

Accuracy

Precision

Recall

F1

ROC-AUC

Logistic Regression

0.744

0.511

0.781

0.618

0.848

Random Forest

0.776

0.565

0.671

0.614

0.842

XGBoost

0.757

0.528

0.770

0.627

0.849

The final churn model is the XGBoost pipeline, saved as:

models/xgboost_churn_model.pkl

The model uses class imbalance handling through scale_pos_weight.

2. CLTV Regression

Target

CLTV

Models Evaluated

Linear Regression

Random Forest Regressor

XGBoost Regressor

Baseline Results

Model

MAE

RMSE

R²

Linear Regression

898.17

1062.85

0.168

Random Forest

894.73

1056.19

0.179

XGBoost

889.91

1049.73

0.189

Feature Engineering

The final model retained:

Tenure Group

Charge_Per_Tenure_Ratio

Tenure Group:

0-12
13-24
25-48
49-72

Charge_Per_Tenure_Ratio uses Monthly Charges divided by Tenure Months, with Monthly Charges used as the fallback for zero-tenure customers.

Final Results

The final Linear Regression model achieved:

MAE: 871.97

RMSE: 1025.55

R²: 0.226

5-fold cross-validation:

Mean R²: 0.212

Standard deviation: 0.061

The saved model is:

models/linear_regression_cltv_model.pkl

3. Customer Segmentation

Customer segmentation was performed using K-Means clustering.

Preprocessing

The clustering pipeline uses:

StandardScaler for numerical features.

OneHotEncoder for categorical features.

The clustering model uses 19 customer attributes, including:

Gender

Senior Citizen

Partner

Dependents

Tenure Months

Phone Service

Multiple Lines

Internet Service

Online Security

Online Backup

Device Protection

Tech Support

Streaming TV

Streaming Movies

Contract

Paperless Billing

Payment Method

Monthly Charges

Total Charges

Final Segmentation

The final project uses 3 clusters.

Segment

Description

Customers

Avg. Tenure

Avg. Monthly Charges

Observed Churn

Segment 1

Newer / Higher-Risk Internet Customers

3,192

15.3 months

$67.88

44%

Segment 2

Basic / Non-Internet Customers

1,526

30.6 months

$21.08

7%

Segment 3

Long-Tenure / Higher-Value Customers

2,325

57.0 months

$89.15

15%

The segmentation model and preprocessing pipeline are saved as:

models/kmeans_customer_segmentation.pkl
models/clustering_preprocessor.pkl

Churn was not used as an input feature for clustering. It was used afterward for segment profiling and interpretation.

Streamlit Application

The project includes an interactive Streamlit application with two main pages.

Overview

The Overview page provides:

Customer population KPIs.

Churn analysis.

Customer filters.

Contract analysis.

Internet service analysis.

Payment method analysis.

Customer segmentation.

Dynamic segment profiles based on selected filters.

The segmentation displayed in the Overview page is generated using the saved K-Means model and clustering preprocessor.

Customer Prediction

The Customer Prediction page allows users to enter customer information and generate:

Churn Prediction

Churn probability.

Predicted churn status.

CLTV Prediction

Estimated Customer Lifetime Value.

Customer Segmentation

Predicted customer segment.

Segment description.

The app uses the saved model pipelines rather than retraining models during prediction.

Example Prediction Scenarios

The application was tested with two contrasting customer profiles.

New Customer

Tenure: 2 months
Monthly Charges: $90
Total Charges: $180
Internet Service: Fiber optic
Contract: Month-to-month

Observed application output:

Churn Risk: 84.7%
Predicted CLTV: $3,823
Segment: Segment 1

Long-Tenure Customer

Tenure: 60 months
Monthly Charges: $90
Total Charges: $5,400
Internet Service: Fiber optic
Contract: Two year

Observed application output:

Churn Risk: 6.1%
Predicted CLTV: $5,040
Segment: Segment 3

These examples demonstrate how different customer profiles produce different model-based predictions.

Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Telco-Customer-Intelligence

Create and activate a virtual environment:

Windows PowerShell

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run the Streamlit App

From the project root:

streamlit run app/app.py

The Streamlit application will open in your browser.

Running the command from the project root is important because the application uses project-relative paths for the processed data and saved models.

Deployment

The application can be deployed using Streamlit Community Cloud directly from GitHub.

Deployment requirements:

A GitHub repository containing the project.

requirements.txt in the repository root.

The saved models.

The processed dataset required by the Overview page.

app/app.py as the Streamlit entrypoint.

After pushing the project to GitHub:

Sign in to Streamlit Community Cloud.

Connect your GitHub account.

Create a new app.

Select the repository and branch.

Set the main file to:

app/app.py

Choose an app subdomain.

Deploy.

Streamlit Community Cloud installs the dependencies from requirements.txt and runs the selected entrypoint.

Business Questions Addressed

Which customers are more likely to churn?

How do contract and service characteristics relate to churn?

How does customer tenure relate to churn behavior?

Which customer profiles have higher estimated CLTV?

What customer segments exist based on service usage and customer characteristics?

How can customer-level predictions be exposed through an interactive application?

Key Project Takeaways

Churn prediction can be framed as an imbalanced binary classification problem.

Contract type, tenure, internet service, payment method, and monthly charges show notable relationships with observed churn in this dataset.

CLTV is substantially related to accumulated customer tenure and charges, while the final regression model provides moderate predictive performance.

K-Means identifies three interpretable customer segments with different tenure, spending, service, and observed churn profiles.

Combining supervised learning, regression, and clustering provides a broader customer intelligence workflow than using a single churn model.

The Streamlit application turns the trained models into an interactive decision-support interface.

Technologies

Python

Pandas

NumPy

Scikit-learn

XGBoost

Joblib

Plotly

Streamlit

Jupyter Notebook

Git / GitHub

Notes

Model predictions are estimates and are not guarantees of future customer behavior.

Observed relationships in the EDA and segment profiles do not establish causation.

Churn-related fields such as Churn Score and Churn Reason were excluded from the churn prediction features to reduce leakage risk.

The original raw dataset is kept separate from the processed dataset.
