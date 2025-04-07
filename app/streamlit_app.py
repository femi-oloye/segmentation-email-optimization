import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load data and model
df = pd.read_csv('/home/oluwafemi/segmentation-email-optimization/seg-email-opt-env/segmentation-email-optimization/data/clean_marketing_data.csv')
model = joblib.load('/home/oluwafemi/segmentation-email-optimization/seg-email-opt-env/segmentation-email-optimization/model/conversion_model.pkl')  # We'll save this shortly



# ----- PAGE SETUP -----
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 Customer Segmentation & Conversion Prediction")

# ----- SUMMARY STATS -----
st.header("1. Summary Statistics")
st.dataframe(df.describe())


# ----- SEGMENTATION PLOT -----
st.header("2. Cluster Visualization")

fig, ax = plt.subplots()
sns.scatterplot(
    x='TimeOnSite',
    y='PurchaseAmount',
    hue='SegmentName',
    data=df,
    palette='Set2',
    ax=ax
)
st.pyplot(fig)

# ----- CONVERSION PREDICTION -----
st.header("3. Conversion Prediction")

with st.form("conversion_form"):
    st.subheader("Enter Customer Info")

    time_on_site = st.slider("Time on Site (minutes)", 0, 20, 5)
    pages_visited = st.slider("Pages Visited", 0, 20, 4)
    email_clicks = st.slider("Email Clicks", 0, 10, 1)
    purchase_amount = st.number_input("Previous Purchase Amount", 0.0, 500.0, 0.0)
    
    engagement_rate = email_clicks / (pages_visited + 1)
    value_per_page = purchase_amount / (pages_visited + 1)

    submitted = st.form_submit_button("Predict")

    if submitted:
        X_input = pd.DataFrame([[
            time_on_site,
            pages_visited,
            email_clicks,
            purchase_amount,
            engagement_rate,
            value_per_page
        ]], columns=[
            'TimeOnSite', 'PagesVisited', 'EmailClicks', 'PurchaseAmount',
            'EngagementRate', 'ValuePerPage'
        ])
        
        prediction = model.predict(X_input)[0]
        result = "✅ Will Convert" if prediction == 1 else "❌ Will Not Convert"
        st.success(f"Prediction: {result}")
