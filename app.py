import streamlit as st
import numpy as np
import pickle
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load models
with open("reg_pro.pkl", "rb") as f:
    reg_model = pickle.load(f)

with open("clf_pro.pkl", "rb") as f:
    clf_model = pickle.load(f)

# Load environment variables and configure Gemini LLM
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')    

# Helper mappings
yes_no_map = {'Yes': 1, 'No': 0}
furnish_map = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}
category_map = {0: "Affordable", 1: "Expensive", 2: "Very Expensive"}

# Set page configuration
st.set_page_config(page_title="🏠 Smart Property Advisor", layout="wide")

# ------------------------
# Custom CSS for Main UI Styling
# ------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 30px;
        font-weight: bold;
        color: #33ffe3;
        text-align: center;
        margin-bottom: 10px;
    }
    .problem-statement {
        font-size: 18px;
        font-weight: 500;
        color: #ffffff;
        text-align: center;
        margin-bottom: 30px;
    }
    .form-container {
        border: 2px solid #2E8B57;
        border-radius: 10px;
        padding: 20px;
        background-color: #F9FFF9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# Sidebar Content
# ------------------------
st.sidebar.title("🚀 What Is This System?")
about_text = """
**🏠 EstateMentor: A Property Guide**

Property Intelligence Suite is a smart real estate advisory system that combines 
machine learning and generative AI to assist users in making informed property 
decisions. \n
It predicts property prices using a regression model, evaluates affordability with a 
classification model based on financial inputs, and delivers personalized investment 
insights and improvement tips through a generative AI model.
"""

st.sidebar.info(about_text)

st.sidebar.markdown("## 🌟 Key Features")
st.sidebar.markdown("- 🏷️ **Price Prediction** using regression")  
st.sidebar.markdown("- 🧮 **Affordability Analysis** via classification")  
st.sidebar.markdown("- 🤖 **LLM-Based Insights** powered by Google Gemini")  
st.sidebar.markdown("- 💻 **Modern UI** with an intuitive Streamlit interface")

st.sidebar.markdown("## 👤 Creator")
st.sidebar.markdown("- **Shailesh Patil**")

st.sidebar.markdown("## ⚙️ Tech Stack")
st.sidebar.markdown("- Streamlit  \n- Machine Learning  \n- Generative-AI  \n- Python")


# ------------------------
# Main Header
# ------------------------
st.markdown('<div class="main-header">🏠 EstateMentor: A Property Intelligence Guide</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="problem-statement">Quickly predict your property’s price and assess its affordability with our AI-powered system!</div>',
    unsafe_allow_html=True,
)

# ------------------------
# Input Form
# ------------------------
st.markdown('<div class="form-container">', unsafe_allow_html=True)

with st.form("property_form"):
    st.subheader("Enter Property & Financial Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.number_input("📏 Area (sqft)", min_value=500, max_value=10000, value=3000)
        bedrooms = st.selectbox("🛏️ Bedrooms", [1, 2, 3, 4, 5], index=2)
        bathrooms = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4], index=1)
        stories = st.selectbox("🏢 Stories", [1, 2, 3, 4], index=1)
    
    with col2:
        mainroad = st.selectbox("🚗 Facing Main Road?", ['Yes', 'No'])
        guestroom = st.selectbox("🛋️ Guest Room?", ['Yes', 'No'])
        basement = st.selectbox("🏗️ Basement?", ['Yes', 'No'])
        airconditioning = st.selectbox("❄️ Air Conditioning?", ['Yes', 'No'])
    
    with col3:
        parking = st.slider("🚙 Parking Spaces", 0, 4, 1)
        furnishingstatus = st.selectbox("🪑 Furnishing", ['Unfurnished', 'Semi-Furnished', 'Furnished'])
        annual_income = st.number_input("💰 Annual Income (₹)", min_value=100000, value=5000000)
        savings = st.number_input("🏦 Total Savings (₹)", min_value=50000, value=1000000)
        ownership = st.selectbox("🏘️ Own another property?", ['Yes', 'No'])
    
    submitted = st.form_submit_button("🔍 Predict")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------
# Prediction & LLM Feedback
# ------------------------
if submitted:
    # Prepare regression input
    reg_input = np.array([[ 
        area, bedrooms, bathrooms, stories,
        yes_no_map[mainroad], yes_no_map[guestroom], yes_no_map[basement],
        yes_no_map[airconditioning], parking, furnish_map[furnishingstatus]
    ]])
    
    predicted_price = reg_model.predict(reg_input)[0]
    
    annual_revenue = annual_income
    total_savings = savings
    sum_of_revenue_and_savings = annual_revenue + total_savings
    owns_any_property = yes_no_map[ownership]
    
    clf_input = np.array([[ 
        annual_revenue, total_savings, sum_of_revenue_and_savings, owns_any_property, predicted_price
    ]])
    
    category = clf_model.predict(clf_input)[0]
    category_label = category_map.get(category, "Unknown")
    
    # Display model outputs
    st.markdown("<hr>", unsafe_allow_html=True)
    st.success(f"🏷️ **Predicted Property Price:** ₹{predicted_price:,.2f}")
    st.info(f"📊 **Affordability Category:** {category_label}")

    # Gemini LLM Prompt
    prompt = f"""
        You are a real estate investment advisor.

        Below are the user-provided details:

        🏠 Property Details:
        - Area: {area} sqft
        - Bedrooms: {bedrooms}
        - Bathrooms: {bathrooms}
        - Stories: {stories}
        - Facing Main Road: {mainroad}
        - Guest Room: {guestroom}
        - Basement: {basement}
        - Air Conditioning: {airconditioning}
        - Parking Spaces: {parking}
        - Furnishing: {furnishingstatus}

        💰 Financial Details:
        - Annual Income: ₹{annual_income:,.2f}
        - Total Savings: ₹{savings:,.2f}
        - Owns Another Property: {ownership}

        🔍 Model Predictions:
        - Predicted Property Price: ₹{predicted_price:,.2f}
        - Affordability Category: {category_label}

        Based on the above, provide a response in the following format and don't give any other text:

        📈 Investment Insight:
        (Brief analysis on whether this property seems like a good investment and why in points.)

        💸 Affordability Advice:
        (Advice tailored to the user's financial situation and affordability category in points..)

        🛠️ Improvement Suggestions:
        (Any suggestions to improve property value or buyer readiness in points..)

        Keep the tone professional, friendly, and helpful.
    """

    llm_response = model.generate_content(prompt)
    
    st.markdown("### 🤖 LLM Suggestions & Feedback")
    st.markdown(llm_response.text, unsafe_allow_html=True)
