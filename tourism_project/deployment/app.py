import json
import os

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_tourism_model_v1.joblib")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = 0.5
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            threshold = json.load(f).get("threshold", 0.5)
    return model, threshold


if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run the GitHub Actions pipeline first.")
    st.stop()

model, threshold = load_model()

st.title("Wellness Tourism Package Predictor")
st.write(
    "Visit with Us — predict whether a customer is likely to purchase the "
    "Wellness Tourism Package before the sales team contacts them."
)

st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0.0, value=23000.0, step=500.0
    )
    city_tier = st.selectbox("City Tier", [1, 2, 3])
with col2:
    num_persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    num_children = st.number_input("Number of Children Visiting (under 5)", 0, 5, 1)
    num_trips = st.number_input("Average Trips per Year", 0, 30, 3)
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    passport = st.radio("Has Passport?", ["Yes", "No"], horizontal=True)
    own_car = st.radio("Owns Car?", ["Yes", "No"], horizontal=True)

st.subheader("Interaction details")
col3, col4 = st.columns(2)
with col3:
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
with col4:
    duration = st.number_input("Duration of Pitch (minutes)", 0, 200, 15)
    followups = st.number_input("Number of Follow-ups", 0, 10, 3)

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict", type="primary"):
    prob = float(model.predict_proba(input_df)[:, 1][0])
    if prob >= threshold:
        st.success(f"Likely to purchase the Wellness Package (probability {prob:.0%}).")
    else:
        st.warning(f"Unlikely to purchase the Wellness Package (probability {prob:.0%}).")
    with st.expander("Input sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}).astype(str))
