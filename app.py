import os,joblib,pandas as pd,streamlit as st
from huggingface_hub import hf_hub_download
path=hf_hub_download(repo_id=os.environ["HF_MODEL_REPO"],filename="best_model.joblib",token=os.getenv("HF_TOKEN"))
model=joblib.load(path)
st.title("Tourism Package Purchase Predictor")
st.caption("Predict whether a customer is likely to purchase the offered tourism package.")
fields={"Age":st.number_input("Age",18,100,35),"TypeofContact":st.selectbox("Type of Contact",["Self Enquiry","Company Invited"]),"CityTier":st.selectbox("City Tier",[1,2,3]),"DurationOfPitch":st.number_input("Duration of Pitch",1.0,120.0,15.0),"Occupation":st.selectbox("Occupation",["Salaried","Small Business","Large Business","Free Lancer"]),"Gender":st.selectbox("Gender",["Male","Female"]),"NumberOfPersonVisiting":st.number_input("Persons Visiting",1,20,2),"NumberOfFollowups":st.number_input("Followups",0.0,20.0,3.0),"ProductPitched":st.selectbox("Product Pitched",["Basic","Deluxe","Standard","Super Deluxe","King"]),"PreferredPropertyStar":st.selectbox("Preferred Property Star",[3.0,4.0,5.0]),"MaritalStatus":st.selectbox("Marital Status",["Married","Divorced","Unmarried","Single"]),"NumberOfTrips":st.number_input("Number of Trips",0.0,30.0,2.0),"Passport":st.selectbox("Passport",[0,1]),"PitchSatisfactionScore":st.selectbox("Pitch Satisfaction",[1,2,3,4,5]),"OwnCar":st.selectbox("Own Car",[0,1]),"NumberOfChildrenVisiting":st.number_input("Children Visiting",0.0,10.0,0.0),"Designation":st.selectbox("Designation",["Executive","Manager","Senior Manager","AVP","VP"]),"MonthlyIncome":st.number_input("Monthly Income",0.0,1000000.0,22000.0)}
if st.button("Predict"):
 p=float(model.predict_proba(pd.DataFrame([fields]))[0,1]); st.metric("Purchase probability",f"{p:.1%}")
