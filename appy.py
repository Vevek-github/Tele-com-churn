import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import telecom_model

with st.sidebar:
    st.image(r"social-network-connection-avatar-icon-vector.jpg")
    st.title("Customer Churn Predictor")
    st.info("This is an awesome web app , helps to understand the Customer churn analysis in the Tele-com Industry.")
    #choice = st.radio("Navigation",["Upload as Batch","Upload as Element","Result"])



tab1 , tab2 ,tab3= st.tabs(["Tab1","Tab2","Tab3"])

with tab1:
    st.write("lets start")
    st.title("For Batch Prediction")
    file = st.file_uploader("Upload your 'csv' file here.")
    if file:
        df =pd.read_csv(file,index_col=None)
        st.dataframe(df)
        df.to_csv(r"current_data_batch_csv.csv",index=None)



with tab2:
    st.title("Element Churn Prediction")

    # Create empty dictionary to store user input
    user_data = {
        'state': st.selectbox("State",['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI',
       'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC',
       'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR',
       'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM', 'NV', 'DC',
       'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND']),
        'area.code': st.selectbox("Area Code",['area_code_415', 'area_code_408', 'area_code_510']),
        'intl.plan': st.selectbox("International Plan", ['yes', 'no']),
        'voice.plan': st.selectbox("Voice Plan", ['yes', 'no']),
        'account.length': st.text_input("Account Length"),
        'voice.messages': st.text_input("Number of Voicemail Messages"),
        'intl.mins': st.text_input("International Minutes Used"),
        'intl.calls': st.text_input("Total Number of International Calls"),
        'intl.charge': st.text_input("Total International Charge"),
        'day.mins': st.text_input("Day Minutes Used"),
        'day.calls': st.text_input("Total Number of Calls During the Day"),
        'day.charge': st.text_input("Total Charge During the Day"),
        'eve.mins': st.text_input("Evening Minutes Used"),
        'eve.calls': st.text_input("Total Number of Calls During the Evening"),
        'eve.charge': st.text_input("Total Charge During the Evening"),
        'night.mins': st.text_input("Night Minutes Used"),
        'night.calls': st.text_input("Total Number of Calls During the Night"),
        'night.charge': st.text_input("Total Charge During the Night"),
        'customer.calls': st.text_input("Number of Calls to Customer Service")
    }

    # Display the user input
    st.write("## User Input Data")
    df_user = pd.DataFrame(user_data, index=[0])
    st.write(df_user)
    df_user.to_csv(r"current_data.csv")




with tab3:
    st.subheader("Choose CSV or Element wise Prediction")
    df_user = pd.read_csv(r"current_data.csv",index_col=None)
    df= pd.read_csv(r"current_data_batch_csv.csv",index_col=None)
    Batch_vs_element =st.radio("CSV or Element",["CSV","Element"])
    if Batch_vs_element == "CSV":
        df=df.copy()
    else:
        df = df_user.copy()
    
    st.write(df)
    predict_button = st.button("Predict the Churn")
    if predict_button:
        hello =telecom_model.input_run(df)
        hd = pd.DataFrame({"Customer_status" : hello})
        hd['Customer_status']=hd['Customer_status'].map({0:"Loyal",1 :"Churn"})
        st.title("Comparision of the scores for the Trained Models")
        
        col1 ,col2 = st.columns(2)
        col1=st.write(hd)
        col2 = st.image("EDA images/Model_comparision.png")
        
        
    Model_importances = st.button("Feature Importances of the Model")
    if Model_importances:
        col4 ,col3 = st.columns(2)
        col3 = st.image(r"EDA images/corr_x_train.png")
        col4 = st.image(r"EDA images/fearture_importance_gbm.png")
        

        
        


    
