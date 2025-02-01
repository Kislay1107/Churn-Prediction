import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import streamlit as st 

model = tf.keras.models.load_model('ANN.keras')

with open('one_hot_encoder.pkl', 'rb') as file:
    saved_columns = pickle.load(file)
    
with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
    
st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', ['Germany', 'France', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92, 21)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# print(geography, gender, age, balance, credit_score, estimated_salary, tenure, num_of_products, has_cr_card, is_active_member)

if geography == 'Spain':
    Geography_Germany = 0
    Geography_Spain = 1
elif geography == 'Germany':
    Geography_Germany = 1
    Geography_Spain = 0
elif geography == 'France':
    Geography_Germany = 0
    Geography_Spain = 0

if gender == "Female":
    Gender_Male = 0
else:
    Gender_Male = 1

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography_Germany' : [Geography_Germany], 
    'Geography_Spain' : [Geography_Spain],
    'Gender_Male': [Gender_Male],

})

# print(input_data)

scaled_test = scaler.transform(input_data)
prediction = (model.predict(scaled_test) > 0.5).astype(int)

if prediction == 0:
    st.write("The customer will not churn")
else:
    st.write("The customer will churn")
    
    