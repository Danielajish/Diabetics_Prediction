import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
st.title('Web Deployment of Medical Diagnostic app 😊💉')

st.subheader('Is the person diabetic ?')
df = pd.read_csv('diabetes.csv')
st.set_option('deprecation.showPyplotGlobalUse', False)


if st.sidebar.checkbox('View Data',False):
    st.write(df)
    
if st.sidebar.checkbox('View Distributions',False):
    j =1
    for i in df.select_dtypes(include='number').columns:
        plt.subplot(3,3,j)
        sns.distplot(df[i],kde =True)
        j +=1
    plt.tight_layout()
    st.pyplot()

    #load pickle
model = open('rfc.pickle','rb')
clf = pickle.load(model)
model.close()


## get front end user name 

preg = st.number_input('Pregnancies',0,17,0)
glucose = st.slider('Glucose',df['Glucose'].min(),df['Glucose'].max(),df['Glucose'].min())
blp = st.slider('BloodPressure',df['BloodPressure'].min(),df['BloodPressure'].max(),df['BloodPressure'].min()) 
skin = st.slider('SkinThickness',df['SkinThickness'].min(),df['SkinThickness'].max(),df['SkinThickness'].min()) 
insl = st.slider('Insulin',df['SkinThickness'].min(),df['SkinThickness'].max(),df['SkinThickness'].min())
bmi = st.slider('BMI',df['BMI'].min(),df['BMI'].max(),df['BMI'].min())
dia = st.slider('DiabetesPedigreeFunction',df['DiabetesPedigreeFunction'].min(),df['DiabetesPedigreeFunction'].max(),df['DiabetesPedigreeFunction'].min())
age = st.slider('Age',df['Age'].min(),df['Age'].max(),21)
x = pd.DataFrame({0:preg,1:glucose,2:blp,3:skin,4:insl,5:bmi,6:dia,7:age},index = [0])

pred = clf.predict(x)
if pred == 1:
    pred = 'Diabetic dead'
else:
    pred = 'Non-Diabetic'
if st.button('Predict'):
    st.subheader(pred)


