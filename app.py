import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("decision-model.pkl","rb")
model1=pickle.load(pickle_in)
dataset= pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [1,2, 3]].values
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(UserID, Gender,Age,EstimatedSalary):
  output= model1.predict(sc.transform([[Gender,Age,EstimatedSalary]]))
  print("Purchased", output)
  if output==[1]:
    prediction="Item will be purchased"
  else:
    prediction="Item will not be purchased"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Internship Project Deployment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Item Purchase Prediction")
    UserID = st.text_input("UserID","")
    Gender = st.selectbox('Enter Gender',('Male','Female'))
    if Gender == 'Male':
         Gender = 1
    else:
         Gender = 0
    Age = st.number_input("Insert Age",18,60)
    EstimatedSalary = st.number_input("Insert salary",15000,150000)
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(UserID, Gender,Age,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Shakti Sarada Prasad")
      st.subheader("Student , Department of Computer Engineering")

if __name__=='__main__':
  main()