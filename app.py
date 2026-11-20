
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกข้อมูลโรคหัวใจ")

st.image("./img/ll.png", width=250)

col1, col2, col3 = st.columns(3)

with col1:
   st.header("ข้อมูลผู้ป่วย")
   st.write("Age")
   st.write("Sex")
   st.write("ChestPainType")

with col2:
   st.header("ข้อมูลสุขภาพ")
   st.write("RestingBP")
   st.write("Cholesterol")
   st.write("FastingBS")

with col3:
   st.header("ผลการตรวจ")
   st.write("RestingECG")
   st.write("MaxHR")
   st.write("ExerciseAngina")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>สถิติข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./Heart3.csv")
st.write(dt.head(10))

dt1 = dt['Age'].sum()
dt2 = dt['Cholesterol'].sum()
dt3 = dt['RestingBP'].sum()
dt4 = dt['MaxHR'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["Age", "Cholesterol", "RestingBP", "MaxHR"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.bar_chart(dx2)
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>ทำนายข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

age = st.slider("กรุณาเลือกข้อมูล Age", 1, 100, 50)
sex = st.selectbox("กรุณาเลือกข้อมูล Sex", [0, 1])

chest = st.selectbox("กรุณาเลือกข้อมูล ChestPainType", [1, 2, 3, 4])

bp = st.number_input("กรุณาเลือกข้อมูล RestingBP", 0, 300, 120)
chol = st.number_input("กรุณาเลือกข้อมูล Cholesterol", 0, 1000, 200)

fasting = st.selectbox("กรุณาเลือกข้อมูล FastingBS", [0, 1])
ecg = st.selectbox("กรุณาเลือกข้อมูล RestingECG", [0, 1, 2])

maxhr = st.number_input("กรุณาเลือกข้อมูล MaxHR", 1, 300, 150)

angina = st.selectbox("กรุณาเลือกข้อมูล ExerciseAngina", [0, 1])

oldpeak = st.number_input("กรุณาเลือกข้อมูล Oldpeak", -10.0, 20.0, 0.0)

slope = st.selectbox("กรุณาเลือกข้อมูล ST_Slope", [0, 1, 2])

if st.button("ทำนายผล"):

   X = dt.drop('HeartDisease', axis=1)
   y = dt['HeartDisease']

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)

   x_input = np.array([[
       age, sex, chest, bp, chol,
       fasting, ecg, maxhr, angina,
       oldpeak, slope
   ]])

   st.write(Knn_model.predict(x_input))

   out = Knn_model.predict(x_input)

   if out[0] == 0:
      st.success("ผลการทำนาย: ไม่พบโรคหัวใจ")
   else:
      st.error("ผลการทำนาย: พบโรคหัวใจ")

else:
    st.write("ไม่ทำนาย")
