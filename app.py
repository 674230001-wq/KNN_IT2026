
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==============================
# ตั้งค่าหน้าเว็บไซต์
# ==============================

st.set_page_config(
    page_title="Heart AI - KNN",
    page_icon="❤️",
    layout="wide"
)

# ==============================
# ตกแต่งเว็บไซต์
# ==============================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Prompt', sans-serif;
}

.stApp {
    background-color: #F3F7FB;
}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(120deg,#102B46,#155B83,#1596A8);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 32px;
}

.hero p {
    color: #DCECF5;
}

.panel {
    background: white;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 18px;
    border: 1px solid #E4EDF4;
}

.metric {
    background: white;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #E4EDF4;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
}

.metric-value {
    color: #123653;
    font-size: 26px;
    font-weight: bold;
}

.result-0 {
    background: #EAF8F0;
    border: 1px solid #A7E4C0;
    padding: 22px;
    border-radius: 15px;
    color: #12633B;
}

.result-1 {
    background: #FFF0F0;
    border: 1px solid #FFC4C4;
    padding: 22px;
    border-radius: 15px;
    color: #9F2525;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg,#1479A8,#1196AD);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    min-height: 45px;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# อ่านข้อมูล
# ==============================

@st.cache_data
def load_data():

    data = pd.read_csv("Heart3.csv")

    columns = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
        "HeartDisease"
    ]

    data = data[columns]

    return data


try:
    dt = load_data()

except Exception as e:
    st.error(f"ไม่สามารถอ่านไฟล์ Heart3.csv ได้: {e}")
    st.stop()


# ==============================
# แยกข้อมูลสำหรับ KNN
# ==============================

features = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope"
]

X = dt[features]
y = dt["HeartDisease"]


# ==============================
# เมนูด้านข้าง
# ==============================

st.sidebar.title("❤️ HEART AI")

st.sidebar.write("ระบบจำแนกข้อมูลโรคหัวใจ")

menu = st.sidebar.radio(
    "เลือกเมนู",
    [
        "หน้าหลัก",
        "ข้อมูลชุดข้อมูล",
        "ทำนายโรคหัวใจ",
        "ประเมินโมเดล",
        "เกี่ยวกับผู้จัดทำ"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("ตั้งค่า KNN")

k = st.sidebar.slider(
    "เลือกจำนวน K",
    min_value=1,
    max_value=25,
    value=5,
    step=2
)

test_percent = st.sidebar.slider(
    "สัดส่วนข้อมูลทดสอบ (%)",
    10,
    40,
    20,
    5
)


# ==============================
# สร้างและฝึกโมเดล
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_percent / 100,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

# โมเดลสำหรับทำนายข้อมูลใหม่ ฝึกด้วยข้อมูลทั้งหมด
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k))
])

final_model.fit(X, y)


# ==============================
# หน้าที่ 1 : หน้าหลัก
# ==============================

if menu == "หน้าหลัก":

    st.markdown("""
    <div class="hero">
        <h1>❤️ ระบบจำแนกข้อมูลโรคหัวใจ</h1>
        <p>
        Heart Disease Classification
        using K-Nearest Neighbors (KNN)
        </p>
    </div>
    """, unsafe_allow_html=True)

    total = len(dt)

    count0 = (dt["HeartDisease"] == 0).sum()

    count1 = (dt["HeartDisease"] == 1).sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric">
        <div class="metric-label">ข้อมูลทั้งหมด</div>
        <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric">
        <div class="metric-label">กลุ่ม 0</div>
        <div class="metric-value">{count0}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric">
        <div class="metric-label">กลุ่ม 1</div>
        <div class="metric-value">{count1}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">{accuracy*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # กราฟสัดส่วนข้อมูล

    st.markdown("""
    <div class="panel">
    <h3 style="text-align:center">
    สถิติข้อมูลโรคหัวใจ
    </h3>
    </div>
    """, unsafe_allow_html=True)

    counts = dt["HeartDisease"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    labels = [
        "ไม่พบโรคหัวใจ",
        "พบโรคหัวใจ"
    ]

    values = [
        int(counts[0]),
        int(counts[1])
    ]

    colors = [
        "#2878B5",
        "#F28E2B"
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                sort=False,
                marker=dict(
                    colors=colors,
                    line=dict(
                        color="white",
                        width=3
                    )
                ),
                textinfo="percent",
                textposition="inside",
                textfont=dict(
                    size=18,
                    color="white"
                ),
                hovertemplate=
                "<b>%{label}</b><br>" +
                "จำนวน %{value} รายการ<br>" +
                "สัดส่วน %{percent}<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title="HeartDisease",
        title_x=0.5,
        height=450,
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.1,
            x=0.5,
            xanchor="center"
        ),
        annotations=[
            dict(
                text=f"{total}<br>ข้อมูลทั้งหมด",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    size=18,
                    color="#173B57"
                )
            )
        ],
        margin=dict(
            t=60,
            b=60
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            f"ไม่พบโรคหัวใจ: {values[0]} รายการ "
            f"({values[0]/total*100:.1f}%)"
        )

    with col2:
        st.error(
            f"พบโรคหัวใจ: {values[1]} รายการ "
            f"({values[1]/total*100:.1f}%)"
        )

    # แสดงกราฟข้อมูล

    st.markdown(
        '<div class="panel"><h3>กราฟข้อมูลสุขภาพ</h3></div>',
        unsafe_allow_html=True
    )

    chart_data = dt[
        ["Age", "Cholesterol", "RestingBP", "MaxHR"]
    ].sum()

    st.bar_chart(chart_data)

    # แสดงตัวอย่างข้อมูล

    st.markdown(
        '<div class="panel"><h3>ตัวอย่างข้อมูล</h3></div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        dt.head(10),
        use_container_width=True
    )


# ==============================
# หน้าที่ 2 : ข้อมูลชุดข้อมูล
# ==============================

elif menu == "ข้อมูลชุดข้อมูล":

    st.title("📊 ข้อมูลชุดข้อมูล Heart3.csv")

    st.write("ข้อมูลตัวอย่าง 10 รายการ")

    st.dataframe(
        dt.head(10),
        use_container_width=True
    )

    st.write("สถิติเบื้องต้นของข้อมูล")

    st.dataframe(
        dt.describe(),
        use_container_width=True
    )

    st.write("จำนวนข้อมูลแต่ละคอลัมน์")

    st.write(dt.count())


# ==============================
# หน้าที่ 3 : ทำนายข้อมูลใหม่
# ==============================

elif menu == "ทำนายโรคหัวใจ":

    st.markdown("""
    <div class="hero">
        <h1>🩺 ทำนายข้อมูลโรคหัวใจ</h1>
        <p>กรอกข้อมูลใหม่เพื่อให้ระบบ KNN จำแนกข้อมูล</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "กรอกข้อมูลทั้ง 11 ตัวแปร โดยตัวแปรหมวดหมู่ใช้รหัสตัวเลขตามไฟล์ CSV"
    )

    with st.form("prediction_form"):

        st.subheader("1. ข้อมูลทั่วไป")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider(
                "Age (อายุ)",
                28, 77, 50
            )

        with col2:
            sex = st.selectbox(
                "Sex (รหัส)",
                [0, 1]
            )

        with col3:
            chest = st.selectbox(
                "ChestPainType (รหัส)",
                [1, 2, 3, 4]
            )

        st.divider()

        st.subheader("2. ข้อมูลสุขภาพ")

        col1, col2, col3 = st.columns(3)

        with col1:
            bp = st.number_input(
                "RestingBP",
                0, 200, 120
            )

        with col2:
            cholesterol = st.number_input(
                "Cholesterol",
                0, 603, 200
            )

        with col3:
            fasting = st.selectbox(
                "FastingBS (รหัส)",
                [0, 1]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            ecg = st.selectbox(
                "RestingECG (รหัส)",
                [1, 2, 3]
            )

        with col2:
            maxhr = st.number_input(
                "MaxHR",
                60, 202, 140
            )

        with col3:
            angina = st.selectbox(
                "ExerciseAngina (รหัส)",
                [0, 1]
            )

        st.divider()

        st.subheader("3. ผลตรวจเพิ่มเติม")

        col1, col2 = st.columns(2)

        with col1:
            oldpeak = st.number_input(
                "Oldpeak",
                -2.6, 6.2,
                0.0,
                step=0.1
            )

        with col2:
            slope = st.selectbox(
                "ST_Slope (รหัส)",
                [1, 2, 3]
            )

        submit = st.form_submit_button(
            "🔍 ทำนายผล",
            use_container_width=True
        )

    if submit:

        x_input = pd.DataFrame(
            [[
                age,
                sex,
                chest,
                bp,
                cholesterol,
                fasting,
                ecg,
                maxhr,
                angina,
                oldpeak,
                slope
            ]],
            columns=features
        )

        out = final_model.predict(x_input)

        probabilities = final_model.predict_proba(x_input)[0]

        if out[0] == 0:

            st.markdown("""
            <div class="result-0">
                <h2>ผลการทำนาย : กลุ่ม 0</h2>
                <h3>ไม่พบโรคหัวใจตามป้ายกำกับข้อมูล</h3>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-1">
                <h2>ผลการทำนาย : กลุ่ม 1</h2>
                <h3>พบโรคหัวใจตามป้ายกำกับข้อมูล</h3>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "คะแนนกลุ่ม 0",
                f"{probabilities[0]*100:.1f}%"
            )

        with col2:
            st.metric(
                "คะแนนกลุ่ม 1",
                f"{probabilities[1]*100:.1f}%"
            )

        st.write("ข้อมูลที่ป้อน")

        st.dataframe(
            x_input,
            use_container_width=True
        )

        st.warning(
            "ผลการทำนายนี้ใช้เพื่อการศึกษาเท่านั้น "
            "ไม่ใช่การวินิจฉัยโรคทางการแพทย์"
        )


# ==============================
# หน้าที่ 4 : ประเมินโมเดล
# ==============================

elif menu == "ประเมินโมเดล":

    st.title("📈 ประเมินประสิทธิภาพโมเดล KNN")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision*100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall*100:.2f}%"
        )

    with col4:
        st.metric(
            "F1-score",
            f"{f1*100:.2f}%"
        )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["ค่าจริง 0", "ค่าจริง 1"],
        columns=["ทำนาย 0", "ทำนาย 1"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["กลุ่ม 0", "กลุ่ม 1"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True
    )

    st.write("จำนวนข้อมูลฝึก:", len(X_train))
    st.write("จำนวนข้อมูลทดสอบ:", len(X_test))
    st.write("ค่า K ที่ใช้:", k)


# ==============================
# หน้าที่ 5 : เกี่ยวกับผู้จัดทำ
# ==============================

elif menu == "เกี่ยวกับผู้จัดทำ":

    st.title("👨‍💻 เกี่ยวกับผู้จัดทำ")

    col1, col2 = st.columns([1, 2])

    with col1:

        if os.path.exists("./img/ll.png"):

            st.image(
                "./img/ll.png",
                width=250
            )

        else:

            st.warning("ไม่พบรูปภาพ ll.png")

    with col2:

        st.subheader("ผู้จัดทำโครงงาน")

        st.write("ชื่อ: รพีภัทร")

        st.write(
            "หัวข้อโครงงาน: ระบบจำแนกข้อมูลโรคหัวใจ"
        )

        st.write(
            "เทคนิคที่ใช้: K-Nearest Neighbors (KNN)"
        )

        st.write(
            "เครื่องมือที่ใช้: Python และ Streamlit"
        )

    st.divider()

    st.subheader("หลักการทำงานของ KNN")

    st.write("""
    K-Nearest Neighbors (KNN) เป็นเทคนิค Machine Learning
    ที่ใช้จำแนกข้อมูลโดยคำนวณระยะห่างระหว่างข้อมูลใหม่
    กับข้อมูลที่มีอยู่ในชุดฝึก

    จากนั้นเลือกข้อมูลที่อยู่ใกล้ที่สุดตามจำนวน K
    และใช้เสียงข้างมากในการตัดสินผลการจำแนก
    """)

    st.warning(
        "เว็บไซต์นี้จัดทำขึ้นเพื่อการศึกษา "
        "ไม่ใช่เครื่องมือวินิจฉัยโรคทางการแพทย์"
    )
