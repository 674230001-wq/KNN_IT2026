import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------- ตั้งค่าหน้าเว็บไซต์ ----------------
st.set_page_config(
    page_title="Heart AI | KNN Classification",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
.stApp { background: #f3f7fb; }
.block-container { max-width: 1400px; padding-top: 1.6rem; padding-bottom: 3rem; }
.hero {
    background: linear-gradient(120deg, #102b46, #155b83 65%, #1596a8);
    padding: 30px 34px; border-radius: 22px; color: white; margin-bottom: 22px;
    box-shadow: 0 10px 28px rgba(16,43,70,.13);
}
.hero h1 { color: white; font-size: 2rem; margin: 0 0 8px 0; }
.hero p { color: #dcecf5; margin: 0; font-size: 1rem; }
.metric {
    background: white; border: 1px solid #e4edf4; border-radius: 17px;
    padding: 17px 20px; min-height: 105px; box-shadow: 0 4px 14px rgba(16,43,70,.04);
}
.metric-label { color: #64748b; font-size: .88rem; }
.metric-value { color: #123653; font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
.panel {
    background: white; border: 1px solid #e4edf4; border-radius: 18px;
    padding: 21px; margin-bottom: 16px;
}
.result-0 {
    background: #eaf8f0; border: 1px solid #a7e4c0; color: #12633b;
    padding: 22px; border-radius: 17px;
}
.result-1 {
    background: #fff0f0; border: 1px solid #ffc4c4; color: #9f2525;
    padding: 22px; border-radius: 17px;
}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button {
    background: linear-gradient(90deg,#1479a8,#1196ad); color: white;
    border: 0; border-radius: 11px; font-weight: 600; min-height: 2.8rem;
}
[data-testid="stSidebar"] { background: #102b46; }
[data-testid="stSidebar"] * { color: #f3f8fc; }
.note { color: #64748b; font-size: .86rem; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]
TARGET = "HeartDisease"
COLUMNS = FEATURES + [TARGET]

@st.cache_data
def read_dataset(uploaded_file=None):
    if uploaded_file is not None:
        frame = pd.read_csv(uploaded_file)
    else:
        frame = pd.read_csv("Heart3.csv")

    missing_cols = [name for name in COLUMNS if name not in frame.columns]
    if missing_cols:
        raise ValueError("ไม่พบคอลัมน์: " + ", ".join(missing_cols))

    frame = frame[COLUMNS].copy()
    for name in COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    if frame.isna().any().any():
        raise ValueError("พบค่าว่างหรือค่าที่ไม่ใช่ตัวเลขในคอลัมน์ที่ใช้ กรุณาตรวจสอบไฟล์ CSV")

    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("HeartDisease ต้องมีค่าเป็น 0 หรือ 1 เท่านั้น")

    frame[TARGET] = frame[TARGET].astype(int)
    return frame

def create_knn(k):
    # ปรับสเกลก่อนคำนวณระยะทาง เพราะตัวแปรมีหน่วยและช่วงค่าต่างกัน
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

def evaluate_and_fit(frame, k, test_ratio):
    X = frame[FEATURES]
    y = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    test_model = create_knn(k)
    test_model.fit(x_train, y_train)
    predicted = test_model.predict(x_test)

    result = {
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "matrix": confusion_matrix(y_test, predicted, labels=[0, 1]),
        "y_test": y_test,
        "predicted": predicted,
        "train_count": len(x_train),
        "test_count": len(x_test)
    }

    # โมเดลสำหรับทำนายข้อมูลใหม่ ฝึกด้วยข้อมูลทั้งหมด
    final_model = create_knn(k)
    final_model.fit(X, y)
    return final_model, result

def metric_card(label, value):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )

# ---------------- เมนูด้านข้าง ----------------
st.sidebar.markdown("## ❤️ HEART AI")
st.sidebar.caption("ระบบจำแนกข้อมูลโรคหัวใจด้วย KNN")
page = st.sidebar.radio(
    "เลือกเมนู",
    ["ภาพรวมข้อมูล", "ทำนายข้อมูลใหม่", "ประเมินโมเดล", "เกี่ยวกับระบบ"]
)
st.sidebar.divider()
uploaded = st.sidebar.file_uploader("อัปโหลดชุดข้อมูล CSV", type=["csv"])
st.sidebar.caption("หากไม่อัปโหลด ระบบจะอ่าน Heart3.csv จากโฟลเดอร์โปรเจกต์")

try:
    data = read_dataset(uploaded)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ Heart3.csv กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกับ app.py")
    st.stop()
except Exception as error:
    st.error(f"ไม่สามารถอ่านข้อมูลได้: {error}")
    st.stop()

if len(data) < 10:
    st.error("ชุดข้อมูลมีจำนวนน้อยเกินไปสำหรับแบ่ง Train/Test")
    st.stop()

st.sidebar.markdown("### ตั้งค่า KNN")
max_k = min(25, max(1, len(data) - 2))
k_value = st.sidebar.slider("จำนวนเพื่อนบ้าน (K)", 1, max_k, min(5, max_k), 1)
test_percent = st.sidebar.slider("ข้อมูลทดสอบ (%)", 10, 40, 20, 5)

model, metrics = evaluate_and_fit(data, k_value, test_percent / 100)

# ---------------- เมนูภาพรวม ----------------
if page == "ภาพรวมข้อมูล":
    st.markdown("""
    <div class="hero">
      <h1>❤️ ระบบจำแนกข้อมูลโรคหัวใจ</h1>
      <p>Heart Disease Classification using K-Nearest Neighbors (KNN)</p>
    </div>
    """, unsafe_allow_html=True)

    total = len(data)
    count_0 = int((data[TARGET] == 0).sum())
    count_1 = int((data[TARGET] == 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("ข้อมูลทั้งหมด", f"{total:,} รายการ")
    with c2: metric_card("HeartDisease = 0", f"{count_0:,} รายการ")
    with c3: metric_card("HeartDisease = 1", f"{count_1:,} รายการ")
    with c4: metric_card("Accuracy", f"{metrics['accuracy']*100:.2f}%")

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("สัดส่วนกลุ่มข้อมูล")
        counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.3))
        ax.pie(
            counts.values, labels=["กลุ่ม 0", "กลุ่ม 1"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"width": 0.42}
        )
        ax.set_title("HeartDisease")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("จำนวนข้อมูลแต่ละกลุ่ม")
        chart_data = pd.DataFrame({
            "จำนวนข้อมูล": [count_0, count_1]
        }, index=["HeartDisease = 0", "HeartDisease = 1"])
        st.bar_chart(chart_data)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("ตัวอย่างข้อมูลจาก Heart3.csv")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    zero_chol = int((data["Cholesterol"] == 0).sum())
    zero_bp = int((data["RestingBP"] == 0).sum())
    if zero_chol > 0 or zero_bp > 0:
        st.warning(
            f"ข้อสังเกต: Cholesterol = 0 จำนวน {zero_chol} รายการ และ "
            f"RestingBP = 0 จำนวน {zero_bp} รายการ ระบบคงค่าตามไฟล์ต้นฉบับ "
            "ควรตรวจสอบความหมายของค่าศูนย์ก่อนนำไปใช้งานจริง"
        )

# ---------------- เมนูทำนายข้อมูลใหม่ ----------------
elif page == "ทำนายข้อมูลใหม่":
    st.markdown("""
    <div class="hero">
      <h1>🩺 ทำนายข้อมูลผู้ป่วยรายใหม่</h1>
      <p>กรอกข้อมูลทั้ง 11 ตัวแปร แล้วให้โมเดล KNN ช่วยจำแนกกลุ่มข้อมูล</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "ตัวแปรหมวดหมู่จะแสดงเป็นรหัสตัวเลขตาม Heart3.csv "
        "เนื่องจากไฟล์ที่ได้รับไม่มีตารางอธิบายความหมายของรหัสแต่ละค่า"
    )

    with st.form("heart_prediction"):
        st.markdown("#### 1. ข้อมูลทั่วไป")
        a, b, c = st.columns(3)
        with a:
            age = st.slider("Age (อายุ)", 28, 77, 50)
        with b:
            sex = st.selectbox("Sex (รหัส)", [0, 1])
        with c:
            chest = st.selectbox("ChestPainType (รหัส)", [1, 2, 3, 4])

        st.divider()
        st.markdown("#### 2. ข้อมูลสุขภาพ")
        a, b, c = st.columns(3)
        with a:
            resting_bp = st.number_input("RestingBP", 0, 200, 120)
        with b:
            cholesterol = st.number_input("Cholesterol", 0, 603, 200)
        with c:
            fasting_bs = st.selectbox("FastingBS (รหัส)", [0, 1])

        a, b, c = st.columns(3)
        with a:
            resting_ecg = st.selectbox("RestingECG (รหัส)", [1, 2, 3])
        with b:
            max_hr = st.number_input("MaxHR", 60, 202, 140)
        with c:
            exercise_angina = st.selectbox("ExerciseAngina (รหัส)", [0, 1])

        st.divider()
        st.markdown("#### 3. ผลตรวจเพิ่มเติม")
        a, b = st.columns(2)
        with a:
            oldpeak = st.number_input(
                "Oldpeak", min_value=-2.6, max_value=6.2,
                value=0.0, step=0.1, format="%.1f"
            )
        with b:
            st_slope = st.selectbox("ST_Slope (รหัส)", [1, 2, 3])

        submitted = st.form_submit_button(
            "🔍 ทำนายผลด้วย KNN", type="primary", use_container_width=True
        )

    if submitted:
        new_record = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }], columns=FEATURES)

        prediction = int(model.predict(new_record)[0])
        probabilities = model.predict_proba(new_record)[0]
        classes = list(model.named_steps["knn"].classes_)
        probability_by_class = dict(zip(classes, probabilities))
        predicted_score = float(probability_by_class[prediction] * 100)

        st.markdown("### ผลการจำแนกข้อมูล")
        if prediction == 0:
            st.markdown(
                f'<div class="result-0"><h2>ผลการทำนาย: กลุ่ม 0</h2>'
                f'<h3>โมเดลจัดข้อมูลนี้อยู่ในกลุ่ม HeartDisease = 0</h3>'
                f'<p>สัดส่วนเพื่อนบ้านที่ลงคะแนนให้กลุ่มนี้: {predicted_score:.1f}%</p></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-1"><h2>ผลการทำนาย: กลุ่ม 1</h2>'
                f'<h3>โมเดลจัดข้อมูลนี้อยู่ในกลุ่ม HeartDisease = 1</h3>'
                f'<p>สัดส่วนเพื่อนบ้านที่ลงคะแนนให้กลุ่มนี้: {predicted_score:.1f}%</p></div>',
                unsafe_allow_html=True
            )

        p1, p2 = st.columns(2)
        with p1:
            st.metric("คะแนนกลุ่ม 0", f"{probability_by_class.get(0, 0)*100:.1f}%")
        with p2:
            st.metric("คะแนนกลุ่ม 1", f"{probability_by_class.get(1, 0)*100:.1f}%")

        st.caption(
            "คะแนนนี้คือสัดส่วนการโหวตจากเพื่อนบ้าน K รายการ ไม่ใช่ความน่าจะเป็นทางการแพทย์"
        )
        with st.expander("ดูข้อมูลที่ป้อน"):
            st.dataframe(new_record, use_container_width=True, hide_index=True)

        st.warning(
            "ระบบนี้จัดทำเพื่อการศึกษาเท่านั้น ผลการจำแนกไม่ใช่การวินิจฉัยโรค "
            "และไม่ควรใช้แทนคำแนะนำจากแพทย์"
        )

# ---------------- เมนูประเมินโมเดล ----------------
elif page == "ประเมินโมเดล":
    st.markdown("""
    <div class="hero">
      <h1>📊 ประเมินประสิทธิภาพโมเดล</h1>
      <p>ผลการทดสอบโมเดล KNN จากข้อมูลที่แบ่งเป็นชุดฝึกและชุดทดสอบ</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        f"K = {k_value} | Train = {100-test_percent}% | Test = {test_percent}% | "
        "Random state = 42"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    with c2: metric_card("Precision", f"{metrics['precision']*100:.2f}%")
    with c3: metric_card("Recall", f"{metrics['recall']*100:.2f}%")
    with c4: metric_card("F1-score", f"{metrics['f1']*100:.2f}%")

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        cm = metrics["matrix"]
        cm_table = pd.DataFrame(
            cm, index=["ค่าจริง 0", "ค่าจริง 1"],
            columns=["ทำนาย 0", "ทำนาย 1"]
        )
        st.dataframe(cm_table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["ทำนาย 0", "ทำนาย 1"])
        ax.set_yticks([0, 1], ["ค่าจริง 0", "ค่าจริง 1"])
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=14)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("รายละเอียดการฝึกโมเดล")
        st.write(f"ข้อมูลฝึก (Train): **{metrics['train_count']} รายการ**")
        st.write(f"ข้อมูลทดสอบ (Test): **{metrics['test_count']} รายการ**")
        st.write(f"จำนวนเพื่อนบ้าน (K): **{k_value}**")
        st.write("การปรับสเกล: **StandardScaler**")
        st.write("อัลกอริทึม: **K-Nearest Neighbors**")
        st.write("การแบ่งข้อมูล: **Stratified Train/Test Split**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Classification Report")
    report = classification_report(
        metrics["y_test"], metrics["predicted"],
        labels=[0, 1], target_names=["กลุ่ม 0", "กลุ่ม 1"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- เมนูเกี่ยวกับระบบ ----------------
else:
    st.markdown("""
    <div class="hero">
      <h1>ℹ️ เกี่ยวกับระบบ</h1>
      <p>ระบบจำแนกข้อมูลโรคหัวใจด้วยเทคนิค K-Nearest Neighbors</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 2])
    with left:
        if os.path.exists("img/ll.png"):
            st.image("img/ll.png", caption="ผู้จัดทำ", use_container_width=True)
        else:
            st.info("สามารถเพิ่มรูปผู้จัดทำไว้ที่ img/ll.png")
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("หลักการทำงานของ KNN")
        st.write(
            "K-Nearest Neighbors (KNN) เป็นเทคนิคการเรียนรู้ของเครื่อง "
            "ที่จำแนกข้อมูลใหม่โดยคำนวณระยะห่างจากข้อมูลฝึก "
            "เลือกเพื่อนบ้านที่ใกล้ที่สุดจำนวน K รายการ แล้วใช้เสียงข้างมากในการจำแนก"
        )
        st.subheader("ขั้นตอนการทำงาน")
        st.write(
            "1. อ่านข้อมูลจาก Heart3.csv\n"
            "2. กำหนดตัวแปรอิสระ 11 ตัว และตัวแปรเป้าหมาย HeartDisease\n"
            "3. แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ\n"
            "4. ปรับมาตรฐานข้อมูลด้วย StandardScaler\n"
            "5. ฝึกโมเดล KNN และประเมินผล\n"
            "6. รับข้อมูลใหม่จากผู้ใช้และทำนายกลุ่ม"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("ตัวแปรเป้าหมาย")
    st.write("- HeartDisease = 0: กลุ่มรหัส 0 ตามป้ายกำกับในชุดข้อมูล")
    st.write("- HeartDisease = 1: กลุ่มรหัส 1 ตามป้ายกำกับในชุดข้อมูล")
    st.caption(
        "ไฟล์ CSV ไม่มีคำอธิบายรหัสของตัวแปรหมวดหมู่ จึงแสดงรหัสเดิมโดยไม่ตีความความหมายเพิ่มเติม"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.warning("โปรแกรมนี้มีวัตถุประสงค์เพื่อการศึกษา ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์")