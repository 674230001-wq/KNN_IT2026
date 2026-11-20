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

st.set_page_config(
    page_title="Heart AI | ระบบจำแนกโรคหัวใจ",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
.stApp { background: #f4f7fb; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
.hero {
    background: linear-gradient(120deg,#102b46 0%,#174c72 60%,#168aad 100%);
    padding: 30px 34px; border-radius: 22px; color: white; margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(16,43,70,.15);
}
.hero h1 { color: white; font-size: 2rem; margin: 0 0 7px 0; }
.hero p { color: #d8eaf5; margin: 0; }
.metric-card {
    background: white; border: 1px solid #e6edf4; border-radius: 17px;
    padding: 18px 20px; box-shadow: 0 4px 15px rgba(20,55,90,.04);
}
.metric-label { color: #64748b; font-size: .9rem; }
.metric-value { color: #123653; font-size: 1.8rem; font-weight: 700; margin-top: 3px; }
.section-card {
    background: white; border: 1px solid #e6edf4; border-radius: 18px;
    padding: 22px; margin-bottom: 16px;
}
.result-positive {
    background: #fff1f1; border: 1px solid #fecaca; border-radius: 18px;
    padding: 22px; color: #991b1b;
}
.result-negative {
    background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 18px;
    padding: 22px; color: #065f46;
}
.small-note { color: #64748b; font-size: .85rem; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#1479a8,#0e9ab7);
    border: 0; border-radius: 12px; font-weight: 600; min-height: 3rem;
}
[data-testid="stSidebar"] { background: #102b46; }
[data-testid="stSidebar"] * { color: #f2f7fb; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]
TARGET = "HeartDisease"
EXPECTED_COLUMNS = FEATURES + [TARGET]

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("Heart3.csv")
    missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("ไม่พบคอลัมน์ที่จำเป็น: " + ", ".join(missing))
    data = data[EXPECTED_COLUMNS].copy()
    for c in EXPECTED_COLUMNS:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=EXPECTED_COLUMNS)
    data[TARGET] = data[TARGET].astype(int)
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("HeartDisease ต้องมีค่าเป็น 0 หรือ 1 เท่านั้น")
    return data

def make_model(k):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="minkowski", p=2))
    ])

def train_model(data, k, test_size):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = make_model(k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "matrix": confusion_matrix(y_test, pred, labels=[0, 1]),
        "y_test": y_test,
        "pred": pred,
        "train_count": len(X_train),
        "test_count": len(X_test)
    }
    # Refit using all available rows for the new-record prediction page.
    final_model = make_model(k)
    final_model.fit(X, y)
    return final_model, metrics

def metric_card(label, value, suffix=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}{suffix}</div></div>',
        unsafe_allow_html=True
    )

# -------------------- SIDEBAR / DATA --------------------
st.sidebar.markdown("## ❤️ HEART AI")
st.sidebar.caption("ระบบจำแนกโรคหัวใจด้วย KNN")
# รูปผู้จัดทำเว็บไซต์ (ไม่ใช่ข้อมูลที่ใช้ฝึกโมเดล)
profile_image = Path(__file__).with_name("ll.png")
if profile_image.exists():
    st.sidebar.image(str(profile_image), caption="ผู้จัดทำโครงงาน", use_container_width=True)
st.sidebar.markdown("**Heart Disease Classification**")
st.sidebar.caption("พัฒนาโดยใช้ Python + Streamlit + KNN")
st.sidebar.divider()
page = st.sidebar.radio(
    "เมนูหลัก",
    ["ภาพรวมข้อมูล", "ทำนายข้อมูลใหม่", "ประเมินโมเดล", "เกี่ยวกับระบบ"]
)
st.sidebar.divider()
uploaded = st.sidebar.file_uploader("เปลี่ยนชุดข้อมูล CSV (ถ้าต้องการ)", type=["csv"])

try:
    data = load_data(uploaded)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ Heart3.csv กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกับ app.py หรืออัปโหลด CSV ที่แถบด้านซ้าย")
    st.stop()
except Exception as e:
    st.error(f"อ่านข้อมูลไม่สำเร็จ: {e}")
    st.stop()

st.sidebar.markdown("### ตั้งค่า KNN")
k_value = st.sidebar.slider("จำนวนเพื่อนบ้าน (K)", min_value=1, max_value=25, value=5, step=2)
test_percent = st.sidebar.slider("สัดส่วนข้อมูลทดสอบ (%)", 10, 40, 20, 5)
test_size = test_percent / 100

if len(data) <= k_value:
    st.error("จำนวนข้อมูลต้องมากกว่าค่า K")
    st.stop()

model, scores = train_model(data, k_value, test_size)

# -------------------- PAGE: OVERVIEW --------------------
if page == "ภาพรวมข้อมูล":
    st.markdown("""
    <div class="hero">
      <h1>❤️ ระบบจำแนกโรคหัวใจด้วย KNN</h1>
      <p>Heart Disease Classification • Machine Learning Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    total = len(data)
    positive = int((data[TARGET] == 1).sum())
    negative = int((data[TARGET] == 0).sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("จำนวนข้อมูลทั้งหมด", f"{total:,}")
    with c2: metric_card("กลุ่มรหัส 0", f"{negative:,}")
    with c3: metric_card("กลุ่มรหัส 1", f"{positive:,}")
    with c4: metric_card("Accuracy (ชุดทดสอบ)", f"{scores['accuracy']*100:.1f}", "%")

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("สัดส่วนผลลัพธ์ในชุดข้อมูล")
        counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.pie(counts.values, labels=["รหัส 0", "รหัส 1"], autopct="%1.1f%%",
               startangle=90, wedgeprops={"width": 0.42})
        ax.set_title("HeartDisease")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("จำนวนข้อมูลแต่ละกลุ่ม")
        st.bar_chart(counts.rename(index={0: "รหัส 0", 1: "รหัส 1"}))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("ตัวอย่างข้อมูล")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    zero_chol = int((data["Cholesterol"] == 0).sum())
    zero_bp = int((data["RestingBP"] == 0).sum())
    if zero_chol or zero_bp:
        st.warning(
            f"ข้อสังเกตจากข้อมูล: Cholesterol = 0 จำนวน {zero_chol} แถว และ "
            f"RestingBP = 0 จำนวน {zero_bp} แถว ระบบยังคงใช้ค่าตามไฟล์ต้นฉบับ "
            "ไม่ได้แทนค่าอัตโนมัติ ควรตรวจสอบความหมายก่อนนำไปใช้จริง"
        )

# -------------------- PAGE: PREDICTION --------------------
elif page == "ทำนายข้อมูลใหม่":
    st.markdown("""
    <div class="hero">
      <h1>🩺 ทำนายข้อมูลผู้ป่วยรายใหม่</h1>
      <p>กรอกข้อมูลให้ครบทั้ง 11 ตัวแปร แล้วกดปุ่มเพื่อให้โมเดล KNN จำแนกผล</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("กรุณากรอกตัวแปรตามรหัสที่ใช้ใน Heart3.csv โดยเฉพาะตัวแปรประเภทหมวดหมู่ ซึ่งหน้าเว็บจะแสดงเป็นรหัสตัวเลขตามต้นฉบับ")

    with st.form("prediction_form"):
        st.subheader("1. ข้อมูลพื้นฐาน")
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("อายุ (Age)", min_value=1, max_value=120, value=50)
        with b:
            sex = st.selectbox("เพศ (Sex: รหัส)", [0, 1], index=1,
                               format_func=lambda x: f"รหัส {x}")
        with c:
            chest = st.selectbox("ประเภทอาการเจ็บหน้าอก (ChestPainType: รหัส)", [1, 2, 3, 4])

        st.divider()
        st.subheader("2. ผลตรวจสุขภาพ")
        a, b, c = st.columns(3)
        with a:
            bp = st.number_input("ความดันขณะพัก (RestingBP)", min_value=0, max_value=300, value=130)
        with b:
            chol = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=0, max_value=1000, value=200)
        with c:
            fasting = st.selectbox("FastingBS (รหัส)", [0, 1], format_func=lambda x: f"รหัส {x}")

        a, b, c = st.columns(3)
        with a:
            ecg = st.selectbox("RestingECG (รหัส)", [1, 2, 3])
        with b:
            maxhr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=1, max_value=300, value=150)
        with c:
            angina = st.selectbox("ExerciseAngina (รหัส)", [0, 1], format_func=lambda x: f"รหัส {x}")

        st.divider()
        st.subheader("3. ค่าการตรวจเพิ่มเติม")
        a, b = st.columns(2)
        with a:
            oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=20.0, value=0.0, step=0.1)
        with b:
            slope = st.selectbox("ST_Slope (รหัส)", [1, 2, 3])

        submitted = st.form_submit_button("🔍 ทำนายผลด้วย KNN", type="primary", use_container_width=True)

    if submitted:
        new_data = pd.DataFrame([{
            "Age": age, "Sex": sex, "ChestPainType": chest, "RestingBP": bp,
            "Cholesterol": chol, "FastingBS": fasting, "RestingECG": ecg,
            "MaxHR": maxhr, "ExerciseAngina": angina, "Oldpeak": oldpeak,
            "ST_Slope": slope
        }], columns=FEATURES)
        prediction = int(model.predict(new_data)[0])
        probabilities = model.predict_proba(new_data)[0]
        class_index = list(model.named_steps["knn"].classes_).index(prediction)
        confidence = float(probabilities[class_index] * 100)

        st.markdown("### ผลการจำแนก")
        if prediction == 1:
            st.markdown(
                f'<div class="result-positive"><h2>⚠️ ผลจำแนก: รหัส 1</h2>'
                f'<h3>โมเดลจัดข้อมูลนี้อยู่ในกลุ่มที่พบโรคหัวใจ</h3>'
                f'<p>สัดส่วนคะแนนของกลุ่มที่โมเดลทำนาย: {confidence:.1f}%</p></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-negative"><h2>✅ ผลจำแนก: รหัส 0</h2>'
                f'<h3>โมเดลจัดข้อมูลนี้อยู่ในกลุ่มที่ไม่พบโรคหัวใจ</h3>'
                f'<p>สัดส่วนคะแนนของกลุ่มที่โมเดลทำนาย: {confidence:.1f}%</p></div>',
                unsafe_allow_html=True
            )
        st.write("")
        p1, p2 = st.columns(2)
        with p1:
            st.metric("คะแนนกลุ่มรหัส 0", f"{probabilities[0]*100:.1f}%")
        with p2:
            st.metric("คะแนนกลุ่มรหัส 1", f"{probabilities[1]*100:.1f}%")
        st.caption("คะแนนดังกล่าวเป็นสัดส่วนของเพื่อนบ้านในโมเดล KNN ไม่ใช่ความน่าจะเป็นทางการแพทย์")
        with st.expander("ตรวจสอบข้อมูลที่ป้อน"):
            st.dataframe(new_data, use_container_width=True, hide_index=True)

        st.warning("ผลลัพธ์นี้เป็นการสาธิตการจำแนกด้วยชุดข้อมูลเพื่อการศึกษา ไม่ใช่การวินิจฉัยโรคหรือคำแนะนำทางการแพทย์ หากมีอาการหรือข้อกังวลควรปรึกษาแพทย์")

# -------------------- PAGE: EVALUATION --------------------
elif page == "ประเมินโมเดล":
    st.markdown("""
    <div class="hero">
      <h1>📊 ประเมินประสิทธิภาพโมเดล</h1>
      <p>ทดสอบโมเดล KNN ด้วยข้อมูลที่แบ่งแยกออกจากชุดฝึก</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"ตั้งค่า K = {k_value} • Train = {100-test_percent}% • Test = {test_percent}% • random_state = 42")
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Accuracy", f"{scores['accuracy']*100:.2f}", "%")
    with m2: metric_card("Precision", f"{scores['precision']*100:.2f}", "%")
    with m3: metric_card("Recall", f"{scores['recall']*100:.2f}", "%")
    with m4: metric_card("F1-score", f"{scores['f1']*100:.2f}", "%")

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        cm = scores["matrix"]
        cm_df = pd.DataFrame(cm, index=["จริง: 0", "จริง: 1"], columns=["ทำนาย: 0", "ทำนาย: 1"])
        st.dataframe(cm_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["ทำนาย 0", "ทำนาย 1"])
        ax.set_yticks([0, 1], ["จริง 0", "จริง 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("จำนวนข้อมูลที่ใช้")
        st.write(f"- ข้อมูลฝึกสอน (Train): **{scores['train_count']}** แถว")
        st.write(f"- ข้อมูลทดสอบ (Test): **{scores['test_count']}** แถว")
        st.write(f"- จำนวนเพื่อนบ้าน K: **{k_value}**")
        st.write("- การปรับมาตรฐาน: **StandardScaler**")
        st.write("- ระยะทาง: **Euclidean Distance**")
        st.write("- การแบ่งข้อมูล: **Stratified Train/Test Split**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("รายงานผลการจำแนก")
    report = classification_report(
        scores["y_test"], scores["pred"], labels=[0, 1],
        target_names=["รหัส 0", "รหัส 1"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PAGE: ABOUT --------------------
else:
    st.markdown("""
    <div class="hero">
      <h1>ℹ️ เกี่ยวกับระบบ</h1>
      <p>รายละเอียดชุดข้อมูลและหลักการทำงานของระบบจำแนกโรคหัวใจ</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("หลักการ K-Nearest Neighbors (KNN)")
    st.write("""
    KNN เป็นเทคนิค Machine Learning สำหรับจำแนกข้อมูล โดยคำนวณระยะห่างระหว่างข้อมูลใหม่กับข้อมูลในชุดฝึก
    จากนั้นเลือกข้อมูลเพื่อนบ้านที่ใกล้ที่สุดจำนวน K รายการ และใช้เสียงข้างมากในการตัดสินกลุ่ม
    """)
    st.subheader("ขั้นตอนการทำงาน")
    st.write("""
    1. อ่านข้อมูลจากไฟล์ Heart3.csv
    2. แยกตัวแปรอิสระ 11 ตัวแปร และตัวแปรเป้าหมาย HeartDisease
    3. แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    4. ปรับมาตรฐานข้อมูลด้วย StandardScaler
    5. ฝึกโมเดล KNN ด้วยค่า K ที่ผู้ใช้เลือก
    6. ประเมินผลด้วย Accuracy, Precision, Recall และ F1-score
    7. ใช้โมเดลที่ฝึกจากข้อมูลทั้งหมดเพื่อทำนายข้อมูลใหม่
    """)
    st.subheader("ตัวแปรเป้าหมาย")
    st.write("- รหัส 0: กลุ่มไม่พบโรคหัวใจตามป้ายกำกับในชุดข้อมูล")
    st.write("- รหัส 1: กลุ่มพบโรคหัวใจตามป้ายกำกับในชุดข้อมูล")
    st.markdown('</div>', unsafe_allow_html=True)
    st.warning("ระบบนี้จัดทำเพื่อการศึกษาเท่านั้น ไม่ใช่อุปกรณ์ทางการแพทย์ และไม่ควรใช้แทนการตรวจวินิจฉัยโดยบุคลากรทางการแพทย์")
