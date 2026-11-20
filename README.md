# Heart AI - KNN Streamlit

## ไฟล์ในโปรเจกต์
- `app.py` : โปรแกรม Streamlit
- `Heart3.csv` : ชุดข้อมูลสำหรับฝึกโมเดล
- `requirements.txt` : ไลบรารีที่ต้องติดตั้ง

## วิธีรันบนเครื่อง
1. ติดตั้ง Python
2. เปิด Terminal ในโฟลเดอร์โปรเจกต์
3. ติดตั้งไลบรารี

```bash
pip install -r requirements.txt
```

4. รันเว็บไซต์

```bash
streamlit run app.py
```

## วิธี Deploy บน Streamlit Community Cloud
1. สร้าง GitHub repository
2. อัปโหลด `app.py`, `requirements.txt` และ `Heart3.csv`
3. เข้า https://share.streamlit.io/
4. เลือก Deploy an app
5. เลือก repository, branch และไฟล์หลัก `app.py`
6. กด Deploy

## หมายเหตุ
ระบบใช้ StandardScaler ก่อน KNN เนื่องจากตัวแปรมีหน่วยและช่วงค่าต่างกัน
รหัสหมวดหมู่ใช้ตามค่าตัวเลขใน CSV เนื่องจากไฟล์ไม่มี codebook อธิบายรหัส
ผลลัพธ์ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์
