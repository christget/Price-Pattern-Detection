import streamlit as st
import io
import sys

from main import main_function

# initialize page config
st.set_page_config(page_title="Price Pattern Detection", page_icon=":chart_with_upwards_trend:", layout="centered")
st.title(" :chart_with_downwards_trend: Price Pattern Detection")
st.markdown("<style>div.block-container{padding-top:1rem;}<style>", unsafe_allow_html=True)

# parameter preparation
model = "model/best.pt"

st.subheader("About the Model")
about = '''
โมเดลนี้มีวัตถุประสงค์เพื่อตรวจจับหา Price Patterns รูปแบบต่างๆ ที่เกิดขึ้นในตลาดซื้อขายสินทรัพย์ 
โดยจะค่อยๆ แสกนหาที่ละช่วงเวลา จากปัจุบันย้อนหลังไปยังช่วงเวลาที่กำหนด
'''
st.markdown(about)

how_ex = st.expander("How This Model Work 🧐")
with how_ex:
    st.subheader("How This Model Works!")
    how = '''
    1. ดึงข้อมูลราคาของ Asset ย้อนหลัง ตามวันที่กำหนด `Period` ที่กำหนด
    2. สร้างรูปภาพจากช่วงเวลาที่กำหนด โดยจำนวนรูปภาพที่ได้จะขึ้นอยู่กับ `Period` และ `Timeframe`
        - หากเลือก `Timeframe day (1d)` รูปภาพจะมีกราฟแท่งเทียนทั้งหมด 15 แท่ง
        - ในขณะ `Timeframe 1 hour (1h)` จะมีแท่งเทียน 72 แท่งต่อ 1 รูปภาพ
    3. โมเดลจะเริ่มแสกนทีละรูปภาพจากภาพช่วงเวลาล่าสุดไปภาพสุดท้าย 
        - สามารถตั้งค่าให้โมเดลแสกนแค่รูปภาพช่วงเวลาล่าสุดได้โดยกำหนด `Latest Periods` :blue[True]
    4. Price Patterns ที่ปัจจุบันโมเดลสามารถตรวจจับได้ ประกอบด้วย 
        - Double top
        - Double Bottom
        - Head and Shoulder
        - Inverse Head and Shoulder
    5. เมื่อเจอ Pattern โมเดลจะหยุดทำงาน ณ รูปภาพที่เจอ และให้ผลลัพธ์ประกอบด้วย
        - Patterns ทั้งหมดที่เจอในรูปภาพนั้น
        - ค่า Confidece ของ Patterns ที่เจอ
        - ช่วงเวลาที่เจอ และห่างจากช่วงเวลาปัจจุบันเท่าไหร่'''
    st.markdown(how)

st.subheader("Inputs")
inputs = '''- `symbol` : สัญลักษณ์ของสินทรัพย์ (crypto ต้องมี -USD ลงท้าย) 
- `period` : ช่วงเวลาที่ต้องการดึงย้อนหลัง (หน่วยเป็น วัน)
- `timefrmae` : เวลาต่อแท่งเทียน
- `confidence` : ค่าความมั่นใจที่จะตรวจเจอ pattern
- `latest Period` : True จะ detect เฉพาะภาพล่าสุด False จะลูปจนกว่าจะ detect เจอ'''
st.markdown(inputs)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    symbol = st.text_input("Symbol", value="btc-usd")
    symbol = symbol.upper()
with col2:
    #period = int(st.text_input("Period (days)", value=30))
    period = st.slider("Period", min_value=30, max_value=300, value=30, step=30)
with col3:
    tf = st.selectbox("Timeframe", options= ["1h", "1d"]) 
with col4:
    confidence = st.slider("Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
with col5:
    last_day = st.selectbox("Latest Period", options= [False, True])
with col6:
    st.markdown(
    """
    <style>
        div.stButton > button {
            margin-top: 12px; 
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    trigger = st.button("Let's Try!", type="primary",)
    

st.subheader("Output")
if trigger:
    with st.spinner("🫠 Hold on a moment!!"):
        try:
            result = main_function(symbol=symbol,
                                    period=period,
                                    timeframe=tf,
                                    model=model,
                                    conf=confidence,
                                    last_day=last_day)
        except Exception as e:
            st.error("Please check your symbol agian (if crypto need '-usd' e.g. btc-usd)", icon="😵‍💫")
            st.markdown("More detail about symbols please go checking at [Yahoo Finance](https://finance.yahoo.com/lookup/)")
            sys.exit()

        image = result["image"]
        image_byte = io.BytesIO()
        image.save(image_byte, format="PNG")
        if result["class"][0] == "No Detection":
            st.warning("No Pattern Found", icon="🥲")
            st.subheader("Image Output")
            st.image(image_byte, caption="Image Result", use_column_width=True)
        elif len(result["class"]) > 1:
            # classes_str = ", ".join(result["class"][:-1]) + f" and {result['class'][-1]}"
            # conf_strings = list(map(str, result["conf"]))
            # conf = ", ".join(conf_strings[:-1]) + f" and {conf_strings[-1]}"
            text_result = f"Found {result['class']} with {result['conf']} confidence\nabout {result['when']['days']} days, {result['when']['hours']} hours ago"
            st.success(f"{len(result['class'])} Patterns Found", icon="😎")
            st.text(text_result)
            st.subheader("Image Output")
            st.image(image_byte, caption="Image Result", use_column_width=True)
        else:
            text_result = f"Found '{result['class'][0]}' with {result['conf'][0]:.2f} confidence\nabout {result['when']['days']} days and {result['when']['hours']} hours ago"
            st.success(f"{len(result['class'])} Pattern Found", icon="😎")
            st.text(text_result)
            st.subheader("Image Output")
            st.image(image_byte, caption="Data from Yahoo Finance", use_column_width=True)