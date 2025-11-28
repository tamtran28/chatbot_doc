import streamlit as st
from PIL import Image
import pytesseract

st.set_page_config(page_title="OCR + Chatbot", layout="wide")
st.title("📄 OCR + Chatbot Tiếng Việt (Siêu nhẹ – Streamlit Cloud)")

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

uploaded = st.file_uploader("Tải ảnh (jpg/png)…", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)

    if st.button("🔍 Chạy OCR"):
        text = pytesseract.image_to_string(img, lang="vie")
        st.session_state.ocr_text = text
        st.success("OCR hoàn tất!")
        st.text_area("📌 Văn bản OCR:", text, height=200)

st.subheader("💬 Hỏi đáp dựa theo OCR")

def reply(ocr, q):
    if "tiền" in q or "tien" in q:
        return "Dữ liệu có vẻ liên quan số tiền. Đây là nội dung OCR:\n" + ocr
    if "ngày" in q or "date" in q:
        return "Có thể bạn đang hỏi về ngày tháng. Đây là OCR:\n" + ocr
    return "Dựa trên OCR, mình trả lời thế này:\n" + ocr

question = st.text_input("Nhập câu hỏi:")
if st.button("🤖 Trả lời"):
    st.write(reply(st.session_state.ocr_text, question))
