import streamlit as st
from PIL import Image
import pytesseract
import numpy as np


# ============================================
# STREAMLIT CONFIG
# ============================================
st.set_page_config(page_title="OCR + Chatbot (Siêu nhẹ)", layout="wide")
st.title("📄 OCR + 🤖 Chatbot Tiếng Việt (Bản siêu nhẹ – không dùng AI nặng)")


# ============================================
# SIMPLE RULE-BASED CHATBOT
# ============================================
def chatbot_answer(ocr_text, question):

    question = question.lower()

    # 1. Nếu người dùng hỏi tóm tắt
    if "tóm tắt" in question or "tom tat" in question or "nội dung" in question:
        return f"Tóm tắt nội dung OCR:\n{ocr_text[:300]}..."

    # 2. Hỏi về giá tiền
    if "tiền" in question or "giá" in question or "total" in question:
        import re
        prices = re.findall(r"\d[\d,.]*", ocr_text)
        if prices:
            return f"Mình tìm thấy các con số liên quan đến tiền: {', '.join(prices)}"
        else:
            return "Không tìm thấy số tiền nào trong văn bản."

    # 3. Hỏi về ngày tháng
    if "ngày" in question or "date" in question:
        import re
        dates = re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}", ocr_text)
        if dates:
            return f"Ngày tháng có thể là: {', '.join(dates)}"
        else:
            return "Không tìm thấy ngày tháng trong văn bản."

    # 4. Hỏi chung chung → trả lời dựa trên từ khóa có trong OCR
    keywords = [w for w in question.split() if w in ocr_text.lower()]
    if keywords:
        return f"Mình tìm thấy các từ khóa {keywords} trong OCR. Dưới đây là nội dung:\n\n{ocr_text}"

    # 5. Default fallback
    return "Mình đã đọc nội dung OCR nhưng không hiểu câu hỏi. Bạn thử diễn đạt lại nhé!"


# ============================================
# UI
# ============================================
uploaded = st.file_uploader("📤 Tải ảnh (jpg/png)…", type=["jpg", "jpeg", "png"])

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""


# ============================================
# OCR PROCESS
# ============================================
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    if st.button("🔍 Chạy OCR"):
        with st.spinner("Đang chạy OCR..."):
            text = pytesseract.image_to_string(img, lang="vie")
            st.session_state.ocr_text = text

        st.success("OCR hoàn tất!")
        st.text_area("📌 Văn bản OCR:", text, height=200)


# ============================================
# CHATBOT PHẦN HỎI ĐÁP
# ============================================
st.subheader("💬 Hỏi chatbot dựa trên nội dung OCR")

if not st.session_state.ocr_text:
    st.info("Hãy upload ảnh và chạy OCR trước.")
else:
    q = st.text_input("Nhập câu hỏi:")

    if st.button("🤖 Trả lời"):
        answer = chatbot_answer(st.session_state.ocr_text, q)
        st.write("### 🧠 Trả lời:")
        st.write(answer)
