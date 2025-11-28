import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
from llama_cpp import Llama
import os

# ------------------------------
# 1. Load LLM Offline
# ------------------------------
LLM_PATH = "models/Phi-3-mini-4k-instruct.Q4_K_M.gguf"

llm = Llama(
    model_path=LLM_PATH,
    n_threads=6,        # chỉnh theo CPU của bạn
    n_ctx=2048,
    verbose=False
)

# ------------------------------
# 2. OCR Model
# ------------------------------
ocr_model = PaddleOCR(lang='vi', use_angle_cls=True)

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.set_page_config(page_title="OCR + Offline LLM", layout="wide")
st.title("📄 OCR + 🤖 Chatbot chạy Offline Hoàn Toàn")

uploaded_file = st.file_uploader("Tải ảnh hóa đơn / giấy tờ", type=["jpg", "png", "jpeg"])

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

# ------------------------------
# 4. OCR PROCESS
# ------------------------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh tải lên", use_column_width=True)

    st.write("### 🔍 Đang chạy OCR...")
    result = ocr_model.ocr(img, cls=True)

    extracted_text = "\n".join([line[1][0] for line in result[0]])
    st.session_state.ocr_text = extracted_text

    st.subheader("📌 Văn bản OCR:")
    st.write(extracted_text)

    st.divider()

# ------------------------------
# 5. OFFLINE CHATBOT
# ------------------------------
if st.session_state.ocr_text:
    st.subheader("💬 Chatbot hỏi đáp chạy offline")

    user_message = st.text_input("Nhập câu hỏi:")

    if user_message:
        prompt = f"""
Bạn là trợ lý AI. Dựa trên văn bản OCR bên dưới, hãy trả lời câu hỏi.

### Văn bản OCR:
{st.session_state.ocr_text}

### Câu hỏi:
{user_message}

### Trả lời:
"""

        output = llm(
            prompt,
            temperature=0.1,
            max_tokens=256,
            stop=["###"]
        )

        answer = output["choices"][0]["text"]

        st.write("### 🤖 Trả lời:")
        st.write(answer)

else:
    st.info("Hãy upload ảnh trước để chạy OCR.")
