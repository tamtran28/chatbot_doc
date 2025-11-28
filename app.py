import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR
from llama_cpp import Llama
import numpy as np

st.set_page_config(page_title="OCR + LLM GPU", layout="wide")
st.title("📄 OCR + 🤖 Chatbot (GPU T4)")

# -----------------------------
# CACHE LOAD MODELS (GPU)
# -----------------------------
@st.cache_resource
def load_llm():
    return Llama(
        model_path="models/Phi-3-mini-4k-instruct.Q4_K_M.gguf",
        n_gpu_layers=60,
        n_ctx=2048,
        verbose=False
    )

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang="vi")  # GPU tự động bật trên HF

llm = load_llm()
ocr_model = load_ocr()

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

# -----------------------------
# OCR
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh tải lên", use_column_width=True)

    st.write("🔍 Đang chạy OCR...")
    result = ocr_model.ocr(np.array(img), cls=True)

    text = "\n".join([line[1][0] for line in result[0]])
    st.session_state.ocr_text = text

    st.subheader("📌 Text OCR:")
    st.write(text)
    st.divider()

# -----------------------------
# CHATBOT
# -----------------------------
if st.session_state.ocr_text:
    query = st.text_input("Nhập câu hỏi:")

    if query:
        prompt = f"""
Dựa trên văn bản OCR sau:

{text}

Trả lời câu hỏi: {query}

Trả lời:
"""

        output = llm(prompt, max_tokens=200)
        answer = output["choices"][0]["text"]

        st.write("### 🤖 Trả lời:")
        st.write(answer)
else:
    st.info("Hãy upload ảnh để OCR.")
