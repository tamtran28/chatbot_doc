import streamlit as st
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from llama_cpp import Llama

# ===========================
# STREAMLIT PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="OCR + LLM Chatbot (HF Spaces)",
    layout="wide"
)

st.title("📄 OCR + 🤖 Chatbot (LLM Offline – HuggingFace Spaces)")

# ===========================
# LOAD MODELS WITH CACHE
# ===========================
@st.cache_resource
def load_ocr_model():
    return PaddleOCR(use_angle_cls=True, lang="vi")

@st.cache_resource
def load_llm_model():
    return Llama(
        model_path="models/Phi-3-mini-4k-instruct.Q4_K_M.gguf",  
        n_ctx=2048,
        n_threads=4,   # HF Spaces CPU typically = 2–4 threads
        verbose=False
    )

ocr = load_ocr_model()
llm = load_llm_model()


# ===========================
# FRONTEND – UPLOAD IMAGE
# ===========================
uploaded_file = st.file_uploader(
    "📤 Tải ảnh hóa đơn / giấy tờ (jpg, png)", 
    type=["jpg", "jpeg", "png"]
)

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""


# ===========================
# OCR PROCESSING
# ===========================
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼 Ảnh đã upload", use_column_width=True)

    st.write("🔍 Đang chạy OCR... vui lòng chờ")

    result = ocr.ocr(np.array(img), cls=True)

    text = "\n".join([line[1][0] for line in result[0]])
    st.session_state.ocr_text = text

    st.subheader("📌 Kết quả OCR:")
    st.write(text)

    st.divider()


# ===========================
# CHATBOT QA USING OFFLINE LLM
# ===========================
if st.session_state.ocr_text:
    st.subheader("💬 Hỏi AI về nội dung OCR")

    query = st.text_input("Nhập câu hỏi:")

    if query:
        prompt = f"""
Bạn là trợ lý AI thông minh.
Dưới đây là văn bản OCR trích từ ảnh:

{text}

Câu hỏi: {query}

Hãy trả lời chi tiết và chính xác.
"""

        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.1
        )

        answer = output["choices"][0]["text"]

        st.subheader("🤖 Trả lời:")
        st.write(answer)

else:
    st.info("⬆️ Hãy upload ảnh để bắt đầu OCR.")
