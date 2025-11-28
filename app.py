import streamlit as st
from PIL import Image
import numpy as np
import easyocr
from llama_cpp import Llama

# Load LLM offline
llm = Llama(
    model_path="models/Phi-3-mini-4k-instruct.Q4_K_M.gguf",
    n_threads=6,
    n_ctx=2048,
)

# Load EasyOCR
reader = easyocr.Reader(['vi', 'en'])

st.set_page_config(page_title="OCR + Offline LLM", layout="wide")
st.title("📄 OCR + 🤖 Chatbot Offline")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    st.write("🔍 Running OCR...")
    results = reader.readtext(np.array(img))

    text = "\n".join([res[1] for res in results])
    st.session_state.ocr_text = text

    st.subheader("📌 OCR text:")
    st.write(text)

    st.divider()

if st.session_state.ocr_text:
    query = st.text_input("Ask the AI:")

    if query:
        prompt = f"""
Dựa trên văn bản OCR sau:
{text}

Câu hỏi: {query}

Trả lời:
"""

        out = llm(prompt, temperature=0.1, max_tokens=256)
        answer = out["choices"][0]["text"]

        st.write("### 🤖 Trả lời:")
        st.write(answer)
else:
    st.info("Hãy upload ảnh trước.")
