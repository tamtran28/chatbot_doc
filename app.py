import streamlit as st
from PIL import Image
import numpy as np
import torch
import easyocr
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================
# CONFIG
# ============================
st.set_page_config(
    page_title="OCR + Chatbot (Tiếng Việt - Streamlit Cloud)",
    layout="wide"
)

st.title("📄 OCR + 🤖 Chatbot Tiếng Việt (Streamlit Cloud - CPU)")


# ============================
# LOAD MODELS (CACHED)
# ============================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["vi", "en"], gpu=False)

reader = load_ocr()


@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    return tokenizer, model

tokenizer, model = load_llm()


# ============================
# GEN ANSWER
# ============================
def generate_answer(ocr_text, question):
    device = "cpu"

    prompt = f"""
Bạn là trợ lý AI hiểu tiếng Việt.

Dưới đây là văn bản OCR lấy từ ảnh:

{ocr_text}

Câu hỏi: {question}

Hãy trả lời ngắn gọn, chính xác.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# ============================
# UI
# ============================
if "ocr" not in st.session_state:
    st.session_state.ocr = ""

uploaded = st.file_uploader("Tải ảnh (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)

    if st.button("🔍 Chạy OCR"):
        with st.spinner("Đang chạy OCR…"):
            result = reader.readtext(np.array(img))
            text = "\n".join([r[1] for r in result])
            st.session_state.ocr = text

        st.text_area("Văn bản OCR:", text, height=200)


st.subheader("💬 Hỏi AI")

if not st.session_state.ocr:
    st.info("Hãy upload ảnh và chạy OCR trước.")
else:
    query = st.text_input("Nhập câu hỏi:")

    if st.button("🤖 Trả lời"):
        with st.spinner("AI đang xử lý…"):
            answer = generate_answer(st.session_state.ocr, query)

        st.write("### 🧠 Trả lời:")
        st.write(answer)
