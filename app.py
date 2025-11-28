import streamlit as st
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(
    page_title="OCR + Chatbot Tiếng Việt (Bản Nhẹ)",
    layout="wide"
)

st.title("📄 OCR + 🤖 Chatbot Tiếng Việt – Bản Nhẹ (Streamlit Cloud)")


# ==============================
# LOAD SMALL LLM (VERY LIGHT)
# ==============================
@st.cache_resource
def load_llm():
    model_name = "VietAI/gpt-j-6B-vi-lite"  # model distill nhỏ

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Chạy CPU (Streamlit Cloud không có GPU)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    return tokenizer, model


tokenizer, model = load_llm()


# ==============================
# FUNCTION: QA FROM OCR
# ==============================
def ask_ai(ocr_text, question):

    prompt = f"""
Bạn là trợ lý AI hiểu tiếng Việt.

Dưới đây là văn bản OCR trích từ ảnh:

{ocr_text}

Câu hỏi: {question}

Hãy trả lời ngắn gọn và chính xác.
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        temperature=0.3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# ==============================
# UI
# ==============================
uploaded = st.file_uploader("📤 Chọn ảnh (jpg/png)…", type=["jpg", "jpeg", "png"])

if "ocr" not in st.session_state:
    st.session_state.ocr = ""


# ------------------------------
# OCR VIA TESSERACT (LIGHT)
# ------------------------------
if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)

    if st.button("🔍 Chạy OCR"):
        with st.spinner("Đang chạy OCR…"):
            text = pytesseract.image_to_string(img, lang="vie")
            st.session_state.ocr = text

        st.success("Hoàn tất OCR!")
        st.text_area("📌 Văn bản OCR:", text, height=200)


# ------------------------------
# CHATBOT
# ------------------------------
st.subheader("💬 Hỏi AI dựa trên OCR")

if not st.session_state.ocr:
    st.info("Hãy upload ảnh và chạy OCR trước.")
else:
    q = st.text_input("Nhập câu hỏi của bạn:")

    if st.button("🤖 Trả lời"):
        with st.spinner("AI đang trả lời…"):
            answer = ask_ai(st.session_state.ocr, q)

        st.write("### 🧠 Trả lời:")
        st.write(answer)
