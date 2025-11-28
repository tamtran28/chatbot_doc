import streamlit as st
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(page_title="OCR + Chatbot (Streamlit Cloud)", layout="wide")
st.title("📄 OCR + 🤖 Chatbot Tiếng Việt (Streamlit Cloud • No GPU)")


# ==================================================
# Load LLM
# ==================================================
@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32
    )

    return tokenizer, model


tokenizer, model = load_llm()


# ==================================================
# LLM Answer Function
# ==================================================
def ask_llm(ocr_text, question):

    prompt = f"""
Bạn là một trợ lý AI thông minh và giỏi tiếng Việt.

Dưới đây là văn bản OCR được trích xuất từ hình ảnh:

{ocr_text}

Câu hỏi: {question}

Hãy trả lời rõ ràng và chính xác.
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# ==================================================
# UI
# ==================================================
uploaded = st.file_uploader("Tải ảnh (jpg/png)", type=["jpg", "jpeg", "png"])

if "ocr" not in st.session_state:
    st.session_state.ocr = ""


if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)

    if st.button("🔍 Chạy OCR"):
        with st.spinner("Đang OCR…"):
            text = pytesseract.image_to_string(img, lang="vie")
            st.session_state.ocr = text

        st.text_area("📌 Văn bản OCR:", st.session_state.ocr, height=200)


st.subheader("💬 Hỏi AI dựa trên nội dung OCR")

if not st.session_state.ocr:
    st.info("Hãy upload ảnh và chạy OCR trước.")
else:
    query = st.text_input("Nhập câu hỏi:")

    if st.button("🤖 Trả lời"):
        with st.spinner("AI đang xử lý…"):
            answer = ask_llm(st.session_state.ocr, query)

        st.write("### 🧠 Trả lời:")
        st.write(answer)
