import streamlit as st
import pytesseract
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="OCR + Chatbot Tiếng Việt (Streamlit Cloud)",
    layout="wide"
)

st.title("📄 OCR + 🤖 Chatbot Tiếng Việt (Streamlit Cloud – CPU)")


# =====================================================
# LOAD LLM: Qwen2.5-0.5B-Instruct (CHẠY ĐƯỢC TRÊN CLOUD)
# =====================================================
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


# =====================================================
# FUNCTION: GENERATE ANSWER
# =====================================================
def ask_ai(ocr_text, question):

    prompt = f"""
Bạn là trợ lý AI tiếng Việt.

Văn bản OCR được trích xuất từ hình ảnh:

{ocr_text}

Câu hỏi: {question}

Hãy trả lời chính xác và dễ hiểu.
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =====================================================
# UI
# =====================================================

uploaded = st.file_uploader("📤 Tải ảnh (jpg/png)…", type=["jpg", "jpeg", "png"])

if "ocr" not in st.session_state:
    st.session_state.ocr = ""


# -----------------------------
# OCR BLOCK (TESSERACT)
# -----------------------------
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Ảnh đã tải lên", use_column_width=True)

    if st.button("🔍 Chạy OCR"):
        with st.spinner("Đang chạy OCR…"):
            text = pytesseract.image_to_string(img, lang="vie")
            st.session_state.ocr = text

        st.success("OCR hoàn tất!")
        st.text_area("📌 Văn bản OCR:", text, height=200)


# -----------------------------
# CHATBOT BLOCK
# -----------------------------
st.subheader("💬 Chatbot hỏi đáp dựa trên nội dung OCR")

if not st.session_state.ocr:
    st.info("⚠️ Hãy tải ảnh và chạy OCR trước.")
else:
    q = st.text_input("Nhập câu hỏi:")

    if st.button("🤖 Trả lời"):
        with st.spinner("AI đang xử lý…"):
            answer = ask_ai(st.session_state.ocr, q)

        st.write("### 🧠 Trả lời:")
        st.write(answer)
