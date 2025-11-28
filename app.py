import streamlit as st
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================
# STREAMLIT CONFIG
# =========================================
st.set_page_config(page_title="OCR + Chatbot Tiếng Việt", layout="wide")
st.title("📄 OCR + 🤖 Chatbot Tiếng Việt (Bản siêu nhẹ - Streamlit Cloud)")


# =========================================
# LOAD SMALL LLM (FASTEST FOR STREAMLIT)
# =========================================
@st.cache_resource
def load_llm():
    model_name = "vinai/gpt2-vi-small"  # model Việt hóa rất nhẹ

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    return tokenizer, model


tokenizer, model = load_llm()


# =========================================
# AI ANSWER FUNCTION
# =========================================
def ask_ai(ocr_text, question):
    prompt = f"""
Bạn là trợ lý AI giỏi tiếng Việt.

Văn bản OCR từ ảnh:

{ocr_text}

Câu hỏi: {question}

Trả lời:
    """

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# =========================================
# UI
# =========================================
uploaded_image = st.file_uploader("📤 Chọn ảnh (jpg/png)…", type=["jpg", "jpeg", "png"])

if "ocr" not in st.session_state:
    st.session_state.ocr = ""


# =========================================
# OCR USING TESSERACT (LIGHT & CLOUD SAFE)
# =========================================
if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Ảnh đã tải", use_column_width=True)

    if st.button("🔍 Chạy OCR"):
        with st.spinner("Đang xử lý OCR…"):
            text = pytesseract.image_to_string(img, lang="vie")
            st.session_state.ocr = text

        st.success("OCR hoàn tất!")
        st.text_area("📌 Văn bản OCR:", st.session_state.ocr, height=200)


# =========================================
# QA SECTION
# =========================================
st.subheader("💬 Hỏi chatbot dựa trên văn bản OCR")

if not st.session_state.ocr:
    st.info("Hãy upload ảnh và chạy OCR trước.")
else:
    q = st.text_input("Nhập câu hỏi:")

    if st.button("🤖 Trả lời"):
        with st.spinner("AI đang xử lý…"):
            answer = ask_ai(st.session_state.ocr, q)
        st.write("### 🧠 Trả lời:")
        st.write(answer)
