import streamlit as st
from PIL import Image
import numpy as np
import torch
import easyocr
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 1. CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="OCR + LLM Chatbot (Tiếng Việt - Offline)",
    layout="wide"
)

st.title("📄 OCR + 🤖 Chatbot LLM (Tiếng Việt - Offline/Free)")
st.write("Upload ảnh → OCR → hỏi AI dựa trên nội dung trong ảnh.")


# =========================
# 2. LOAD OCR
# =========================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["vi", "en"], gpu=torch.cuda.is_available())

reader = load_ocr()


# =========================
# 3. LOAD LLM (Qwen2.5-1.5B)
# =========================
@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"  # GPU nếu có
    )

    return tokenizer, model

tokenizer, model = load_llm()


# =========================
# 4. SINH TRẢ LỜI TỪ LLM
# =========================
def answer_llm(ocr_text: str, question: str):
    device = model.device

    system_prompt = (
        "Bạn là trợ lý AI hiểu tiếng Việt. "
        "Chỉ dựa vào văn bản OCR được cung cấp, hãy trả lời chính xác – ngắn gọn – rõ ràng."
    )

    prompt = f"""
<|system|>
{system_prompt}
</s>
<|user|>
Văn bản OCR:

{ocr_text}

Câu hỏi: {question}
</s>
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("assistant", 1)[-1].strip()


# =========================
# 5. UI
# =========================

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""


# Upload ảnh
st.subheader("1️⃣ Upload ảnh để OCR")
uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)

    if st.button("🔍 Chạy OCR"):
        with st.spinner("Đang chạy OCR..."):
            ocr_result = reader.readtext(np.array(img))
            txt = "\n".join([r[1] for r in ocr_result])
            st.session_state.ocr_text = txt

        st.success("Hoàn tất OCR!")
        st.text_area("📌 Kết quả OCR:", txt, height=200)


# Chatbot
st.subheader("2️⃣ Hỏi AI dựa trên văn bản OCR")

if not st.session_state.ocr_text:
    st.info("Hãy upload ảnh và chạy OCR trước.")
else:
    q = st.text_input("Nhập câu hỏi:")
    if st.button("🤖 Trả lời"):
        with st.spinner("AI đang suy nghĩ..."):
            ans = answer_llm(st.session_state.ocr_text, q)

        st.markdown("### 💡 Trả lời:")
        st.write(ans)
