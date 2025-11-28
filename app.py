import streamlit as st
import pdfplumber
from docx import Document
from PIL import Image
import tempfile
import os


st.set_page_config(page_title="PDF → Word (Giữ bảng - No Java)", layout="wide")
st.title("📄 Chuyển PDF → Word (Giữ dữ liệu bảng) – NO JAVA")


# ============================================
# FUNCTION: Extract tables manually
# ============================================
def extract_tables(pdf_path):
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables()

            for table in extracted:
                tables.append(table)

    return tables


# ============================================
# FUNCTION: Convert to Word
# ============================================
def create_word_from_tables(tables):
    doc = Document()

    for index, table in enumerate(tables):
        doc.add_heading(f"Bảng {index + 1}", level=2)

        rows = len(table)
        cols = len(table[0])

        word_table = doc.add_table(rows=rows, cols=cols)

        for r in range(rows):
            for c in range(cols):
                cell_text = table[r][c] if table[r][c] else ""
                word_table.rows[r].cells[c].text = cell_text

        doc.add_paragraph("")

    return doc


# ============================================
# UI
# ============================================
uploaded = st.file_uploader("📤 Chọn file PDF", type="pdf")

if uploaded:
    st.success("PDF đã tải thành công!")

    # Save PDF temp
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(uploaded.read())
    temp_pdf.close()

    if st.button("🔍 Trích bảng"):
        with st.spinner("Đang trích bảng..."):

            tables = extract_tables(temp_pdf.name)

        if not tables:
            st.error("❌ Không có bảng nào trong PDF.")
        else:
            st.success(f"✔ Tìm thấy {len(tables)} bảng!")

            # preview
            for i, table in enumerate(tables):
                st.subheader(f"Bảng {i+1}")
                st.table(table)

            # convert to Word
            doc = create_word_from_tables(tables)
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
            doc.save(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Tải file Word",
                    data=f,
                    file_name="output_tables.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    os.unlink(temp_pdf.name)
