import streamlit as st
import pdfplumber
from docx import Document
import tempfile
import os

st.set_page_config(page_title="PDF → Word Full", layout="wide")
st.title("📄 Chuyển PDF → Word (Text + Table) – CHẠY ĐƯỢC 100% TRÊN CLOUD")


# ==============================================
# HÀM CHUYỂN PDF → Word
# ==============================================
def pdf_to_word(pdf_path):
    doc = Document()

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):

            doc.add_heading(f"Trang {page_index + 1}", level=1)

            # --- TEXT ---
            text = page.extract_text()
            if text:
                paragraphs = text.split("\n")
                for p in paragraphs:
                    doc.add_paragraph(p)

            doc.add_paragraph("")  # khoảng cách

            # --- TABLES ---
            tables = page.extract_tables()
            for tb_index, table in enumerate(tables):
                doc.add_heading(f"Bảng {tb_index + 1}", level=2)

                row_count = len(table)
                col_count = len(table[0])

                table_doc = doc.add_table(rows=row_count, cols=col_count)

                for r in range(row_count):
                    for c in range(col_count):
                        val = table[r][c] if table[r][c] else ""
                        table_doc.rows[r].cells[c].text = str(val)

                doc.add_paragraph("")

            doc.add_page_break()

    return doc


# ==============================================
# UI
# ==============================================
uploaded = st.file_uploader("📤 Chọn file PDF", type="pdf")

if uploaded:
    st.success("PDF đã tải lên!")

    # Lưu PDF tạm
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(uploaded.read())
    temp_pdf.close()

    if st.button("🔄 Chuyển sang Word"):
        with st.spinner("Đang xử lý PDF → Word..."):
            doc = pdf_to_word(temp_pdf.name)
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
            doc.save(out_path)

        with open(out_path, "rb") as f:
            st.download_button(
                label="📥 Tải file Word",
                data=f,
                file_name="converted.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    os.unlink(temp_pdf.name)
