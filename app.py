import streamlit as st
import tabula
from docx import Document
import tempfile
import os

st.set_page_config(page_title="PDF → Word (Giữ bảng)", layout="wide")
st.title("📄 Chuyển PDF → Word (GIỮ NGUYÊN DỮ LIỆU BẢNG)")

st.write("Ứng dụng này trích bảng từ PDF và xuất sang Word mà không làm mất dữ liệu.")


# ===============================================
# TRÍCH BẢNG PDF
# ===============================================
def extract_tables(pdf_path):
    dfs = tabula.read_pdf(
        pdf_path,
        pages="all",
        multiple_tables=True,
        stream=True  # đọc theo dòng giữ bảng chính xác hơn
    )
    return dfs


# ===============================================
# TẠO WORD TỪ CÁC BẢNG
# ===============================================
def create_word_from_tables(dfs):
    doc = Document()

    for idx, df in enumerate(dfs):
        doc.add_heading(f"Bảng {idx+1}", level=2)

        table = doc.add_table(rows=1, cols=len(df.columns))
        hdr_cells = table.rows[0].cells

        # Header
        for i, col in enumerate(df.columns):
            hdr_cells[i].text = str(col)

        # Data rows
        for _, row in df.iterrows():
            row_cells = table.add_row().cells
            for i, cell in enumerate(row):
                row_cells[i].text = str(cell)

        doc.add_paragraph("")  # khoảng trắng

    return doc


# ===============================================
# UI
# ===============================================

uploaded = st.file_uploader("📤 Tải file PDF", type="pdf")

if uploaded:
    st.success("PDF đã tải lên!")

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(uploaded.read())
    temp_pdf.close()

    if st.button("🔍 Trích bảng"):
        with st.spinner("Đang phân tích PDF…"):
            tables = extract_tables(temp_pdf.name)

        if not tables:
            st.error("❌ Không tìm thấy bảng nào trong PDF!")
        else:
            st.success(f"✔ Tìm thấy {len(tables)} bảng!")

            # Hiển thị preview
            for i, df in enumerate(tables):
                st.subheader(f"Bảng {i+1}")
                st.dataframe(df)

            # Tạo Word file
            doc = create_word_from_tables(tables)
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
            doc.save(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Tải file Word",
                    f,
                    file_name="tables_output.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    os.unlink(temp_pdf.name)
