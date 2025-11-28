import streamlit as st
import pdfplumber
from docx import Document
import tempfile
import os

st.set_page_config(page_title="PDF → Word Full", layout="wide")
st.title("📄 Chuyển PDF → Word (Giữ bảng + text) – NO JAVA – CHẠY CLOUD")


# =====================================================
# HÀM: Lấy tất cả block (text + bảng) theo thứ tự
# =====================================================
def parse_pdf(pdf_path):
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            blocks = []

            # --- TEXT BLOCKS ---
            for obj in page.extract_words():
                blocks.append({
                    "type": "text",
                    "y0": obj["top"],
                    "content": obj["text"]
                })

            # --- TABLE BLOCKS ---
            tables = page.extract_tables()
            for tb in tables:
                # estimate top position of table
                try:
                    y0 = page.extract_table({"vertical_strategy": "lines"})[0][0][1]
                except:
                    y0 = 99999

                blocks.append({
                    "type": "table",
                    "y0": y0,
                    "content": tb
                })

            # sort theo vị trí top
            blocks = sorted(blocks, key=lambda x: x["y0"])
            pages_data.append(blocks)

    return pages_data


# =====================================================
# HÀM: GHI vào Word theo đúng thứ tự PDF
# =====================================================
def write_to_word(pdf_data):
    doc = Document()

    for page_idx, blocks in enumerate(pdf_data):
        doc.add_heading(f"Trang {page_idx+1}", level=1)

        for block in blocks:
            if block["type"] == "text":
                doc.add_paragraph(block["content"])

            elif block["type"] == "table":
                table_data = block["content"]

                if table_data and len(table_data) > 0:

                    # tạo bảng Word
                    table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))

                    for r, row in enumerate(table_data):
                        for c, val in enumerate(row):
                            table.rows[r].cells[c].text = str(val) if val else ""

                    doc.add_paragraph("")  # khoảng cách sau bảng

        doc.add_page_break()

    return doc


# =====================================================
# UI
# =====================================================
uploaded = st.file_uploader("📤 Chọn PDF", type="pdf")

if uploaded:
    st.success("Đã tải PDF!")

    # Save PDF tạm
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(uploaded.read())
    temp_pdf.close()

    if st.button("🔄 Chuyển sang Word"):
        with st.spinner("Đang chuyển đổi PDF → Word..."):
            pdf_data = parse_pdf(temp_pdf.name)
            doc = write_to_word(pdf_data)

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
            doc.save(out_path)

        with open(out_path, "rb") as f:
            st.download_button(
                "📥 Tải file Word",
                data=f,
                file_name="converted.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    os.unlink(temp_pdf.name)
