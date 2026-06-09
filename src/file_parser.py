from io import BytesIO

from docx import Document
from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()


def extract_text_from_docx(uploaded_file) -> str:
    document = Document(BytesIO(uploaded_file.getvalue()))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    return "\n".join(paragraphs).strip()


def extract_text_from_txt(uploaded_file) -> str:
    raw_text = uploaded_file.getvalue()
    return raw_text.decode("utf-8", errors="replace").strip()


def extract_text_from_uploaded_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    if file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT file.")
