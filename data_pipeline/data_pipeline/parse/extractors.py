import pymupdf


def extract_pdf_text(data: bytes) -> list[str]:
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages_text = [page.get_text() for page in doc]
    doc.close()
    return pages_text
