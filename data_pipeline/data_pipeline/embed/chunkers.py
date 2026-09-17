from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_pipeline.helpers.config import get_settings

settings = get_settings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
)


def chunck_pages(pages: list[dict], source: str) -> list[dict]:
    chuncks = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"].strip()

        if not page_text:
            continue

        page_chuncks = text_splitter.split_text(page_text)

        for chunck_index, chunck_text in enumerate(page_chuncks):
            chuncks.append(
                {
                    "text": chunck_text,
                    "metadata": {
                        "source": source,
                        "page": page_number,
                        "chunck_id": chunck_index,
                    },
                }
            )

        return chuncks
