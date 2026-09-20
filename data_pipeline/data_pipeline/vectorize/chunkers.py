from langchain_text_splitters import RecursiveCharacterTextSplitter
from prefect import task
from prefect.logging import get_run_logger

from data_pipeline.helpers.config import get_settings

settings = get_settings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
)


@task(name="chunck_pages", retries=3, retry_delay_seconds=5)
async def chunck_pages(pages: list[dict], source: str) -> list[dict]:
    logger = get_run_logger()

    chuncks = []

    logger.info(f"Starting chuncking for {source}, {len(pages)} pages")

    for page in pages:
        page_number = page["page"]
        page_text = page["text"].strip()

        if not page_text:
            logger.warning(f"Skiping empty page {page_number} from {source}")
            continue

        page_chuncks = text_splitter.split_text(page_text)
        logger.debug(f"Page {page_number} created {len(page_chuncks)} chuncks")

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

        logger.info(f"Finished chunking {source}: {len(chuncks)} chunk(s) created")

        return chuncks
