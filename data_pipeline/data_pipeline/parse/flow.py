import json
from pathlib import PurePosixPath

from botocore.client import BaseClient

from data_pipeline.helpers.config import get_settings
from data_pipeline.helpers.utils import (
    download_bytes,
    get_file_extention,
    get_s3_client,
    list_bucket_folder_files,
    move_file,
    upload_bytes,
)
from data_pipeline.parse.extractors import extract_pdf_text

settings = get_settings()


s3_client = get_s3_client()


def list_files_to_extract(client: BaseClient, bucket: str, path: str) -> list[str]:
    return list_bucket_folder_files(client, bucket, path)


def extract_text(client: BaseClient, bucket: str, key: str):
    file_extention = get_file_extention(key)

    if file_extention not in settings.file_allowed_types:
        print(f"unsupported file format: {key}")
        move_file(client, bucket, key, "unsupported")
        print("moved to unsupported folder")
        return []

    data = download_bytes(client, bucket, key)

    pages = []
    if file_extention == ".txt":
        pages = [{"text": data.decode("utf-8", errors="ignore"), "page": 1}]

    elif file_extention == ".pdf":
        pages_text = extract_pdf_text(data)
        pages = [
            {"page": page_num, "text": text}
            for page_num, text in enumerate(pages_text, start=1)
        ]

    else:
        return []

    # move_file(settings.bucket_name, file_path, "processed")
    return pages


def get_extracted_s3_key(key: str) -> str:
    path = PurePosixPath(key)
    filename = path.stem + ".json"
    return f"extracted/{filename}"


def save_extracted_text(
    client: BaseClient, bucket: str, source_key: str, pages: list[dict]
):
    extracted_key = get_extracted_s3_key(source_key)

    payload = {"source": source_key, "pages": pages}

    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    upload_bytes(client, bucket, extracted_key, data, content_type="application/json")
    return extracted_key


def process_file(key: str):
    pages = extract_text(s3_client, settings.bucket_name, key)

    if not pages:
        return None

    extracted_key = save_extracted_text(s3_client, settings.bucket_name, key, pages)

    print(f"Extracted: {key}")
    print(f"Saved to: {extracted_key}")

    moved_key = move_file(s3_client, settings.bucket_name, key, "processed")
    print(f"moved successfully to {moved_key}")

    return extracted_key


def main():
    files = list_files_to_extract(s3_client, bucket=settings.bucket_name, path="stage")

    print(f"founded {len(files)} files")

    for key in files:
        process_file(key)


if __name__ == "__main__":
    main()
