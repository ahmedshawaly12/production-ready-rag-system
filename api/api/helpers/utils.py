def serialize_documents(documents) -> list[dict]:
    result = []
    for document in documents:
        metadata = document.metadata or {}
        result.append(
            {
                "score": document.score,
                "text": document.text,
                "metadata": {
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                },
            }
        )
    return result
