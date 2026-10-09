from string import Template

document_prompt = Template(
    """
[Document ${index}]
Source: ${source}
Page: ${page}

Content:
${content}
"""
)


def format_document(doc, index: int) -> str:
    metadata = doc.metadata or {}

    return document_prompt.safe_substitute(
        index=index,
        source=metadata.get("source", "Unknown"),
        page=metadata.get("page", "Unknown"),
        content=doc.text,
    )


def build_context(documents) -> str:
    return "\n".join(
        format_document(doc, index) for index, doc in enumerate(documents, start=1)
    )
