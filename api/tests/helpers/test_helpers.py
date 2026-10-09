from api.helpers.prompts import build_context, format_document
from api.helpers.utils import serialize_documents


def test_serialize_documents_keeps_only_public_fields(make_doc):
    docs = [make_doc(0.9, "text", {"source": "a.pdf", "page": 2, "secret": "x"})]

    assert serialize_documents(docs) == [
        {"score": 0.9, "text": "text", "metadata": {"source": "a.pdf", "page": 2}}
    ]


def test_serialize_documents_handles_missing_metadata(make_doc):
    out = serialize_documents([make_doc(0.5, "t", None)])
    assert out[0]["metadata"] == {"source": None, "page": None}


def test_serialize_empty_list():
    assert serialize_documents([]) == []


def test_format_document_includes_source_page_and_text(make_doc):
    out = format_document(make_doc(0.9, "Body text", {"source": "a.pdf", "page": 3}), 1)

    assert "[Document 1]" in out
    assert "Source: a.pdf" in out
    assert "Page: 3" in out
    assert "Body text" in out


def test_format_document_defaults_to_unknown(make_doc):
    out = format_document(make_doc(0.9, "Body", None), 2)
    assert "Source: Unknown" in out and "Page: Unknown" in out


def test_build_context_numbers_documents_from_one(make_doc):
    ctx = build_context([make_doc(0.9, "first", {}), make_doc(0.8, "second", {})])

    assert "[Document 1]" in ctx and "[Document 2]" in ctx
    assert ctx.index("first") < ctx.index("second")


def test_build_context_empty():
    assert build_context([]) == ""
