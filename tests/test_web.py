from pathlib import Path
from urllib.parse import urlencode

from app.rag import DEFAULT_SAMPLE_PDF, parse_pdf_path_lines
from app.web import _render_markdown, create_app


def test_parse_pdf_path_lines_uses_sample_when_blank():
    paths = parse_pdf_path_lines("")
    assert DEFAULT_SAMPLE_PDF.resolve() in paths
    assert any("esg_scraper/data/pdfs" in str(path) for path in paths)


def test_web_homepage_renders_successfully():
    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(
        app(
            {
                "PATH_INFO": "/",
                "REQUEST_METHOD": "GET",
                "wsgi.input": __import__("io").BytesIO(b""),
                "CONTENT_LENGTH": "0",
            },
            start_response,
        )
    ).decode("utf-8")

    assert captured["status"] == "200 OK"
    assert "ESG RAG Studio" in body
    assert "Build Index" in body


def test_render_markdown_formats_sections_and_tables():
    html = _render_markdown(
        """**Key takeaway**
- Danone has validated 2030 climate targets.

**Detailed answer**
| Metric | Target |
|---|---|
| Scope 1 & 2 | -46.3% |
"""
    )

    assert "<h3>Key takeaway</h3>" in html
    assert "<li>Danone has validated 2030 climate targets.</li>" in html
    assert "<table>" in html
    assert "<th>Metric</th>" in html
    assert "<td>Scope 1 &amp; 2</td>" in html


def test_web_ask_keeps_question_and_renders_markdown(monkeypatch):
    def fake_retrieve_chunks(**kwargs):
        return (
            [
                {
                    "chunk_id": "chunk-0001",
                    "source_file": "report.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.8,
                    "text": "Target evidence.",
                }
            ],
            "embedding-model",
        )

    def fake_answer_question(**kwargs):
        return (
            "**Key takeaway**\n- The target is disclosed.\n\n**Uncertainty**\n- Scope is partial.",
            "text-model",
        )

    monkeypatch.setattr("app.web.retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr("app.web.answer_question", fake_answer_question)

    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    captured = {}
    body_bytes = urlencode({"question": "What are Danone's climate targets?"}).encode("utf-8")

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(
        app(
            {
                "PATH_INFO": "/ask",
                "REQUEST_METHOD": "POST",
                "wsgi.input": __import__("io").BytesIO(body_bytes),
                "CONTENT_LENGTH": str(len(body_bytes)),
            },
            start_response,
        )
    ).decode("utf-8")

    assert captured["status"] == "200 OK"
    assert "What are Danone&#x27;s climate targets?" in body
    assert "<h3>Key takeaway</h3>" in body
    assert "<li>The target is disclosed.</li>" in body
