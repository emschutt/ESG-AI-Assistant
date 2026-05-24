import pytest

from app.ragas_eval import _parse_json_object, extract_cited_chunk_ids


def test_parse_json_object_accepts_plain_json():
    parsed = _parse_json_object('{"score": 1.0, "reasoning": "supported"}')

    assert parsed["score"] == 1.0


def test_parse_json_object_accepts_fenced_json():
    parsed = _parse_json_object('```json\n{"score": 0.75, "reasoning": "mostly supported"}\n```')

    assert parsed["reasoning"] == "mostly supported"


def test_parse_json_object_rejects_non_object_json():
    with pytest.raises(Exception):
        _parse_json_object('["not", "an", "object"]')


def test_extract_cited_chunk_ids_normalizes_typographic_hyphens():
    answer = "Supported by [chunk‑0001], [chunk–0002], [ chunk-0003 ], and 【chunk-0004】."

    assert extract_cited_chunk_ids(answer) == {"chunk-0001", "chunk-0002", "chunk-0003", "chunk-0004"}
