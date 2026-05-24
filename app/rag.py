"""Local PDF-to-RAG pipeline using Albert embeddings and optional FAISS storage."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import requests

from app.utils import OUTPUT_DIR, ensure_directories, normalize_whitespace

DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_INDEX_DIR = OUTPUT_DIR / "rag_index"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DIR = REPO_ROOT / "sample_data"
DEFAULT_SAMPLE_PDF = DEFAULT_SAMPLE_DIR / "totalenergies_sustainability-climate-2024-progress-report_2024_en_pdf.pdf"
DEFAULT_DATABASE_PDF_DIRS = (
    DEFAULT_SAMPLE_DIR,
    OUTPUT_DIR / "downloaded_reports",
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "esg_scraper" / "data" / "pdfs",
)
TOKEN_PATTERN = re.compile(r"\S+")
SEARCH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
VALID_RETRIEVAL_ARCHITECTURES = ("semantic", "hybrid", "semantic_rerank", "dense", "lexical")
SOURCE_FILENAME_STOPWORDS = {
    "accessibleversion",
    "annual",
    "bd",
    "climate",
    "data",
    "databook",
    "document",
    "downloaded",
    "en",
    "esg",
    "integrated",
    "pdf",
    "pdfs",
    "progress",
    "raw",
    "registration",
    "report",
    "reports",
    "results",
    "sustainability",
    "tracker",
    "universal",
}
SOURCE_IDENTIFIER_ALIASES = {
    "herm_s": "hermes",
    "l_or_al": "loreal oreal",
    "l_oreal": "loreal oreal",
    "nestl": "nestle",
}


def _normalize_retrieval_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode in ("dense", "semantic"):
        return "semantic"
    if mode == "lexical":
        return "lexical"
    if mode == "hybrid":
        return "hybrid"
    if mode == "semantic_rerank":
        return "semantic_rerank"
    supported = ", ".join(VALID_RETRIEVAL_ARCHITECTURES)
    raise RuntimeError(f"Unknown retrieval architecture '{mode}'. Choose from: {supported}.")

try:  # pragma: no cover - availability depends on the local environment.
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback path is covered instead.
    faiss = None


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    source_path: str
    page_start: int
    page_end: int
    token_count: int
    text: str
    contextual_summary: str = ""
    esg_pillar: str = ""
    section_title: str = ""
    report_year: str = ""
    contains_table: bool = False
    contains_targets: bool = False
    # Defaults keep `ChunkRecord(**item)` working on chunks.json built before
    # these fields were populated; the converter fills them from metadata.jsonl.
    company: str = ""
    speaker_role: str = ""
    chunk_kind: str = ""
    doc_type: str = ""


class AlbertClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 120) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        retryable_statuses = {429, 500, 502, 503, 504}
        delay_seconds = 2.0

        for attempt in range(1, 7):
            response = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
            if response.status_code not in retryable_statuses or attempt == 6:
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After", "").strip()
            try:
                wait_seconds = float(retry_after) if retry_after else delay_seconds
            except ValueError:
                wait_seconds = delay_seconds
            time.sleep(min(max(wait_seconds, 1.0), 30.0))
            delay_seconds = min(delay_seconds * 2.0, 30.0)

        raise RuntimeError("Albert request retry loop exited unexpectedly.")

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/models").json().get("data", [])

    def get_embedding_model(self, preferred: str | None = "bge-m3") -> str:
        models = self.list_models()
        if not models:
            raise RuntimeError("Albert returned no models.")

        if preferred:
            preferred_lower = preferred.lower()
            for model in models:
                identifier = str(model.get("id", ""))
                if preferred_lower in identifier.lower():
                    return identifier

        embedding_types = {"embedding", "embeddings", "text-embedding", "text-embeddings"}
        for model in models:
            model_type = str(model.get("type", "")).lower()
            if model_type in embedding_types:
                return str(model["id"])

        for model in models:
            identifier = str(model.get("id", ""))
            if "embed" in identifier.lower():
                return identifier

        raise RuntimeError("No Albert embedding model was returned by /models.")

    def get_text_generation_model(self) -> str:
        for model in self.list_models():
            if str(model.get("type", "")).lower() == "text-generation":
                return str(model["id"])
        raise RuntimeError("No Albert text-generation model was returned by /models.")

    def create_embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        payload = {"model": model, "input": inputs}
        response = self._request("POST", "/embeddings", json=payload).json()
        data = response.get("data") or []
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in ordered]
        if not embeddings or any(not isinstance(embedding, list) for embedding in embeddings):
            raise RuntimeError("Albert returned an unexpected embeddings payload.")
        return embeddings

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if temperature is not None:
            payload["temperature"] = temperature
        response = self._request(
            "POST",
            "/chat/completions",
            json=payload,
        ).json()
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Albert returned no chat completion choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "\n".join(part for part in text_parts if part).strip()
        raise RuntimeError("Albert returned an unexpected chat completion payload.")


def require_api_key() -> str:
    api_key = os.environ.get("ALBERT_API_KEY")
    if not api_key:
        raise RuntimeError("Set ALBERT_API_KEY in your environment before running the RAG commands.")
    return api_key


def set_api_key(api_key: str | None) -> None:
    if api_key:
        os.environ["ALBERT_API_KEY"] = api_key.strip()


def estimate_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    words = TOKEN_PATTERN.findall(text)
    if len(words) <= max_tokens:
        return [normalize_whitespace(text)]

    window_count = (len(words) + max_tokens - 1) // max_tokens
    window_size = min(max_tokens, (len(words) + window_count - 1) // window_count)
    windows: list[str] = []
    for start in range(0, len(words), window_size):
        window = words[start : start + window_size]
        if window:
            windows.append(" ".join(window))
    return windows


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            text = raw_text.strip()
            if text:
                pages.append(PageText(page_number=index, text=text))
    return pages


def _starts_with_header(text: str) -> bool:
    first_line = text.split("\n", 1)[0].strip()
    if not first_line or len(first_line) > 120:
        return False
    if re.match(r"^[\dIVXivx]+[\.\)]\s", first_line):
        return True
    if first_line.isupper() and len(first_line.split()) <= 8:
        return True
    return False


def _build_units(pages: list[PageText], max_tokens: int) -> list[tuple[int, str, int]]:
    units: list[tuple[int, str, int]] = []
    for page in pages:
        paragraphs = [normalize_whitespace(part) for part in re.split(r"\n\s*\n", page.text)]
        paragraphs = [part for part in paragraphs if part]
        if not paragraphs:
            paragraphs = [page.text]
        for paragraph in paragraphs:
            token_count = estimate_token_count(paragraph)
            for window in _split_long_text(paragraph, max_tokens=max_tokens):
                units.append((page.page_number, window, estimate_token_count(window)))
    return units


def _detect_esg_pillar(text: str) -> str:
    lower = text.lower()
    env_keywords = ["emission", "carbon", "climate", "ghg", "scope 1", "scope 2", "scope 3", "energy", "biodiversity", "water", "waste", "pollution", "renewable"]
    social_kw = ["employee", "safety", "trir", "diversity", "inclusion", "community", "worker", "human right", "labor", "stem", "volunteering", "health"]
    gov_kw = ["board", "committee", "risk", "compliance", "audit", "tcfd", "taxonomy", "governance", "director", "executive", "shareholder", "ethics"]
    env_score = sum(1 for kw in env_keywords if kw in lower)
    soc_score = sum(1 for kw in social_kw if kw in lower)
    gov_score = sum(1 for kw in gov_kw if kw in lower)
    if env_score > soc_score and env_score > gov_score:
        return "environmental"
    if soc_score > env_score and soc_score > gov_score:
        return "social"
    if gov_score > env_score and gov_score > soc_score:
        return "governance"
    return ""


def _detect_section_title(chunk_text: str) -> str:
    lines = chunk_text.strip().split("\n")
    for line in lines[:5]:
        stripped = line.strip()
        if stripped and len(stripped) < 100 and (
            stripped.isupper() or re.match(r"^[\dIVXivx]+[\.\)\s]", stripped)
        ):
            return stripped
    return ""


def _detect_contains_table(text: str) -> bool:
    lines = text.split("\n")
    pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
    number_dense_lines = sum(1 for line in lines if len(re.findall(r"\d+", line)) >= 3)
    return pipe_lines >= 2 or number_dense_lines >= 3


def _detect_contains_targets(text: str) -> bool:
    target_patterns = [
        r"target.*20\d{2}", r"objective.*20\d{2}", r"goal.*20\d{2}",
        r"reduce.*by.*20\d{2}", r"achieve.*by.*20\d{2}", r"commit.*to.*20\d{2}",
    ]
    lower = text.lower()
    return any(re.search(pat, lower) for pat in target_patterns)


def _extract_report_year(source_file: str) -> str:
    years = re.findall(r"(20\d{2})", source_file)
    return years[-1] if years else ""


def _generate_contextual_summary(chunk_text: str) -> str:
    section = _detect_section_title(chunk_text)
    pillar = _detect_esg_pillar(chunk_text)
    first_sentence = chunk_text.split(".")[0][:120].strip() if "." in chunk_text else chunk_text[:120].strip()
    parts = []
    if section:
        parts.append(f"Section: {section}")
    if pillar:
        parts.append(f"ESG Pillar: {pillar}")
    parts.append(first_sentence)
    return " | ".join(parts)


def _enrich_chunk_metadata(chunk: ChunkRecord) -> ChunkRecord:
    chunk.esg_pillar = _detect_esg_pillar(chunk.text)
    chunk.section_title = _detect_section_title(chunk.text)
    chunk.report_year = _extract_report_year(chunk.source_file)
    chunk.contains_table = _detect_contains_table(chunk.text)
    chunk.contains_targets = _detect_contains_targets(chunk.text)
    return chunk


def _finalize_chunk(units: list[tuple[int, str, int]], chunk_number: int, source_file: str, source_path: str) -> ChunkRecord:
    chunk_text = "\n\n".join(text for _, text, _ in units).strip()
    page_numbers = [page_number for page_number, _, _ in units]
    chunk = ChunkRecord(
        chunk_id=f"chunk-{chunk_number:04d}",
        source_file=source_file,
        source_path=source_path,
        page_start=min(page_numbers),
        page_end=max(page_numbers),
        token_count=estimate_token_count(chunk_text),
        text=chunk_text,
    )
    return _enrich_chunk_metadata(chunk)


def _merge_chunk_records(first: ChunkRecord, second: ChunkRecord, chunk_id: str) -> ChunkRecord:
    merged_text = f"{first.text}\n\n{second.text}".strip()
    return ChunkRecord(
        chunk_id=chunk_id,
        source_file=first.source_file,
        source_path=first.source_path,
        page_start=min(first.page_start, second.page_start),
        page_end=max(first.page_end, second.page_end),
        token_count=estimate_token_count(merged_text),
        text=merged_text,
    )


def _compact_small_chunks(chunks: list[ChunkRecord], min_tokens: int, max_tokens: int) -> list[ChunkRecord]:
    compacted: list[ChunkRecord] = []
    soft_max_tokens = max_tokens + 50
    index = 0
    while index < len(chunks):
        current = chunks[index]

        while (
            current.token_count < min_tokens
            and index + 1 < len(chunks)
            and chunks[index + 1].source_file == current.source_file
        ):
            candidate = chunks[index + 1]
            combined = estimate_token_count(f"{current.text}\n\n{candidate.text}")
            if combined <= max_tokens or (
                current.token_count < (min_tokens // 2) and combined <= soft_max_tokens
            ):
                current = _merge_chunk_records(current, candidate, chunk_id=current.chunk_id)
                index += 1
                continue
            break

        if (
            compacted
            and current.token_count < min_tokens
            and compacted[-1].source_file == current.source_file
        ):
            combined = estimate_token_count(f"{compacted[-1].text}\n\n{current.text}")
            if combined <= max_tokens or (
                current.token_count < (min_tokens // 2) and combined <= soft_max_tokens
            ):
                previous = compacted.pop()
                current = _merge_chunk_records(previous, current, chunk_id=previous.chunk_id)

        compacted.append(current)
        index += 1

    for chunk_number, chunk in enumerate(compacted, start=1):
        chunk.chunk_id = f"chunk-{chunk_number:04d}"
    return compacted


def _tail_units_by_tokens(units: list[tuple[int, str, int]], overlap_tokens: int) -> list[tuple[int, str, int]]:
    if overlap_tokens <= 0:
        return []

    collected: list[tuple[int, str, int]] = []
    remaining_tokens = overlap_tokens
    for unit in reversed(units):
        page_number, text, token_count = unit
        if token_count <= remaining_tokens:
            collected.append(unit)
            remaining_tokens -= token_count
        else:
            words = TOKEN_PATTERN.findall(text)
            tail_words = words[-remaining_tokens:] if remaining_tokens > 0 else []
            if tail_words:
                tail_text = " ".join(tail_words)
                collected.append((page_number, tail_text, estimate_token_count(tail_text)))
            break
        if remaining_tokens <= 0:
            break
    return list(reversed(collected))


def chunk_pages(
    pages: list[PageText],
    *,
    source_file: str,
    source_path: str,
    target_tokens: int = 420,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
    section_aware: bool = False,
) -> list[ChunkRecord]:
    units = _build_units(pages, max_tokens=max_tokens)
    if not units:
        return []

    chunks: list[ChunkRecord] = []
    current_units: list[tuple[int, str, int]] = []
    current_tokens = 0

    for unit in units:
        page_number, text, token_count = unit
        if not current_units:
            current_units.append((page_number, text, token_count))
            current_tokens = token_count
            continue

        force_break = (
            section_aware
            and current_tokens >= min_tokens
            and _starts_with_header(text)
        )

        if not force_break:
            should_append = current_tokens + token_count <= max_tokens and (
                current_tokens < target_tokens or current_tokens < min_tokens
            )
            if should_append:
                current_units.append((page_number, text, token_count))
                current_tokens += token_count
                continue

        chunks.append(
            _finalize_chunk(
                current_units,
                chunk_number=len(chunks) + 1,
                source_file=source_file,
                source_path=source_path,
            )
        )
        overlap_units = _tail_units_by_tokens(current_units, overlap_tokens)
        if sum(unit[2] for unit in overlap_units) >= max_tokens:
            overlap_units = [current_units[-1]]
        current_units = [*overlap_units, (page_number, text, token_count)]
        current_tokens = sum(unit[2] for unit in current_units)

    if current_units:
        chunks.append(
            _finalize_chunk(
                current_units,
                chunk_number=len(chunks) + 1,
                source_file=source_file,
                source_path=source_path,
            )
        )

    return chunks


def chunk_text(
    text: str,
    *,
    source_file: str = "document.txt",
    source_path: str = "document.txt",
    target_tokens: int = 420,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
    section_aware: bool = False,
) -> list[ChunkRecord]:
    pages = [PageText(page_number=1, text=text)]
    return chunk_pages(
        pages,
        source_file=source_file,
        source_path=source_path,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        section_aware=section_aware,
    )


def gather_pdf_paths(pdf_paths: list[str] | None) -> list[Path]:
    if pdf_paths:
        resolved = [Path(path).expanduser().resolve() for path in pdf_paths]
        return resolved

    discovered: list[Path] = []
    seen: set[Path] = set()
    for directory in DEFAULT_DATABASE_PDF_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.pdf")):
            resolved = path.resolve()
            if resolved not in seen:
                discovered.append(resolved)
                seen.add(resolved)

    if discovered:
        return discovered

    if DEFAULT_SAMPLE_PDF.exists():
        return [DEFAULT_SAMPLE_PDF.resolve()]

    return []


def parse_pdf_path_lines(raw_value: str) -> list[Path]:
    values = [line.strip() for line in raw_value.splitlines() if line.strip()]
    return gather_pdf_paths(values or None)


def get_index_summary(index_dir: Path) -> dict[str, Any] | None:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_chunk_records(
    pdf_paths: list[Path],
    *,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int = 0,
    section_aware: bool = False,
    contextual_chunking: bool = False,
    skipped_pdfs: list[dict[str, str]] | None = None,
) -> list[ChunkRecord]:
    chunk_records: list[ChunkRecord] = []
    for pdf_path in pdf_paths:
        try:
            pages = extract_pdf_pages(pdf_path)
        except Exception as exc:
            if skipped_pdfs is not None:
                skipped_pdfs.append({"path": str(pdf_path), "reason": str(exc)})
            continue
        chunk_records.extend(
            chunk_pages(
                pages,
                source_file=pdf_path.name,
                source_path=str(pdf_path),
                target_tokens=target_tokens,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                section_aware=section_aware,
            )
        )

    chunk_records = _compact_small_chunks(chunk_records, min_tokens=min_tokens, max_tokens=max_tokens)

    if contextual_chunking:
        for chunk in chunk_records:
            chunk.contextual_summary = _generate_contextual_summary(chunk.text)
            chunk = _enrich_chunk_metadata(chunk)

    return chunk_records


def _normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(
    client: AlbertClient,
    texts: list[str],
    *,
    embedding_model: str,
    batch_size: int = 16,
) -> np.ndarray:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(client.create_embeddings(embedding_model, batch))
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.size == 0:
        raise RuntimeError("No embeddings were generated for the chunks.")
    return _normalize_embeddings(vectors)


def search_vectors(query_vector: np.ndarray, vectors: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    if vectors.ndim != 2 or vectors.size == 0:
        return []
    normalized_query = _normalize_embeddings(query_vector.reshape(1, -1))[0]
    normalized_vectors = _normalize_embeddings(vectors)
    scores = normalized_vectors @ normalized_query
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(index), float(scores[index])) for index in top_indices]


def _tokenize_for_search(text: str) -> list[str]:
    return SEARCH_TOKEN_PATTERN.findall(text.lower())


def _tokens_for_source_identifier(source_file: str, source_path: str = "") -> set[str]:
    source_parts = [Path(source_file).stem]
    if source_path:
        source = Path(source_path)
        source_parts.extend(part for part in source.parts[-4:-1] if part)
    raw_identifier = " ".join(source_parts).lower()
    normalized_identifier = re.sub(r"[^a-z0-9]+", "_", raw_identifier)
    alias_text = " ".join(
        aliases
        for slug, aliases in SOURCE_IDENTIFIER_ALIASES.items()
        if slug in normalized_identifier
    )
    stem = f"{raw_identifier} {alias_text}".replace("_", " ").replace("-", " ")
    tokens = set(_tokenize_for_search(stem))
    return {
        token
        for token in tokens
        if len(token) > 2
        and token not in SOURCE_FILENAME_STOPWORDS
        and not re.fullmatch(r"20\d{2}", token)
    }


def _infer_question_source_paths(question: str, chunks: list[ChunkRecord]) -> set[str]:
    question_tokens = set(_tokenize_for_search(question))
    if not question_tokens:
        return set()

    scores: dict[str, int] = {}
    for chunk in chunks:
        source_tokens = _tokens_for_source_identifier(chunk.source_file, chunk.source_path)
        overlap = source_tokens & question_tokens
        if overlap:
            scores[chunk.source_path] = max(scores.get(chunk.source_path, 0), len(overlap))

    if not scores:
        return set()

    best_score = max(scores.values())
    return {source_path for source_path, score in scores.items() if score == best_score}


def _normalize_match_scores(matches: list[tuple[int, float]]) -> dict[int, float]:
    if not matches:
        return {}

    scores = [score for _, score in matches]
    max_score = max(scores)
    min_score = min(scores)
    if math.isclose(max_score, min_score):
        return {index: 1.0 for index, _ in matches}
    span = max_score - min_score
    return {index: (score - min_score) / span for index, score in matches}


def _score_lexical_matches(
    question: str,
    chunks: list[ChunkRecord],
    *,
    candidate_indices: list[int] | None = None,
) -> dict[int, float]:
    query_terms = Counter(_tokenize_for_search(question))
    if not query_terms:
        return {}

    selected_indices = candidate_indices or list(range(len(chunks)))
    doc_term_counts: dict[int, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    document_lengths: dict[int, int] = {}

    for index in selected_indices:
        terms = Counter(_tokenize_for_search(chunks[index].text))
        if not terms:
            continue
        doc_term_counts[index] = terms
        document_lengths[index] = sum(terms.values())
        for term in terms:
            document_frequency[term] += 1

    if not doc_term_counts:
        return {}

    avg_length = sum(document_lengths.values()) / len(document_lengths)
    k1 = 1.2
    b = 0.75
    scores: dict[int, float] = {}

    for index, term_counts in doc_term_counts.items():
        doc_length = document_lengths[index]
        score = 0.0
        for term, query_count in query_terms.items():
            term_frequency = term_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            doc_frequency = document_frequency[term]
            idf = math.log(1.0 + ((len(doc_term_counts) - doc_frequency + 0.5) / (doc_frequency + 0.5)))
            length_penalty = 1.0 - b + b * (doc_length / max(avg_length, 1.0))
            numerator = term_frequency * (k1 + 1.0)
            denominator = term_frequency + (k1 * length_penalty)
            score += idf * (numerator / max(denominator, 1e-6)) * (0.5 + 0.5 * min(query_count, 3))
        if score > 0:
            scores[index] = score
    return scores


def _rank_score_map(scores: dict[int, float], limit: int) -> list[tuple[int, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def select_chunk_matches(
    *,
    question: str,
    chunks: list[ChunkRecord],
    semantic_matches: list[tuple[int, float]],
    top_k: int,
    retrieval_architecture: str = "semantic",
    search_breadth: int | None = None,
    candidate_indices: list[int] | None = None,
) -> list[tuple[int, float]]:
    architecture = _normalize_retrieval_mode(retrieval_architecture)

    breadth = max(top_k, search_breadth or top_k)
    semantic_matches = semantic_matches[:breadth]
    if architecture == "semantic":
        return semantic_matches[:top_k]

    if architecture == "lexical":
        lexical_matches = _rank_score_map(
            _score_lexical_matches(question, chunks, candidate_indices=candidate_indices),
            breadth,
        )
        return lexical_matches[:top_k]

    if architecture == "semantic_rerank":
        candidate_indices = [index for index, _ in semantic_matches]
        lexical_matches = _rank_score_map(
            _score_lexical_matches(question, chunks, candidate_indices=candidate_indices),
            breadth,
        )
        lexical_norm = _normalize_match_scores(lexical_matches)
        semantic_norm = _normalize_match_scores(semantic_matches)
        reranked = [
            (
                index,
                (0.7 * semantic_norm.get(index, 0.0)) + (0.3 * lexical_norm.get(index, 0.0)),
            )
            for index in candidate_indices
        ]
        return sorted(reranked, key=lambda item: item[1], reverse=True)[:top_k]

    lexical_matches = _rank_score_map(
        _score_lexical_matches(question, chunks, candidate_indices=candidate_indices),
        breadth,
    )
    fused_scores: Counter[int] = Counter()
    for ranking in (semantic_matches, lexical_matches):
        for rank, (index, _score) in enumerate(ranking, start=1):
            fused_scores[index] += 1.0 / (60.0 + rank)

    semantic_lookup = dict(semantic_matches)
    lexical_lookup = dict(lexical_matches)
    ranked_indices = sorted(
        fused_scores,
        key=lambda index: (
            fused_scores[index],
            semantic_lookup.get(index, 0.0),
            lexical_lookup.get(index, 0.0),
        ),
        reverse=True,
    )
    return [(index, float(fused_scores[index])) for index in ranked_indices[:top_k]]


def _load_chunks(index_dir: Path) -> list[ChunkRecord]:
    payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    return [ChunkRecord(**item) for item in payload]


def _store_index(index_dir: Path, vectors: np.ndarray) -> str:
    np.save(index_dir / "vectors.npy", vectors)
    if faiss is None:
        return "numpy"

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.copy())
    faiss.write_index(index, str(index_dir / "index.faiss"))
    return "faiss"


def _load_vector_backend(index_dir: Path, backend: str) -> Any:
    if backend == "faiss" and faiss is not None and (index_dir / "index.faiss").exists():
        return faiss.read_index(str(index_dir / "index.faiss"))
    return np.load(index_dir / "vectors.npy")


def build_index(
    *,
    pdf_paths: list[Path],
    index_dir: Path,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int = 0,
    section_aware: bool = False,
    contextual_chunking: bool = False,
    batch_size: int,
    embedding_model: str | None = None,
    dry_run: bool = False,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    ensure_directories()
    index_dir.mkdir(parents=True, exist_ok=True)

    skipped_pdfs: list[dict[str, str]] = []
    chunks = build_chunk_records(
        pdf_paths,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        section_aware=section_aware,
        contextual_chunking=contextual_chunking,
        skipped_pdfs=skipped_pdfs,
    )
    if not chunks:
        raise RuntimeError("No extractable text was found in the provided PDFs.")

    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "built_at": datetime.now(UTC).isoformat(),
        "pdfs": [str(path) for path in pdf_paths],
        "skipped_pdfs": skipped_pdfs,
        "chunk_count": len(chunks),
        "target_tokens": target_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "section_aware": section_aware,
        "contextual_chunking": contextual_chunking,
        "vector_backend": "none",
    }

    if dry_run:
        (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_embedding_model = embedding_model or client.get_embedding_model(preferred="bge-m3")

    if contextual_chunking:
        texts_to_embed = []
        for chunk in chunks:
            summary = chunk.contextual_summary
            if summary:
                texts_to_embed.append(f"[Context: {summary}]\n\n{chunk.text}")
            else:
                texts_to_embed.append(chunk.text)
    else:
        texts_to_embed = [chunk.text for chunk in chunks]

    vectors = embed_texts(
        client,
        texts_to_embed,
        embedding_model=selected_embedding_model,
        batch_size=batch_size,
    )
    vector_backend = _store_index(index_dir, vectors)

    manifest.update(
        {
            "embedding_model": selected_embedding_model,
            "embedding_dimension": int(vectors.shape[1]),
            "vector_backend": vector_backend,
        }
    )
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _filter_chunks(
    chunks: list[dict[str, Any]],
    *,
    filter_year: str | None = None,
    filter_pillar: str | None = None,
    filter_report_type: str | None = None,
    filter_company: str | None = None,
    filter_speaker_role: str | None = None,
    filter_chunk_kind: str | None = None,
    filter_doc_type: str | None = None,
) -> list[dict[str, Any]]:
    filtered = chunks
    if filter_year:
        filtered = [c for c in filtered if filter_year in c.get("source_file", "")]
    if filter_pillar and filter_pillar != "all":
        pillar_keywords = {
            "environmental": ["emission", "carbon", "climate", "ghg", "scope", "energy", "biodiversity"],
            "social": ["employee", "safety", "trir", "diversity", "inclusion", "community", "worker"],
            "governance": ["board", "committee", "risk", "compliance", "audit", "tcfd", "taxonomy"],
        }
        keywords = pillar_keywords.get(filter_pillar, [])
        filtered = [c for c in filtered if any(kw in c["text"].lower() for kw in keywords)]

    def _ci_equal(field_value: Any, requested: str) -> bool:
        if field_value is None:
            return False
        return str(field_value).strip().lower() == requested.strip().lower()

    if filter_company:
        filtered = [c for c in filtered if _ci_equal(c.get("company"), filter_company)]
    if filter_speaker_role:
        filtered = [c for c in filtered if _ci_equal(c.get("speaker_role"), filter_speaker_role)]
    if filter_chunk_kind:
        filtered = [c for c in filtered if _ci_equal(c.get("chunk_kind"), filter_chunk_kind)]
    if filter_doc_type:
        filtered = [c for c in filtered if _ci_equal(c.get("doc_type"), filter_doc_type)]
    return filtered


def _chunk_matches_filters(
    chunk: ChunkRecord,
    *,
    allowed_source_paths: set[str] | None = None,
    filter_year: str | None = None,
    filter_pillar: str | None = None,
    filter_report_type: str | None = None,
    filter_company: str | None = None,
    filter_speaker_role: str | None = None,
    filter_chunk_kind: str | None = None,
    filter_doc_type: str | None = None,
) -> bool:
    if allowed_source_paths and chunk.source_path not in allowed_source_paths:
        return False

    if filter_year and filter_year not in chunk.source_file and filter_year != chunk.report_year:
        return False

    if filter_pillar and filter_pillar != "all":
        pillar = chunk.esg_pillar
        if not pillar:
            pillar = _detect_esg_pillar(chunk.text)
        if pillar != filter_pillar:
            return False

    # Reserved for future report-type metadata. Keep the argument accepted so
    # callers can pass it without losing compatibility.
    if filter_report_type:
        if filter_report_type.lower() not in chunk.source_file.lower():
            return False

    # Fine-grained exact-match filters (case-insensitive) on the four metadata
    # fields populated by the converter from metadata.jsonl.
    def _ci_equal(field_value: Any, requested: str) -> bool:
        if field_value is None:
            return False
        return str(field_value).strip().lower() == requested.strip().lower()

    if filter_company and not _ci_equal(chunk.company, filter_company):
        return False
    if filter_speaker_role and not _ci_equal(chunk.speaker_role, filter_speaker_role):
        return False
    if filter_chunk_kind and not _ci_equal(chunk.chunk_kind, filter_chunk_kind):
        return False
    if filter_doc_type and not _ci_equal(chunk.doc_type, filter_doc_type):
        return False

    return True


def _search_vectors_for_indices(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    indices: list[int],
    top_k: int,
) -> list[tuple[int, float]]:
    if not indices or vectors.ndim != 2 or vectors.size == 0:
        return []
    normalized_query = _normalize_embeddings(query_vector.reshape(1, -1))[0]
    normalized_vectors = _normalize_embeddings(vectors[indices])
    scores = normalized_vectors @ normalized_query
    order = np.argsort(scores)[::-1][:top_k]
    return [(indices[int(position)], float(scores[int(position)])) for position in order]


def retrieve_chunks(
    *,
    index_dir: Path,
    question: str,
    top_k: int,
    embedding_model: str | None = None,
    retrieval_architecture: str = "semantic",
    search_breadth: int | None = None,
    base_url: str = DEFAULT_BASE_URL,
    filter_year: str | None = None,
    filter_pillar: str | None = None,
    filter_report_type: str | None = None,
    filter_company: str | None = None,
    filter_speaker_role: str | None = None,
    filter_chunk_kind: str | None = None,
    filter_doc_type: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    vector_backend = manifest.get("vector_backend", "none")
    if vector_backend == "none":
        raise RuntimeError("This index was built with --dry-run and has no embeddings to search.")

    chunks = _load_chunks(index_dir)
    allowed_source_paths = _infer_question_source_paths(question, chunks)
    allowed_indices = [
        index
        for index, chunk in enumerate(chunks)
        if _chunk_matches_filters(
            chunk,
            allowed_source_paths=allowed_source_paths,
            filter_year=filter_year,
            filter_pillar=filter_pillar,
            filter_report_type=filter_report_type,
            filter_company=filter_company,
            filter_speaker_role=filter_speaker_role,
            filter_chunk_kind=filter_chunk_kind,
            filter_doc_type=filter_doc_type,
        )
    ]
    if not allowed_indices:
        return [], embedding_model or str(manifest.get("embedding_model", ""))

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_embedding_model = embedding_model or manifest.get("embedding_model")
    if not selected_embedding_model:
        selected_embedding_model = client.get_embedding_model(preferred="bge-m3")

    query_vector = embed_texts(
        client,
        [question],
        embedding_model=selected_embedding_model,
        batch_size=1,
    )[0]
    breadth = max(top_k, search_breadth or top_k)
    normalized_mode = _normalize_retrieval_mode(retrieval_architecture)
    filters_active = bool(
        allowed_source_paths
        or filter_year
        or filter_pillar
        or filter_report_type
        or filter_company
        or filter_speaker_role
        or filter_chunk_kind
        or filter_doc_type
    )

    if normalized_mode == "semantic" and vector_backend == "faiss" and faiss is not None and not filters_active:
        store = _load_vector_backend(index_dir, vector_backend)
        if hasattr(store, "search"):
            scores, indices = store.search(query_vector.reshape(1, -1), breadth)
            matches = [(int(index), float(score)) for score, index in zip(scores[0], indices[0]) if index >= 0]
        else:
            vectors = np.load(index_dir / "vectors.npy")
            matches = search_vectors(query_vector, vectors, breadth)
    else:
        vectors = np.load(index_dir / "vectors.npy")
        semantic_matches = _search_vectors_for_indices(query_vector, vectors, allowed_indices, breadth)
        matches = select_chunk_matches(
            question=question,
            chunks=chunks,
            semantic_matches=semantic_matches,
            top_k=top_k,
            retrieval_architecture=retrieval_architecture,
            search_breadth=breadth,
            candidate_indices=allowed_indices,
        )

    results: list[dict[str, Any]] = []
    for chunk_index, score in matches:
        chunk = chunks[chunk_index]
        results.append(
            {
                "score": score,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "source_path": chunk.source_path,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "text": chunk.text,
                "contextual_summary": chunk.contextual_summary,
                "esg_pillar": chunk.esg_pillar,
                "section_title": chunk.section_title,
                "report_year": chunk.report_year,
                "contains_table": chunk.contains_table,
                "contains_targets": chunk.contains_targets,
                "company": chunk.company,
                "speaker_role": chunk.speaker_role,
                "chunk_kind": chunk.chunk_kind,
                "doc_type": chunk.doc_type,
            }
        )

    return results, selected_embedding_model


def retrieve_with_transform(
    *,
    index_dir: Path,
    question: str,
    top_k: int = 5,
    retrieval_mode: str = "dense",
    candidate_k: int = 20,
    query_transform: str | None = None,
    reranker: str | None = None,
    embedding_model: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    api_key = require_api_key()
    client = AlbertClient(api_key=api_key, base_url=base_url)

    if not embedding_model:
        embedding_model = client.get_embedding_model(preferred="bge-m3")
    text_model = client.get_text_generation_model()

    if query_transform in ("hyde", "hyde_mqr") or (query_transform == "mqr"):
        from app.query_transforms import generate_hyde_answer, generate_multi_queries, combine_hyde_mqr

        if query_transform == "hyde":
            hyde_answer = generate_hyde_answer(question, client, text_model)
            diagnostics.append({"transform": "hyde", "hyde_answer": hyde_answer})
            all_questions = [hyde_answer]
            weights = [1.0]
        elif query_transform == "mqr":
            mqr_queries = generate_multi_queries(question, client, text_model)
            diagnostics.append({"transform": "mqr", "queries": mqr_queries})
            all_questions = [q["query"] for q in mqr_queries]
            weights = [1.0 / len(all_questions)] * len(all_questions)
        else:
            combined = combine_hyde_mqr(question, client, text_model)
            diagnostics.append({"transform": "hyde_mqr", "combined": combined})
            all_questions = combined["all_query_strings"]
            hyde_weight = 0.4
            mqr_weight = 0.6 / (len(all_questions) - 1) if len(all_questions) > 1 else 0.6
            weights = [hyde_weight] + [mqr_weight] * (len(all_questions) - 1)

        all_retrieved: dict[str, tuple[dict[str, Any], float]] = {}
        for q, weight in zip(all_questions, weights):
            chunks, _ = retrieve_chunks(
                index_dir=index_dir,
                question=q,
                top_k=top_k,
                retrieval_architecture=retrieval_mode,
                search_breadth=candidate_k,
                base_url=base_url,
            )
            for chunk in chunks:
                cid = chunk["chunk_id"]
                if cid in all_retrieved:
                    existing, old_weight = all_retrieved[cid]
                    combined_score = existing["score"] * old_weight + chunk["score"] * weight
                    existing["score"] = combined_score / (old_weight + weight)
                    all_retrieved[cid] = (existing, old_weight + weight)
                else:
                    chunk["score"] *= weight
                    all_retrieved[cid] = (chunk, weight)

        merged = sorted([c for c, _ in all_retrieved.values()], key=lambda c: c["score"], reverse=True)
        results = merged[:top_k]
    else:
        results, _ = retrieve_chunks(
            index_dir=index_dir,
            question=question,
            top_k=min(top_k * 3, candidate_k * 2),
            retrieval_architecture=retrieval_mode,
            search_breadth=candidate_k,
            base_url=base_url,
        )

    if reranker and reranker != "none":
        from app.reranker import rerank_candidates
        results = rerank_candidates(
            question=question,
            candidates=results,
            top_k=top_k,
            reranker=reranker,
            client=client,
            embedding_model=embedding_model,
            text_model=text_model,
        )
        diagnostics.append({"reranker": reranker, "reranked_count": len(results)})

    if not query_transform:
        results = results[:top_k]

    return results[:top_k], embedding_model, diagnostics


def answer_question(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    text_model: str | None = None,
    temperature: float | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[str, str]:
    if not retrieved_chunks:
        raise RuntimeError("No retrieved chunks were provided to the answer stage.")

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_text_model = text_model or client.get_text_generation_model()

    seen_texts: set[str] = set()
    context_blocks: list[str] = []
    for chunk in retrieved_chunks:
        normalized = normalize_whitespace(chunk["text"])
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        context_blocks.append(
            f"[{chunk['chunk_id']}] source={chunk['source_file']} "
            f"pages={chunk['page_start']}-{chunk['page_end']} "
            f"score={chunk['score']:.4f}\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    top_score = max(chunk["score"] for chunk in retrieved_chunks) if retrieved_chunks else 0.0
    weak_retrieval_note = ""
    if top_score < 0.3:
        weak_retrieval_note = (
            "Note: the top retrieval score is low, indicating the context may not "
            "contain sufficient evidence. If you cannot find a clear answer, say so explicitly.\n"
        )

    system_prompt = (
        "You are an ESG analyst producing reliable, reproducible answers from a document database. "
        "Use only the supplied context; never use outside knowledge or infer missing figures.\n"
        "\n"
        "Required output format. Use these exact Markdown section headings, in this order:\n"
        "**Key takeaway**\n"
        "- Give the direct answer in 1-3 short sentences. If the answer is not evidenced, write: "
        "'The provided context does not contain sufficient evidence to answer this question.'\n"
        "\n"
        "**Detailed answer**\n"
        "- Explain the answer with enough detail for an ESG analyst to audit it.\n"
        "- Cite supporting chunk IDs in brackets for every factual claim, e.g. [chunk-0003]. "
        "Copy chunk IDs exactly with the ASCII hyphen character.\n"
        "- When extracting ESG targets or metrics, include metric, value, year, baseline, scope, coverage, "
        "and methodology only when those fields are explicitly present.\n"
        "\n"
        "**Evidence excerpts**\n"
        "- Include 2-5 verbatim snippets copied from the context. Each excerpt must be under 35 words "
        "and followed by its chunk ID.\n"
        "\n"
        "**Uncertainty**\n"
        "- State what is unknown, partial, conflicting, low-scoring, or absent. Do not guess.\n"
        "\n"
        "Keep the tone professional and concise. Avoid unsupported interpretation. If interpretation is "
        "necessary, label it with 'Based on the disclosed information'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {question}\n\n{weak_retrieval_note}Context:\n{context}",
        },
    ]
    return client.chat_completion(selected_text_model, messages, temperature=temperature), selected_text_model


def answer_question_cited(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    text_model: str | None = None,
    temperature: float | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[str, str, dict[int, str], list[dict[str, Any]]]:
    """Variant of `answer_question` that uses numbered citations `[1]`, `[2]`...

    Returns (answer_text, model_id, citation_map, ordered_sources) where
    `citation_map` is `{1: chunk_id, 2: chunk_id, ...}` and `ordered_sources`
    is the deduplicated list of chunk dicts in the same order. This kills the
    `[chunk-NNNN]` / `[tXXXX]` suffix-latching that happens when the model
    sees long chunk_ids and the system-prompt example `[chunk-0003]`.

    The other callers of `answer_question` are left on the original 2-tuple
    signature; only the corpus-RAG CLI / Streamlit paths use this.
    """
    if not retrieved_chunks:
        raise RuntimeError("No retrieved chunks were provided to the answer stage.")

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_text_model = text_model or client.get_text_generation_model()

    seen_texts: set[str] = set()
    ordered_sources: list[dict[str, Any]] = []
    context_blocks: list[str] = []
    citation_map: dict[int, str] = {}

    for chunk in retrieved_chunks:
        normalized = normalize_whitespace(chunk["text"])
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        n = len(ordered_sources) + 1
        ordered_sources.append(chunk)
        citation_map[n] = chunk["chunk_id"]
        company = chunk.get("company") or ""
        doc_type = chunk.get("doc_type") or chunk.get("section_title") or ""
        report_year = chunk.get("report_year") or ""
        header = (
            f"[{n}] {company} · {doc_type} · FY{report_year} · "
            f"{chunk['source_file']} pages {chunk['page_start']}-{chunk['page_end']}"
        )
        context_blocks.append(f"{header}\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    top_score = max(chunk["score"] for chunk in retrieved_chunks) if retrieved_chunks else 0.0
    weak_retrieval_note = ""
    if top_score < 0.3:
        weak_retrieval_note = (
            "Note: the top retrieval score is low, indicating the context may not "
            "contain sufficient evidence. If you cannot find a clear answer, say so explicitly.\n"
        )

    # Same structured-output contract as upstream's answer_question, but with
    # NUMBERED citations [1]/[2]/[3] instead of [chunk-NNNN] (which the model
    # would otherwise truncate to e.g. [chunk-0079] or [t1456] when chunk_ids
    # are long).
    system_prompt = (
        "You are an ESG analyst producing reliable, reproducible answers from a document database. "
        "Use only the supplied context; never use outside knowledge or infer missing figures.\n"
        "\n"
        "Required output format. Use these exact Markdown section headings, in this order:\n"
        "**Key takeaway**\n"
        "- Give the direct answer in 1-3 short sentences. If the answer is not evidenced, write: "
        "'The provided context does not contain sufficient evidence to answer this question.'\n"
        "\n"
        "**Detailed answer**\n"
        "- Explain the answer with enough detail for an ESG analyst to audit it.\n"
        "- Cite supporting passages by their bracketed NUMBER, e.g. [1] or [2], [3]. "
        "Use ONLY the numbers shown in the context headers — do not invent labels.\n"
        "- When extracting ESG targets or metrics, include metric, value, year, baseline, scope, coverage, "
        "and methodology only when those fields are explicitly present.\n"
        "\n"
        "**Evidence excerpts**\n"
        "- Include 2-5 verbatim snippets copied from the context. Each excerpt must be under 35 words "
        "and followed by its bracketed number.\n"
        "\n"
        "**Uncertainty**\n"
        "- State what is unknown, partial, conflicting, low-scoring, or absent. Do not guess.\n"
        "\n"
        "**Attribution rules (apply across every section)**\n"
        "- Attribute each statement to the speaker named in the passage it comes from. "
        "Earnings-call passages name the speaker inline (for example, "
        "'Name [Executives]:'); use that name.\n"
        "- Do NOT assign a role or title (CEO, CFO, COO, Chair, etc.) to any person "
        "unless that exact title is written in the context. If the speaker has no "
        "stated title, refer to them by name, or as 'a company executive' — never "
        "guess or infer a title.\n"
        "- Do not combine statements from different speakers under a single person's "
        "name or title. If several executives spoke, attribute each point to its own speaker.\n"
        "\n"
        "Keep the tone professional and concise. Avoid unsupported interpretation. If interpretation is "
        "necessary, label it with 'Based on the disclosed information'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {question}\n\n{weak_retrieval_note}Context:\n{context}",
        },
    ]
    answer = client.chat_completion(selected_text_model, messages, temperature=temperature)
    return answer, selected_text_model, citation_map, ordered_sources


def run_rag_build(args: Any) -> int:
    pdf_paths = gather_pdf_paths(args.pdf)
    if not pdf_paths:
        raise RuntimeError("No PDF files were found. Pass --pdf or add a PDF under sample_data/.")

    manifest = build_index(
        pdf_paths=pdf_paths,
        index_dir=args.index_dir,
        target_tokens=args.chunk_target_tokens,
        min_tokens=args.chunk_min_tokens,
        max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.chunk_overlap_tokens,
        section_aware=getattr(args, "section_aware", False),
        contextual_chunking=getattr(args, "contextual_chunking", False),
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        dry_run=args.dry_run,
        base_url=args.base_url,
    )
    mode = "dry run" if args.dry_run else manifest["vector_backend"]
    print(f"Indexed {manifest['chunk_count']} chunks from {len(pdf_paths)} PDF(s) into {args.index_dir} ({mode}).")
    if not args.dry_run:
        print(f"Embedding model: {manifest['embedding_model']}")
    return 0


def run_rag_ask(args: Any) -> int:
    retrieval_mode = getattr(args, "retrieval_mode", None) or getattr(args, "retrieval_architecture", "dense")
    query_transform = getattr(args, "query_transform", None)
    reranker = getattr(args, "reranker", None)
    agentic = getattr(args, "agentic", False)
    filter_year = getattr(args, "filter_year", None)
    filter_pillar = getattr(args, "filter_pillar", None)
    filter_company = getattr(args, "filter_company", None)
    filter_speaker_role = getattr(args, "filter_speaker_role", None)
    filter_chunk_kind = getattr(args, "filter_chunk_kind", None)
    filter_doc_type = getattr(args, "filter_doc_type", None)
    candidate_k = getattr(args, "candidate_k", None) or args.search_breadth or 12

    question = " ".join(args.question).strip()

    if agentic:
        from app.agentic import run_agentic_retrieval
        result = run_agentic_retrieval(
            index_dir=str(args.index_dir),
            question=question,
            top_k=args.top_k,
            candidate_k=candidate_k,
            retrieval_mode=retrieval_mode,
            base_url=args.base_url,
        )
        print(f"Agentic retrieval: {result['iterations']} iterations, sufficient={result['sufficient']}")
        for step in result["trace"]:
            print(f"  iter {step['iteration']}: {step['action']} | query='{step['query'][:60]}...' | top_score={step.get('top_score', 0):.4f}")
        retrieved = result["final_chunks"]
        print(f"Answer:\n{result['final_answer']}")
        return 0

    if query_transform or reranker:
        retrieved, embedding_model, diagnostics = retrieve_with_transform(
            index_dir=args.index_dir,
            question=question,
            top_k=args.top_k,
            retrieval_mode=retrieval_mode,
            candidate_k=candidate_k,
            query_transform=query_transform,
            reranker=reranker,
            base_url=args.base_url,
        )
        print(f"Retrieved {len(retrieved)} chunks using {retrieval_mode} (transform={query_transform}, reranker={reranker}):")
        if diagnostics:
            for d in diagnostics:
                print(f"  [{d.get('transform', d.get('reranker', ''))}] {str(d)[:200]}")
    else:
        retrieved, embedding_model = retrieve_chunks(
            index_dir=args.index_dir,
            question=question,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
            retrieval_architecture=retrieval_mode,
            search_breadth=candidate_k,
            base_url=args.base_url,
            filter_year=filter_year,
            filter_pillar=filter_pillar,
            filter_company=filter_company,
            filter_speaker_role=filter_speaker_role,
            filter_chunk_kind=filter_chunk_kind,
            filter_doc_type=filter_doc_type,
        )
        print(
            f"Retrieved {len(retrieved)} chunks using {embedding_model} "
            f"({retrieval_mode}, breadth={candidate_k or args.top_k}):"
        )

    for chunk in retrieved:
        preview = chunk["text"][:220].replace("\n", " ")
        meta = ""
        if chunk.get("esg_pillar"):
            meta += f" [{chunk['esg_pillar']}]"
        if chunk.get("contextual_summary"):
            meta += f" [{chunk['contextual_summary'][:60]}...]"
        print(
            f"- {chunk['chunk_id']} | score={chunk['score']:.4f} | "
            f"{chunk['source_file']} | pages {chunk['page_start']}-{chunk['page_end']}{meta} | {preview}"
        )

    if args.search_only:
        return 0

    if not retrieved:
        print("\n(no chunks retrieved — refusing to call the answer stage)")
        return 0

    answer, text_model, citation_map, ordered_sources = answer_question_cited(
        question=question,
        retrieved_chunks=retrieved,
        text_model=args.text_model,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    print(f"\nAnswer ({text_model}):\n{answer}")
    print("\nCitations:")
    for n, cid in sorted(citation_map.items()):
        src = ordered_sources[n - 1]
        print(
            f"  [{n}] {cid}  "
            f"({src.get('company') or '?'} · {src.get('doc_type') or src.get('section_title') or '?'} · "
            f"FY{src.get('report_year') or '?'}, "
            f"{src.get('source_file', '?')} pages {src.get('page_start', '?')}-{src.get('page_end', '?')}, "
            f"score={src.get('score', 0):.4f})"
        )
    return 0
