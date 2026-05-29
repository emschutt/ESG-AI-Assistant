"""Local PDF-to-RAG pipeline using Albert embeddings and optional FAISS storage."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import requests
import unicodedata

from app.utils import OUTPUT_DIR, ensure_directories, normalize_whitespace

DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_INDEX_DIR = OUTPUT_DIR / "rag_index"
DEFAULT_CHUNK_EXPORT_DIR = OUTPUT_DIR / "rag_dataset_chunks"
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
VALID_ANSWER_MODES = ("assistant", "benchmark")
VALID_PROMPT_STYLES = ("balanced", "extractive", "audit")
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
SOURCE_COMPANY_ALIASES = {
    "airbus": "Airbus",
    "asml": "ASML",
    "bnp": "BNP Paribas",
    "bnpp": "BNP Paribas",
    "bnp_paribas": "BNP Paribas",
    "danone": "Danone",
    "enel": "Enel",
    "engie": "Engie",
    "esrs_sustainability_report_vw_ar24": "Volkswagen",
    "files_1": "L'Oreal",
    "herm_s": "Hermes",
    "hermes": "Hermes",
    "ib": "Iberdrola",
    "iberdrola": "Iberdrola",
    "l_or_al": "L'Oreal",
    "l_oreal": "L'Oreal",
    "loreal": "L'Oreal",
    "lvmh": "LVMH",
    "nestl": "Nestle",
    "nestle": "Nestle",
    "novartis": "Novartis",
    "prosus": "Prosus",
    "roche": "Roche",
    "sap": "SAP",
    "schneider_electric": "Schneider Electric",
    "schneider_sustainability_impact_q3_2025_results": "Schneider Electric",
    "siemens": "Siemens",
    "totalenergies": "TotalEnergies",
    "urd2024accessibleversion": "Danone",
    "volkswagen": "Volkswagen",
}
RANKING_QUERY_PATTERN = re.compile(
    r"\b(rank|ranking|compare|comparison|benchmark|best|worst|leaders?|laggards?|top\s+\d+|performance)\b",
    re.IGNORECASE,
)
COMPARISON_QUERY_PATTERN = re.compile(r"\b(compare|comparison|versus|vs\.?|against)\b", re.IGNORECASE)
MULTI_COMPANY_LIST_PATTERN = re.compile(
    r"\b(list|show|summari[sz]e|outline|describe|identify|what\s+are|which\s+are)\b",
    re.IGNORECASE,
)
COMPANY_DISPLAY_ALIASES = {
    "l oreal": "L'Oreal",
    "loreal": "L'Oreal",
    "loreal oreal": "L'Oreal",
    "totalenergies": "TotalEnergies",
}
REPORT_TYPE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("climate", "transition", "plan"), "Climate Transition Plan"),
    (("transition", "plan"), "Transition Plan"),
    (("net", "zero"), "Net Zero Plan"),
    (("sustainability",), "Sustainability Report"),
    (("esg",), "ESG Report"),
    (("integrated",), "Integrated Report"),
    (("annual",), "Annual Report"),
    (("universal", "registration", "document"), "Universal Registration Document"),
    (("registration", "document"), "Registration Document"),
    (("tcfd",), "TCFD Report"),
    (("csr",), "CSR Report"),
    (("progress",), "Progress Report"),
    (("climate",), "Climate Report"),
]


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


def _normalize_prompt_style(style: str | None) -> str:
    if not style:
        return "balanced"
    normalized = style.strip().lower().replace("-", "_")
    if normalized in VALID_PROMPT_STYLES:
        return normalized
    supported = ", ".join(VALID_PROMPT_STYLES)
    raise RuntimeError(f"Unknown prompt style '{style}'. Choose from: {supported}.")


def _normalize_answer_mode(mode: str | None) -> str:
    if not mode:
        return "assistant"
    normalized = mode.strip().lower().replace("-", "_")
    if normalized in VALID_ANSWER_MODES:
        return normalized
    supported = ", ".join(VALID_ANSWER_MODES)
    raise RuntimeError(f"Unknown answer mode '{mode}'. Choose from: {supported}.")

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
            try:
                response = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
            except requests.RequestException:
                if attempt == 6:
                    raise
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2.0, 30.0)
                continue
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
        heading_match = re.match(
            r"^((?:[\dIVXivx]+[\.\)]\s+)?(?:[A-Z][A-Z/&-]*(?:\s+[A-Z][A-Z/&-]*){0,11}))\b",
            stripped,
        )
        if heading_match:
            candidate = heading_match.group(1).strip()
            if candidate and len(candidate) < 100 and len(candidate.split()) >= 2:
                return candidate
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


def _normalize_identifier_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _extract_company_tokens(tokens: list[str]) -> list[str]:
    return [
        token
        for token in tokens
        if token not in SOURCE_FILENAME_STOPWORDS
        and not re.fullmatch(r"20\d{2}", token)
    ]


def _format_company_name(tokens: list[str]) -> str:
    if not tokens:
        return ""

    phrase = " ".join(tokens)
    alias = COMPANY_DISPLAY_ALIASES.get(phrase)
    if alias:
        return alias

    words: list[str] = []
    for token in tokens:
        if token.isdigit():
            words.append(token)
        elif len(token) <= 3:
            words.append(token.upper())
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _possessive_suffix(company: str) -> str:
    stripped = company.rstrip()
    if not stripped:
        return ""
    return "'" if stripped[-1].lower() == "s" else "'s"


def _infer_report_type(tokens: list[str]) -> str:
    token_set = set(tokens)
    for required, label in REPORT_TYPE_RULES:
        if all(token in token_set for token in required):
            return label
    if "report" in token_set:
        return "Report"
    return ""


def _format_source_title(source_file: str, source_path: str = "") -> str:
    stem = Path(source_file).stem
    tokens = _normalize_identifier_tokens(stem)
    year = _extract_report_year(source_file)
    report_type = _infer_report_type(tokens)

    company = ""
    if source_path:
        inferred_company = _company_label_from_source(source_file, source_path)
        if inferred_company and not re.fullmatch(r"[0-9a-f]{6,}", inferred_company.lower()):
            company = inferred_company

    company_tokens = _extract_company_tokens(tokens)
    if not company and not company_tokens and source_path:
        path_tokens = _normalize_identifier_tokens(" ".join(Path(source_path).parts[-4:]))
        company_tokens = _extract_company_tokens(path_tokens)

    if not company:
        company = _format_company_name(company_tokens)
    if not company:
        fallback = " ".join(stem.replace("_", " ").replace("-", " ").split()).strip()
        if fallback:
            return " ".join(
                word.upper() if len(word) <= 3 else word.capitalize()
                for word in fallback.split()
            )
        return "Document"

    possessive = _possessive_suffix(company)
    if year and report_type:
        return f"{company}{possessive} {year} {report_type}"
    if year:
        return f"{company}{possessive} {year} Report"
    if report_type:
        return f"{company}{possessive} {report_type}"
    return f"{company}{possessive} Report"


def _format_page_label(page_start: int, page_end: int) -> str:
    if page_start <= 0 or page_end <= 0:
        return ""
    if page_start == page_end:
        return f"p. {page_start}"
    return f"pp. {page_start}-{page_end}"


def _humanize_company_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if slug in SOURCE_COMPANY_ALIASES:
        return SOURCE_COMPANY_ALIASES[slug]
    words = [word for word in re.split(r"[_\-\s]+", value) if word]
    return " ".join(word.upper() if len(word) <= 3 else word.capitalize() for word in words)


def _company_label_from_source(source_file: str, source_path: str = "") -> str:
    path = Path(source_path) if source_path else Path(source_file)
    normalized_identifier = re.sub(r"[^a-z0-9]+", "_", f"{source_path} {source_file}".lower()).strip("_")
    for slug, label in sorted(SOURCE_COMPANY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(^|_){re.escape(slug)}(_|$)", normalized_identifier):
            return label

    parts = list(path.parts)
    marker_pairs = [
        ("pdfs", 1),
        ("raw", 1),
        ("downloaded_reports", 0),
        ("sample_data", 0),
    ]
    for marker, offset in marker_pairs:
        if marker in parts:
            marker_index = parts.index(marker)
            if offset and marker_index + offset < len(parts) - 1:
                return _humanize_company_slug(parts[marker_index + offset])
            break

    stem = Path(source_file).stem
    stem = re.sub(r"^(20\d{2}|undated)_", "", stem)
    stem = re.sub(r"_[0-9a-f]{8}$", "", stem)
    stem = re.sub(
        r"\b(annual|ar|climate|databook|document|esg|framework|integrated|progress|report|"
        r"sustainability|thematic|tracker|urd|universal|registration|statement|results)\b",
        " ",
        stem.replace("_", " ").replace("-", " "),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", stem).strip()
    return _humanize_company_slug(cleaned or Path(source_file).stem)


def _format_citation_label(chunk: dict[str, Any]) -> str:
    title = _format_source_title(chunk["source_file"], chunk.get("source_path", ""))
    page_label = _format_page_label(int(chunk["page_start"]), int(chunk["page_end"]))
    if page_label:
        return f"{title}, {page_label}"
    return title


def is_company_ranking_question(question: str) -> bool:
    lower = question.lower()
    return bool(RANKING_QUERY_PATTERN.search(lower) and any(term in lower for term in ("compan", "esg", "target", "achievement", "improvement")))


def _mentioned_company_labels(question: str) -> list[str]:
    question_lower = question.lower().replace("’", "'")
    normalized_question = re.sub(r"\b([a-z0-9]{2,})'s\b", r"\1", question_lower)
    normalized_question = re.sub(r"[^a-z0-9]+", "_", normalized_question).strip("_")
    alias_map: dict[str, str] = {}
    for slug, label in SOURCE_COMPANY_ALIASES.items():
        alias_map[slug] = label
        if len(slug) <= 5 and not slug.endswith("s"):
            alias_map[f"{slug}s"] = label
        alias_map[re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")] = label

    matches: list[tuple[int, str]] = []
    for alias, label in alias_map.items():
        match = re.search(rf"(^|_){re.escape(alias)}(_|$)", normalized_question)
        if match:
            matches.append((match.start(), label))

    labels: list[str] = []
    for _position, label in sorted(matches, key=lambda item: item[0]):
        if label not in labels:
            labels.append(label)
    return labels


def is_named_company_comparison_question(question: str) -> bool:
    if len(_mentioned_company_labels(question)) < 2:
        return False
    if COMPARISON_QUERY_PATTERN.search(question):
        return True

    lower = question.lower()
    if len(MULTI_COMPANY_LIST_PATTERN.findall(lower)) >= 2:
        return True
    if MULTI_COMPANY_LIST_PATTERN.search(lower) and any(separator in question for separator in (".", ";", "\n")):
        return True
    return False


def _ranking_dimensions(question: str) -> list[tuple[str, str]]:
    lower = question.lower()
    dimensions: list[tuple[str, str]] = []
    if any(term in lower for term in ("ambition", "goal", "goals", "commitment", "commitments", "objective", "objectives")):
        dimensions.append(
            (
                "targets",
                "ESG ambitions goals commitments objectives targets baseline target year scope coverage validation "
                "SBTi net zero reduction pledge roadmap policy biodiversity no deforestation no net loss",
            )
        )
    if "achievement" in lower:
        dimensions.append(("achievements", "ESG achievements realised performance recognitions awards disclosed results"))
    if "target" in lower:
        dimensions.append(("targets", "ESG targets baseline target year scope coverage validation SBTi net zero reduction"))
    if "improvement" in lower or "progress" in lower:
        dimensions.append(("improvements", "ESG improvement progress change over time increased decreased reduced compared year over year"))
    if not dimensions:
        dimensions = [
            (
                "overall",
                "overall ESG performance ESG ratings MSCI CDP DJSI Sustainalytics EcoVadis ISS ESG FTSE4Good awards reductions SBTi validated targets",
            ),
            ("achievements", "ESG achievements realised performance recognitions awards disclosed results"),
            ("targets", "ESG targets baseline target year scope coverage validation"),
            ("improvements", "ESG improvement progress change over time year over year"),
        ]
    return dimensions


def _answer_ranking_dimensions(question: str) -> list[tuple[str, str]]:
    lower = question.lower()
    dimensions: list[tuple[str, str]] = []
    if any(term in lower for term in ("ambition", "goal", "goals", "commitment", "commitments", "objective", "objectives")):
        dimensions.append(("targets", "Goals and Commitments"))
    if "achievement" in lower:
        dimensions.append(("achievements", "Achievements"))
    if "target" in lower:
        dimensions.append(("targets", "Targets"))
    if "improvement" in lower or "progress" in lower:
        dimensions.append(("improvements", "Improvements"))
    if not dimensions:
        dimensions.append(("overall", "Overall ESG Performance"))
    return dimensions


def _build_answer_system_prompt(answer_mode: str, prompt_style: str) -> str:
    mode = _normalize_answer_mode(answer_mode)
    style = _normalize_prompt_style(prompt_style)

    if mode == "benchmark":
        return (
            "You are answering benchmark ESG questions from supplied document context.\n"
            "\n"
            "Use only the supplied context. Do not use outside knowledge. Do not infer missing figures.\n"
            "\n"
            "Output rules:\n"
            "- Return only the answer text.\n"
            "- Do not add headings, bullets, citations, commentary, evidence excerpts, or limitations.\n"
            "- Stay as extractive as possible and preserve the document's wording when the answer is available.\n"
            "- Prefer the shortest complete answer that is still factually correct.\n"
            "- If the context does not contain enough evidence, return exactly:\n"
            "The provided context does not contain sufficient evidence to answer this question."
        )

    shared_prefix = (
        "You are an ESG analyst producing reliable, reproducible answers from a document database. "
        "Use only the supplied context; never use outside knowledge or infer missing figures.\n"
        "\n"
    )

    if style == "audit":
        return (
            shared_prefix
            + "Required output format. Use four blocks in this exact order and keep them concise. "
            "Do not add headings for the first three blocks.\n"
            "\n"
            "Block 1 (key takeaway): Give the direct answer in 1-3 short sentences. "
            "If the answer is not evidenced, write: "
            "'The provided context does not contain sufficient evidence to answer this question.'\n"
            "\n"
            "Block 2 (detailed answer): Explain the answer with enough detail for an ESG analyst to audit it.\n"
            "- Cite supporting sources in brackets for every factual claim using the citation label provided in the "
            "context, e.g. [Engie's 2025 ESG Report, pp. 12-13]. Never mention internal chunk IDs.\n"
            "- Use compact bullets when several metrics or targets must be compared.\n"
            "- Distinguish clearly between disclosed facts and cautious interpretation.\n"
            "\n"
            "Block 3 (Evidence excerpts): Include 2-5 verbatim snippets copied from the context. "
            "Each excerpt must be under 35 words and followed by its citation label in brackets.\n"
            "\n"
            "Block 4 heading must be exactly: **Limitations**\n"
            "- State what is unknown, partial, conflicting, low-scoring, or absent. Make Uncertainty explicit. "
            "Do not guess.\n"
            "\n"
            "Keep the tone professional and concise. Avoid unsupported interpretation. If interpretation is "
            "necessary, label it with 'Based on the disclosed information'."
        )

    if style == "extractive":
        return (
            shared_prefix
            + "Required output format. Use four blocks in this exact order and keep them concise. "
            "Do not add headings for the first three blocks.\n"
            "\n"
            "Block 1 (key takeaway): Answer in 1-2 short sentences using the document's wording as closely as possible. "
            "Prefer exact disclosed numbers, dates, targets, and policy labels. If the answer is not evidenced, write: "
            "'The provided context does not contain sufficient evidence to answer this question.'\n"
            "\n"
            "Block 2 (detailed answer): Expand only with facts that are explicitly present in the context. "
            "Keep paraphrasing minimal and do not add synthesis beyond the source text.\n"
            "- Cite supporting sources in brackets for every factual claim using the citation label provided in the "
            "context, e.g. [Engie's 2025 ESG Report, pp. 12-13]. Never mention internal chunk IDs.\n"
            "- When metrics or targets are present, preserve the exact value, baseline, scope, and timeframe.\n"
            "\n"
            "Block 3 (Evidence excerpts): Include 2-5 verbatim snippets copied from the context. "
            "Each excerpt must be under 35 words and followed by its citation label in brackets.\n"
            "\n"
            "Block 4 heading must be exactly: **Limitations**\n"
            "- State what is unknown, partial, conflicting, low-scoring, or absent. Make Uncertainty explicit. "
            "Do not guess.\n"
            "\n"
            "Keep the tone precise and audit-friendly. Prefer extraction over abstraction."
        )

    return (
        shared_prefix
        + "Required output format. Use four blocks in this exact order and keep them concise. "
        "Do not add headings for the first three blocks.\n"
        "\n"
        "Block 1 (key takeaway): Give the direct answer in 1-3 short sentences. "
        "If the answer is not evidenced, write: "
        "'The provided context does not contain sufficient evidence to answer this question.'\n"
        "\n"
        "Block 2 (detailed answer): Explain the answer with enough detail for an ESG analyst to audit it.\n"
        "- Cite supporting sources in brackets for every factual claim using the citation label provided in the "
        "context, e.g. [Engie's 2025 ESG Report, pp. 12-13]. Never mention internal chunk IDs.\n"
        "- Do not cite raw filenames; use the humanized document title and page range.\n"
        "- When extracting ESG targets or metrics, include metric, value, year, baseline, scope, coverage, "
        "and methodology only when those fields are explicitly present.\n"
        "\n"
        "Block 3 (Evidence excerpts): Include 2-5 verbatim snippets copied from the context. "
        "Each excerpt must be under 35 words and followed by its citation label in brackets.\n"
        "\n"
        "Block 4 heading must be exactly: **Limitations**\n"
        "- State what is unknown, partial, conflicting, low-scoring, or absent. Make Uncertainty explicit. "
        "Do not guess.\n"
        "\n"
        "Keep the tone professional and concise. Avoid unsupported interpretation. If interpretation is "
        "necessary, label it with 'Based on the disclosed information'."
    )


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


def _normalize_embeddings(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)

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
    return SEARCH_TOKEN_PATTERN.findall(normalize_retrieval_text(text).lower())

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


def _normalize_match_scores(
    matches: list[tuple[int, float]],
    *,
    score_type: str = "generic",
) -> dict[int, float]:
    """
    Return scores in [0, 1], but avoid unstable per-query min-max behavior.

    score_type:
    - "cosine": cosine/IP over L2-normalized embeddings, usually [-1, 1]
    - "bm25": positive, unbounded lexical score
    - "generic": fallback robust percentile normalization
    """
    if not matches:
        return {}

    indices = [index for index, _score in matches]
    raw_values = np.asarray([score for _index, score in matches], dtype=np.float32)

    raw_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)

    if score_type == "cosine":
        # Cosine similarity over normalized vectors is theoretically [-1, 1].
        # Convert to [0, 1] without depending on the current candidate set.
        values = (raw_values + 1.0) / 2.0
        values = np.clip(values, 0.0, 1.0)
        return {index: float(value) for index, value in zip(indices, values)}

    if score_type == "bm25":
        # BM25 is positive and unbounded. Compress large values first.
        values = np.log1p(np.maximum(raw_values, 0.0))
    else:
        values = raw_values.copy()

    if values.size == 1:
        return {indices[0]: 0.5}

    if values.size >= 8:
        low = float(np.percentile(values, 5))
        high = float(np.percentile(values, 95))
    else:
        low = float(values.min())
        high = float(values.max())

    if not np.isfinite(low) or not np.isfinite(high) or math.isclose(high, low, rel_tol=1e-6, abs_tol=1e-6):
        # Equal scores should be neutral, not perfect.
        return {index: 0.5 for index in indices}

    normalized = (values - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    return {index: float(value) for index, value in zip(indices, normalized)}
    
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
        if architecture == "lexical":
            lexical_matches = _rank_score_map(
                _score_lexical_matches(question, chunks, candidate_indices=candidate_indices),
                breadth,
            )
            return lexical_matches[:top_k]

    if architecture == "lexical":
        lexical_matches = _rank_score_map(
            _score_lexical_matches(question, chunks, candidate_indices=candidate_indices),
            breadth,
        )
        lexical_norm = _normalize_match_scores(lexical_matches, score_type="bm25")
        return [
            (index, lexical_norm.get(index, 0.0))
            for index, _score in lexical_matches[:top_k]
        ]

    if architecture == "semantic_rerank":
        candidate_indices = [index for index, _ in semantic_matches]

        lexical_matches = _rank_score_map(
            _score_lexical_matches(question, chunks, candidate_indices=candidate_indices),
            breadth,
        )

        lexical_norm = _normalize_match_scores(lexical_matches, score_type="bm25")
        semantic_norm = _normalize_match_scores(semantic_matches, score_type="cosine")

        reranked = [
            (
                index,
                (0.75 * semantic_norm.get(index, 0.0))
                + (0.25 * lexical_norm.get(index, 0.0)),
            )
            for index in candidate_indices
        ]

        return sorted(reranked, key=lambda item: item[1], reverse=True)[:top_k]

    lexical_matches = _rank_score_map(
        _score_lexical_matches(question, chunks, candidate_indices=candidate_indices),
        breadth,
    )

    semantic_norm = _normalize_match_scores(semantic_matches, score_type="cosine")
    lexical_norm = _normalize_match_scores(lexical_matches, score_type="bm25")

    semantic_rank = {index: rank for rank, (index, _score) in enumerate(semantic_matches, start=1)}
    lexical_rank = {index: rank for rank, (index, _score) in enumerate(lexical_matches, start=1)}

    all_indices = set(semantic_rank) | set(lexical_rank)

    hybrid_scores: dict[int, float] = {}

    for index in all_indices:
        rrf = 0.0
        if index in semantic_rank:
            rrf += 1.0 / (60.0 + semantic_rank[index])
        if index in lexical_rank:
            rrf += 1.0 / (60.0 + lexical_rank[index])

        # Normalize RRF approximately into [0, 1].
        # Max possible is about 2 / 61.
        rrf_norm = min(rrf / (2.0 / 61.0), 1.0)

        hybrid_scores[index] = (
            0.60 * semantic_norm.get(index, 0.0)
            + 0.30 * lexical_norm.get(index, 0.0)
            + 0.10 * rrf_norm
        )

    return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

def _load_chunks(index_dir: Path) -> list[ChunkRecord]:
    payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(ChunkRecord)}
    return [ChunkRecord(**{k: v for k, v in item.items() if k in allowed}) for item in payload]


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

def normalize_retrieval_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    # Remove soft hyphen.
    text = text.replace("\u00ad", "")

    # Repair PDF hyphenation: "emis-\nsions" -> "emissions"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Normalize common ESG variants.
    replacements = {
        "CO₂": "CO2 carbon dioxide",
        "co₂": "co2 carbon dioxide",
        "GHG": "greenhouse gas ghg",
        "SBTi": "science based targets initiative sbti",
        "net-zero": "net zero",
        "Net-Zero": "net zero",
        "Scope 1": "scope 1 scope1",
        "Scope 2": "scope 2 scope2",
        "Scope 3": "scope 3 scope3",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Fix spaced-out OCR/PDF letters like "E S G".
    text = re.sub(r"\bE\s+S\s+G\b", "ESG", text, flags=re.IGNORECASE)
    text = re.sub(r"\bG\s+H\s+G\b", "GHG", text, flags=re.IGNORECASE)

    # Remove repeated whitespace.
    text = normalize_whitespace(text)

    return text

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
            body = normalize_retrieval_text(chunk.text)
            summary = normalize_retrieval_text(chunk.contextual_summary)

            if summary:
                texts_to_embed.append(f"[Context: {summary}]\n\n{body}")
            else:
                texts_to_embed.append(body)
    else:
        texts_to_embed = [normalize_retrieval_text(chunk.text) for chunk in chunks]


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




def _extract_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    stat = pdf_path.stat()
    with fitz.open(pdf_path) as document:
        raw_metadata = document.metadata or {}
        metadata = {
            key: value
            for key, value in raw_metadata.items()
            if isinstance(value, str) and value.strip()
        }
        extracted_title = metadata.get("title", "").strip()
        page_count = document.page_count

    return {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "display_title": _format_source_title(pdf_path.name, str(pdf_path)),
        "extracted_title": extracted_title or None,
        "page_count": page_count,
        "report_year": _extract_report_year(pdf_path.name) or None,
        "file_size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "pdf_metadata": metadata,
    }


def export_chunk_dataset(
    *,
    pdf_paths: list[Path],
    output_dir: Path,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int = 0,
    section_aware: bool = False,
    contextual_chunking: bool = False,
) -> dict[str, Any]:
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_pdfs: list[dict[str, str]] = []
    chunk_records = build_chunk_records(
        pdf_paths,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        section_aware=section_aware,
        contextual_chunking=contextual_chunking,
        skipped_pdfs=skipped_pdfs,
    )

    chunk_records_by_path: dict[str, list[ChunkRecord]] = {}
    for chunk in chunk_records:
        chunk_records_by_path.setdefault(chunk.source_path, []).append(chunk)

    documents: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        source_path = str(pdf_path)
        doc_chunks = chunk_records_by_path.get(source_path, [])
        metadata = _extract_pdf_metadata(pdf_path)
        document_entry = {
            **metadata,
            "chunk_count": len(doc_chunks),
            "chunking": {
                "target_tokens": target_tokens,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
                "section_aware": section_aware,
                "contextual_chunking": contextual_chunking,
            },
            "chunks": [],
        }

        for ordinal, chunk in enumerate(doc_chunks, start=1):
            chunk_entry = {
                "chunk_id": chunk.chunk_id,
                "chunk_index": ordinal,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "section_title": chunk.section_title or None,
                "contextual_summary": chunk.contextual_summary or None,
                "esg_pillar": chunk.esg_pillar or None,
                "report_year": chunk.report_year or None,
                "contains_table": chunk.contains_table,
                "contains_targets": chunk.contains_targets,
                "text": chunk.text,
            }
            document_entry["chunks"].append(chunk_entry)
            jsonl_rows.append(
                {
                    **metadata,
                    "chunk_count": len(doc_chunks),
                    **chunk_entry,
                }
            )

        documents.append(document_entry)

    payload = {
        "built_at": datetime.now(UTC).isoformat(),
        "pdf_count": len(pdf_paths),
        "chunk_count": len(chunk_records),
        "skipped_pdfs": skipped_pdfs,
        "chunking": {
            "target_tokens": target_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "section_aware": section_aware,
            "contextual_chunking": contextual_chunking,
        },
        "documents": documents,
    }

    documents_path = output_dir / "documents.json"
    jsonl_path = output_dir / "chunks.jsonl"
    manifest_path = output_dir / "manifest.json"

    documents_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    jsonl_lines = [json.dumps(row, ensure_ascii=False) for row in jsonl_rows]
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "built_at": payload["built_at"],
                "pdf_count": payload["pdf_count"],
                "chunk_count": payload["chunk_count"],
                "documents_path": str(documents_path),
                "jsonl_path": str(jsonl_path),
                "skipped_pdfs": skipped_pdfs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "documents_path": documents_path,
        "jsonl_path": jsonl_path,
        "manifest_path": manifest_path,
        "payload": payload,
    }


def _filter_chunks(
    chunks: list[dict[str, Any]],
    *,
    filter_year: str | None = None,
    filter_pillar: str | None = None,
    filter_report_type: str | None = None,
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
    return filtered


def _chunk_matches_filters(
    chunk: ChunkRecord,
    *,
    allowed_source_paths: set[str] | None = None,
    filter_year: str | None = None,
    filter_pillar: str | None = None,
    filter_report_type: str | None = None,
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
        return filter_report_type.lower() in chunk.source_file.lower()

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
    filters_active = bool(allowed_source_paths or filter_year or filter_pillar or filter_report_type)

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
                "normalized_score": score,
                "retrieval_architecture": normalized_mode,
                "chunk_id": chunk.chunk_id,
                "company_label": _company_label_from_source(chunk.source_file, chunk.source_path),
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
            }
        )

    return results, selected_embedding_model


def retrieve_company_ranking_chunks(
    *,
    index_dir: Path,
    question: str,
    per_company_k: int = 3,
    max_companies: int = 20,
    embedding_model: str | None = None,
    retrieval_architecture: str = "semantic_rerank",
    search_breadth: int = 100,
    target_companies: list[str] | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    vector_backend = manifest.get("vector_backend", "none")
    if vector_backend == "none":
        raise RuntimeError("This index was built with --dry-run and has no embeddings to search.")

    chunks = _load_chunks(index_dir)
    target_company_set = set(target_companies or [])
    company_indices: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        company = _company_label_from_source(chunk.source_file, chunk.source_path)
        if company and (not target_company_set or company in target_company_set):
            company_indices.setdefault(company, []).append(index)

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_embedding_model = embedding_model or manifest.get("embedding_model")
    if not selected_embedding_model:
        selected_embedding_model = client.get_embedding_model(preferred="bge-m3")

    dimensions = _ranking_dimensions(question)
    query_texts = [f"{question}\n{dimension_query}" for _, dimension_query in dimensions]
    query_vectors = embed_texts(
        client,
        query_texts,
        embedding_model=selected_embedding_model,
        batch_size=1,
    )
    vectors = np.load(index_dir / "vectors.npy")

    selected_by_chunk_id: dict[str, dict[str, Any]] = {}
    company_summaries: list[dict[str, Any]] = []
    for company, indices in sorted(company_indices.items()):
        company_matches: list[tuple[str, int, float]] = []
        for (dimension_name, _dimension_query), query_vector in zip(dimensions, query_vectors):
            breadth = min(max(search_breadth, per_company_k), len(indices))
            semantic_matches = _search_vectors_for_indices(query_vector, vectors, indices, breadth)
            matches = select_chunk_matches(
                question=f"{question}\n{dimension_name}",
                chunks=chunks,
                semantic_matches=semantic_matches,
                top_k=per_company_k,
                retrieval_architecture=retrieval_architecture,
                search_breadth=breadth,
                candidate_indices=indices,
            )
            company_matches.extend((dimension_name, chunk_index, score) for chunk_index, score in matches)

        if not company_matches:
            continue

        top_score = max(score for _, _, score in company_matches)
        company_summaries.append(
            {
                "company": company,
                "candidate_chunks": len(indices),
                "selected_chunks": len({chunk_index for _, chunk_index, _ in company_matches}),
                "top_score": top_score,
            }
        )
        for dimension_name, chunk_index, score in company_matches:
            chunk = chunks[chunk_index]
            existing = selected_by_chunk_id.get(chunk.chunk_id)
            if existing:
                existing["score"] = max(float(existing["score"]), float(score))
                dimensions_set = set(existing.get("evidence_dimensions", []))
                dimensions_set.add(dimension_name)
                existing["evidence_dimensions"] = sorted(dimensions_set)
                continue

            selected_by_chunk_id[chunk.chunk_id] = {
                "score": score,
                "chunk_id": chunk.chunk_id,
                "company_label": company,
                "evidence_dimensions": [dimension_name],
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
            }

    company_summaries.sort(key=lambda item: item["top_score"], reverse=True)
    if target_company_set:
        allowed_companies = {item["company"] for item in company_summaries if item["company"] in target_company_set}
    else:
        allowed_companies = {item["company"] for item in company_summaries[:max_companies]}
    filtered_chunks = [chunk for chunk in selected_by_chunk_id.values() if chunk["company_label"] in allowed_companies]
    filtered_chunks.sort(key=lambda item: (item["company_label"], -float(item["score"])))

    diagnostics = {
        "mode": "company_ranking",
        "companies_considered": len(company_summaries),
        "companies_in_context": len(allowed_companies),
        "per_company_k": per_company_k,
        "retrieval_architecture": retrieval_architecture,
        "dimensions": [dimension_name for dimension_name, _ in dimensions],
        "company_summaries": [item for item in company_summaries if item["company"] in allowed_companies],
    }
    return filtered_chunks, selected_embedding_model, diagnostics


RANKING_EVIDENCE_PATTERNS = {
    "achievements": (
        r"\b(msci|cdp|djsi|ftse4good|ecovadis|iss esg|prime|award|recognition|leader|leadership|"
        r"rating|ranked|included|inclusion|top|gold|a[- ]?list)\b",
        r"\b(reduced|reduction|achieved|saved|renewable|certified|validated)\b",
    ),
    "targets": (
        r"\b(target|targets|2030|2040|2045|2050|net[- ]?zero|sbti|science[- ]based|baseline|"
        r"scope 1|scope 2|scope 3|validated|reduction pathway)\b",
        r"\b(reduce|reduction|absolute|intensity|supplier engagement|near[- ]term|long[- ]term)\b",
    ),
    "improvements": (
        r"\b(improved|improvement|progress|reduced|reduction|decreased|increased|upgraded|"
        r"down|fell|saved|compared|year[- ]over[- ]year|since|baseline)\b",
        r"\b(2020|2021|2022|2023|2024|2025|2019|2018)\b",
    ),
    "overall": (
        r"\b(msci|cdp|djsi|ftse4good|sustainalytics|ecovadis|iss esg|prime|rating|ranked|"
        r"award|leader|leadership|sbti|science[- ]based|validated|net[- ]?zero)\b",
        r"\b(reduced|reduction|decreased|saved|improved|upgraded|renewable|emissions)\b",
    ),
}
RANKING_QUANT_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?\s?%|\b20\d{2}\b|\bscope\s?[123]\b)", re.IGNORECASE)
LOW_QUALITY_RANKING_PATTERNS = (
    r"\btable of contents\b",
    r"\bcontents\b.{0,80}\bappendix\b",
    r"\bdisclose the metrics and targets\b",
    r"\bdisclose how the organization\b",
    r"\besrs e[1-5][- ]\d+\b",
    r"\besrs 2\b.{0,80}\bgeneral disclosures\b",
    r"\bappendix:?\s*a\b",
    r"\bpage\s+\d+\b.{0,40}\bpage\s+\d+\b",
    r"\be\s+s\s+g\b|\bg\s+o\s+v\s+e\s+r\s+n\s+a\s+n\s+c\s+e\b|\bi\s+n\s+d\s+i\s+c\s+a\s+t\s+o\s+r\s+s\b",
    r"(?:\b[A-Z]\b\s+){5,}",
)


def _ranking_text_quality(text: str) -> float:
    normalized = normalize_whitespace(text).lower()
    penalty = 1.0
    for pattern in LOW_QUALITY_RANKING_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            penalty *= 0.35
    if len(re.findall(r"\besrs\b|\btaxonomy\b|\bappendix\b|\bcontents\b", normalized)) >= 6:
        penalty *= 0.45
    if len(re.findall(r"\bp\.|pp\.|page\b", normalized)) >= 8:
        penalty *= 0.5
    return max(0.1, penalty)


def _ranking_keyword_score(text: str, dimension: str) -> float:
    normalized = text.lower()
    patterns = RANKING_EVIDENCE_PATTERNS.get(dimension, RANKING_EVIDENCE_PATTERNS["overall"])
    matches = 0
    for pattern in patterns:
        matches += len(re.findall(pattern, normalized, flags=re.IGNORECASE))
    quant_matches = len(RANKING_QUANT_PATTERN.findall(normalized))
    return min(matches, 10) * 0.35 + min(quant_matches, 8) * 0.25


def _chunk_dimension_score(chunk: dict[str, Any], dimension: str) -> float:
    dimensions = set(chunk.get("evidence_dimensions", []))
    dimension_bonus = 1.0 if dimension == "overall" or dimension in dimensions else 0.35
    retrieval_score = float(chunk.get("score", 0.0)) * 3.0
    text = str(chunk.get("text", ""))
    keyword_score = _ranking_keyword_score(text, dimension)
    target_bonus = 0.4 if dimension in ("targets", "overall") and chunk.get("contains_targets") else 0.0
    table_bonus = 0.2 if chunk.get("contains_table") else 0.0
    return (retrieval_score + keyword_score + target_bonus + table_bonus) * dimension_bonus * _ranking_text_quality(text)


def _clean_excerpt_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"\bLVM\s+H\b", "LVMH", text)
    text = re.sub(r"(?:\b[A-Za-z]\b\s+){5,}", " ", text)
    text = re.sub(r"(?:\b\d\b\s+){3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _excerpt_for_dimension(text: str, dimension: str, max_words: int = 22) -> str:
    normalized = _clean_excerpt_text(text)
    patterns = RANKING_EVIDENCE_PATTERNS.get(dimension, RANKING_EVIDENCE_PATTERNS["overall"])
    start_word = 0
    matched_keyword = False
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            start_word = max(0, len(normalized[: match.start()].split()) - 4)
            matched_keyword = True
            break
    words = normalized.split()
    if not matched_keyword and len(words) > 16 and re.match(r"^\d+$", words[0]):
        start_word = 12
    words = words[start_word : start_word + max_words + 1]
    excerpt = " ".join(words[:max_words])
    if len(words) > max_words:
        excerpt += "..."
    return excerpt


def _confidence_label(total_score: float, evidence_count: int, best_chunk: dict[str, Any]) -> str:
    text = str(best_chunk.get("text", ""))
    has_quant = bool(RANKING_QUANT_PATTERN.search(text))
    quality = _ranking_text_quality(text)
    if total_score >= 7.0 and evidence_count >= 2 and has_quant and quality >= 0.8:
        return "High"
    if total_score >= 4.0 and (evidence_count >= 2 or has_quant) and quality >= 0.35:
        return "Medium"
    return "Low"


def _citation_labels(chunks: list[dict[str, Any]], max_labels: int = 2) -> str:
    labels: list[str] = []
    for chunk in chunks:
        label = _format_citation_label(chunk)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= max_labels:
            break
    return "; ".join(labels)


def _ranking_rationale(company_chunks: list[dict[str, Any]], dimension: str) -> str:
    texts = " ".join(str(chunk.get("text", "")) for chunk in company_chunks[:3]).lower()
    points: list[str] = []
    if re.search(r"\bsbti|science[- ]based|validated\b", texts, flags=re.IGNORECASE):
        points.append("validated or science-based targets")
    if re.search(r"\bnet[- ]?zero|2030|2040|2045|2050\b", texts, flags=re.IGNORECASE):
        points.append("time-bound climate commitments")
    if re.search(r"\bmsci|cdp|djsi|ftse4good|ecovadis|iss esg|prime|award|leader\b", texts, flags=re.IGNORECASE):
        points.append("external ESG recognitions or ratings")
    if re.search(r"\breduced|reduction|decreased|saved|improved|upgraded|progress\b", texts, flags=re.IGNORECASE):
        points.append("documented progress or reductions")
    if re.search(r"\bscope 1|scope 2|scope 3|emissions|renewable\b", texts, flags=re.IGNORECASE):
        points.append("quantified climate or emissions evidence")
    if not points:
        points.append("relevant ESG disclosure in the retrieved source")

    if dimension == "achievements":
        prefix = "Ranks here because the evidence shows "
    elif dimension == "targets":
        prefix = "Ranks here because the evidence shows "
    elif dimension == "improvements":
        prefix = "Ranks here because the evidence shows "
    else:
        prefix = "Ranks here based on "
    return prefix + ", ".join(points[:3]) + "."


def generate_company_ranking_answer(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    max_rows: int = 10,
) -> str:
    """Build a deterministic, extractive ranking answer from retrieved company evidence."""
    company_chunks: dict[str, list[dict[str, Any]]] = {}
    for chunk in retrieved_chunks:
        company = str(chunk.get("company_label") or "").strip()
        if company:
            company_chunks.setdefault(company, []).append(chunk)

    if not company_chunks:
        return "The retrieved context does not contain company-labelled evidence for a ranking."

    sections: list[str] = []
    for dimension, title in _answer_ranking_dimensions(question):
        rows: list[dict[str, Any]] = []
        for company, chunks in company_chunks.items():
            scored_chunks = sorted(
                ((chunk, _chunk_dimension_score(chunk, dimension)) for chunk in chunks),
                key=lambda item: item[1],
                reverse=True,
            )
            supporting = [(chunk, score) for chunk, score in scored_chunks if score >= 1.5]
            if not supporting:
                continue
            support_scores = [score for _chunk, score in supporting[:3]]
            total_score = support_scores[0]
            if len(support_scores) > 1:
                total_score += support_scores[1] * 0.25
            if len(support_scores) > 2:
                total_score += support_scores[2] * 0.1
            if total_score < 3.5:
                continue
            best_chunk = supporting[0][0]
            rows.append(
                {
                    "company": company,
                    "score": total_score,
                    "chunks": [chunk for chunk, _score in supporting[:3]],
                    "best_chunk": best_chunk,
                    "confidence": _confidence_label(total_score, len(supporting), best_chunk),
                }
            )

        rows.sort(key=lambda row: row["score"], reverse=True)
        rows = rows[:max_rows]
        if not rows:
            continue

        sections.append(f"**{title}**")
        sections.append(
            "| Rank | Company | Evidence-based rationale | Evidence excerpt | Sources | Confidence |\n"
            "|---|---|---|---|---|---|"
        )
        for rank, row in enumerate(rows, start=1):
            citations = _citation_labels(row["chunks"], max_labels=2)
            excerpt = _excerpt_for_dimension(str(row["best_chunk"].get("text", "")), dimension)
            rationale = _ranking_rationale(row["chunks"], dimension)
            sections.append(
                f"| {rank} | {row['company']} | {rationale} | \"{excerpt}\" | {citations} | {row['confidence']} |"
            )
        sections.append("")

    sections.append("**Limitations**")
    sections.append(
        "- This is a provisional, evidence-weighted ranking from the indexed documents only; it is not an external ESG rating."
    )
    sections.append(
        "- Companies with sparse, qualitative, or non-comparable excerpts may rank lower or be omitted from a dimension."
    )
    sections.append("- Confidence reflects disclosure quality in the retrieved excerpts, not independent verification.")
    return "\n".join(sections).strip()


def generate_company_comparison_answer(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    companies: list[str],
) -> str:
    """Build a side-by-side company comparison from retrieved company evidence."""
    dimensions = _answer_ranking_dimensions(question)
    if dimensions == [("overall", "Overall ESG Performance")] and "ambition" in question.lower():
        dimensions = [("targets", "Ambitions and Targets")]

    company_chunks: dict[str, list[dict[str, Any]]] = {company: [] for company in companies}
    for chunk in retrieved_chunks:
        company = str(chunk.get("company_label") or "").strip()
        if company in company_chunks:
            company_chunks[company].append(chunk)

    sections = ["**Company Comparison**"]
    if len(companies) >= 2:
        sections.append(
            f"Based only on the retrieved documents, this comparison covers {', '.join(companies[:-1])} and {companies[-1]}."
        )

    for dimension, title in dimensions:
        rows: list[dict[str, Any]] = []
        for company in companies:
            chunks = company_chunks.get(company, [])
            scored = sorted(
                ((chunk, _chunk_dimension_score(chunk, dimension)) for chunk in chunks),
                key=lambda item: item[1],
                reverse=True,
            )
            supporting = [(chunk, score) for chunk, score in scored if score >= 0.35]
            if not supporting:
                rows.append(
                    {
                        "company": company,
                        "score": 0.0,
                        "chunks": [],
                        "best_chunk": None,
                        "confidence": "Low",
                    }
                )
                continue
            support_scores = [score for _chunk, score in supporting[:3]]
            total_score = support_scores[0]
            if len(support_scores) > 1:
                total_score += support_scores[1] * 0.25
            if len(support_scores) > 2:
                total_score += support_scores[2] * 0.1
            best_chunk = supporting[0][0]
            rows.append(
                {
                    "company": company,
                    "score": total_score,
                    "chunks": [chunk for chunk, _score in supporting[:3]],
                    "best_chunk": best_chunk,
                    "confidence": _confidence_label(total_score, len(supporting), best_chunk),
                }
            )

        sections.append(f"\n**{title}**")
        sections.append("| Company | Evidence-based assessment | Evidence excerpt | Sources | Confidence |")
        sections.append("|---|---|---|---|---|")
        for row in rows:
            if not row["chunks"]:
                sections.append(
                    f"| {row['company']} | No sufficiently direct evidence was retrieved for this comparison dimension. | N/A | N/A | Low |"
                )
                continue
            rationale = _ranking_rationale(row["chunks"], dimension)
            excerpt = _excerpt_for_dimension(str(row["best_chunk"].get("text", "")), dimension)
            sources = _citation_labels(row["chunks"], max_labels=2)
            sections.append(
                f"| {row['company']} | {rationale} | \"{excerpt}\" | {sources} | {row['confidence']} |"
            )

        scored_rows = [row for row in rows if row["score"] > 0]
        if len(scored_rows) >= 2:
            scored_rows.sort(key=lambda row: row["score"], reverse=True)
            leader = scored_rows[0]
            runner_up = scored_rows[1]
            if leader["score"] >= runner_up["score"] * 1.15:
                sections.append(
                    f"\nBased on the retrieved evidence, {leader['company']} is better supported on {title.lower()} than {runner_up['company']}."
                )
            else:
                sections.append(
                    f"\nBased on the retrieved evidence, {leader['company']} and {runner_up['company']} are close on {title.lower()}; confidence depends on the completeness of their disclosures."
                )

    sections.append("\n**Limitations**")
    sections.append("- This comparison uses only retrieved documents and does not apply an external ESG scoring methodology.")
    sections.append("- A company may appear weaker where the retrieved excerpts are less specific, not necessarily because its real-world ESG ambition is weaker.")
    sections.append("- Sources are shown as document title and page range; raw internal chunk IDs are intentionally omitted.")
    return "\n".join(sections).strip()


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
    answer_mode: str = "assistant",
    prompt_style: str = "balanced",
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[str, str]:
    if not retrieved_chunks:
        raise RuntimeError("No retrieved chunks were provided to the answer stage.")

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_text_model = text_model or client.get_text_generation_model()

    seen_texts: set[str] = set()
    context_blocks: list[str] = []
    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        normalized = normalize_whitespace(chunk["text"])
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        citation_label = _format_citation_label(chunk)
        context_blocks.append(
            f"[source {source_number}] company={chunk.get('company_label', 'Unknown')} citation={citation_label} "
            f"source={chunk['source_file']} pages={chunk['page_start']}-{chunk['page_end']} "
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

    system_prompt = _build_answer_system_prompt(answer_mode, prompt_style)
    if any(chunk.get("company_label") for chunk in retrieved_chunks) and is_company_ranking_question(question):
        system_prompt += (
            "\n\nComparison/ranking mode:\n"
            "- Treat each company as a separate evidence group using the company= labels in context.\n"
            "- Start directly with the first requested ranking table. Do not include a key takeaway, executive summary, or one-line summary before the tables.\n"
            "- If the user asks for multiple rankings, produce a separate ranked markdown table for each requested dimension.\n"
            "- Each table must use columns: Rank, Company, Evidence-based rationale, Key citations, Confidence.\n"
            "- Use the same rank order in each table and its surrounding explanation; never contradict a ranking within the same answer.\n"
            "- Every ranked row must include at least one citation, and each citation must come from a context chunk whose company= label matches that row's Company.\n"
            "- Never use evidence or citations from one company to justify another company.\n"
            "- Do not group uncited companies into an Others row; omit companies without direct evidence for that dimension.\n"
            "- For achievements, rank by disclosed realised performance and recognitions.\n"
            "- For targets, rank by specificity, ambition, timeframe, baseline, scope coverage, and validation only when disclosed.\n"
            "- For improvements, rank by documented change over time, not by static ambition.\n"
            "- Do not rank entities that are subsidiaries or business units as separate companies unless the context database only contains them.\n"
            "- Add a confidence column and mark thin or non-comparable evidence as Low confidence.\n"
            "- Do not refuse solely because absolute ESG scores are unavailable; provide a provisional evidence-based ranking from the retrieved excerpts and flag uncertainty.\n"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {question}\n\n{weak_retrieval_note}Context:\n{context}",
        },
    ]
    return client.chat_completion(selected_text_model, messages, temperature=temperature), selected_text_model


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


def run_rag_export_chunks(args: Any) -> int:
    pdf_paths = gather_pdf_paths(args.pdf)
    if not pdf_paths:
        raise RuntimeError("No PDF files were found. Pass --pdf or add a PDF under sample_data/.")

    result = export_chunk_dataset(
        pdf_paths=pdf_paths,
        output_dir=args.output_dir,
        target_tokens=args.chunk_target_tokens,
        min_tokens=args.chunk_min_tokens,
        max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.chunk_overlap_tokens,
        section_aware=getattr(args, "section_aware", False),
        contextual_chunking=getattr(args, "contextual_chunking", False),
    )
    payload = result["payload"]
    print(
        f"Exported {payload['chunk_count']} chunks from {payload['pdf_count']} PDF(s) "
        f"to {result['documents_path']} and {result['jsonl_path']}."
    )
    if payload["skipped_pdfs"]:
        print(f"Skipped {len(payload['skipped_pdfs'])} PDF(s); see {result['manifest_path']}.")
    return 0


def run_rag_ask(args: Any) -> int:
    retrieval_mode = getattr(args, "retrieval_mode", None) or getattr(args, "retrieval_architecture", "dense")
    query_transform = getattr(args, "query_transform", None)
    reranker = getattr(args, "reranker", None)
    agentic = getattr(args, "agentic", False)
    filter_year = getattr(args, "filter_year", None)
    filter_pillar = getattr(args, "filter_pillar", None)
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

    if is_named_company_comparison_question(question):
        companies = _mentioned_company_labels(question)
        retrieved, embedding_model, diagnostics = retrieve_company_ranking_chunks(
            index_dir=args.index_dir,
            question=question,
            per_company_k=3,
            max_companies=len(companies),
            target_companies=companies,
            embedding_model=args.embedding_model,
            retrieval_architecture="semantic_rerank",
            search_breadth=max(candidate_k, 30),
            base_url=args.base_url,
        )
        print(
            f"Retrieved {len(retrieved)} chunks for company comparison "
            f"across {diagnostics['companies_in_context']} companies using {embedding_model}."
        )
    elif is_company_ranking_question(question):
        retrieved, embedding_model, diagnostics = retrieve_company_ranking_chunks(
            index_dir=args.index_dir,
            question=question,
            per_company_k=1,
            max_companies=max(args.top_k, 20),
            embedding_model=args.embedding_model,
            retrieval_architecture="semantic_rerank",
            search_breadth=max(candidate_k, 30),
            base_url=args.base_url,
        )
        print(
            f"Retrieved {len(retrieved)} chunks for company ranking "
            f"across {diagnostics['companies_in_context']} companies using {embedding_model}."
        )
    elif query_transform or reranker:
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
        )
        print(
            f"Retrieved {len(retrieved)} chunks using {embedding_model} "
            f"({retrieval_mode}, breadth={candidate_k or args.top_k}):"
        )

    for chunk in retrieved:
        preview = chunk["text"][:220].replace("\n", " ")
        meta = ""
        if chunk.get("company_label"):
            meta += f" [{chunk['company_label']}]"
        if chunk.get("evidence_dimensions"):
            meta += f" [{','.join(chunk['evidence_dimensions'])}]"
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

    if is_named_company_comparison_question(question):
        answer = generate_company_comparison_answer(
            question=question,
            retrieved_chunks=retrieved,
            companies=_mentioned_company_labels(question),
        )
        text_model = "deterministic-esg-comparator"
    elif is_company_ranking_question(question):
        answer = generate_company_ranking_answer(question=question, retrieved_chunks=retrieved)
        text_model = "deterministic-esg-ranker"
    else:
        answer, text_model = answer_question(
            question=question,
            retrieved_chunks=retrieved,
            text_model=args.text_model,
            temperature=args.temperature,
            answer_mode=getattr(args, "answer_mode", "assistant"),
            prompt_style=getattr(args, "prompt_style", "balanced"),
            base_url=args.base_url,
        )
    print(f"\nAnswer ({text_model}):\n{answer}")
    return 0
