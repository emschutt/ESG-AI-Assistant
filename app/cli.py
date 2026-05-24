"""Command line interface for the ESG acquisition MVP."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from app.models import CompanySeed
from app.ingest_target_reports import run_ingest_target_reports
from app.pipeline import AcquisitionPipeline
from app.rag import DEFAULT_BASE_URL, DEFAULT_INDEX_DIR, run_rag_ask, run_rag_build
from app.rag_eval import DEFAULT_EVAL_DATASET, DEFAULT_EVAL_OUTPUT, run_rag_eval, run_rag_eval_grid
from app.rag_tune import DEFAULT_TUNE_OUTPUT, run_rag_tune
from app.ragas_eval import DEFAULT_RAGAS_OUTPUT, run_ragas_eval
from app.smoke_test import run_smoke_test
from app.utils import OUTPUT_DIR, ensure_directories
from app.web import run_web_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESG document acquisition MVP")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover candidate reports for a company")
    discover.add_argument("--company", required=True, help="Company name")
    discover.add_argument("--ticker", help="Ticker")
    discover.add_argument("--country", help="Country")
    discover.add_argument("--issuer-domain", help="Known issuer domain")

    fetch = subparsers.add_parser("fetch", help="Discover and fetch the best report for a company")
    fetch.add_argument("--company", required=True, help="Company name")
    fetch.add_argument("--ticker", help="Ticker")
    fetch.add_argument("--country", help="Country")
    fetch.add_argument("--issuer-domain", help="Known issuer domain")

    smoke = subparsers.add_parser("smoke-test", help="Run the live smoke test")
    smoke.add_argument("--top-n", type=int, default=10, help="Number of European companies to test")

    rag_build = subparsers.add_parser("rag-build", help="Build a local ESG RAG index from PDF files")
    rag_build.add_argument(
        "--pdf",
        action="append",
        help=(
            "Path to a PDF to index. Repeat the flag to include multiple files. "
            "If omitted, the local database PDF directories are indexed recursively."
        ),
    )
    rag_build.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory where chunk metadata and vectors should be stored.",
    )
    rag_build.add_argument(
        "--chunk-target-tokens",
        type=int,
        default=380,
        help="Preferred chunk size in approximate tokens.",
    )
    rag_build.add_argument(
        "--chunk-min-tokens",
        type=int,
        default=274,
        help="Minimum chunk size in approximate tokens.",
    )
    rag_build.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=464,
        help="Maximum chunk size in approximate tokens.",
    )
    rag_build.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many chunks to send per embeddings request.",
    )
    rag_build.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=30,
        help="How many approximate tokens to overlap between consecutive chunks.",
    )
    rag_build.add_argument(
        "--section-aware",
        action="store_true",
        help="Enable section/header-aware chunking to align chunks with document structure.",
    )
    rag_build.add_argument(
        "--contextual-chunking",
        action="store_true",
        help="Generate contextual summaries for each chunk and embed contextualized versions.",
    )
    rag_build.add_argument(
        "--embedding-model",
        help="Albert embedding model id. Defaults to a detected BGE-M3-compatible model.",
    )
    rag_build.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )
    rag_build.add_argument(
        "--dry-run",
        action="store_true",
        help="Only extract and chunk the PDFs without calling the Albert API.",
    )

    rag_ask = subparsers.add_parser("rag-ask", help="Ask grounded questions against a local ESG RAG index")
    rag_ask.add_argument(
        "question",
        nargs="+",
        help="Question to ask against the indexed ESG report chunks.",
    )
    rag_ask.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the built RAG index.",
    )
    rag_ask.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many chunks to retrieve before answering.",
    )
    rag_ask.add_argument(
        "--embedding-model",
        help="Albert embedding model id for the question embedding.",
    )
    rag_ask.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["dense", "lexical", "hybrid"],
        help="Retrieval mode: dense (semantic embeddings), lexical (BM25-style), or hybrid (RRF fusion).",
    )
    rag_ask.add_argument(
        "--retrieval-architecture",
        default=argparse.SUPPRESS,
        choices=["semantic", "hybrid", "semantic_rerank", "dense", "lexical"],
        help=argparse.SUPPRESS,
    )
    rag_ask.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help="How many candidate chunks to consider before the final top-k selection.",
    )
    rag_ask.add_argument(
        "--search-breadth",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    rag_ask.add_argument(
        "--text-model",
        help="Albert text-generation model id for answer generation.",
    )
    rag_ask.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the answer-generation model.",
    )
    rag_ask.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )
    rag_ask.add_argument(
        "--search-only",
        action="store_true",
        help="Only retrieve chunks and skip the final LLM answer step.",
    )
    rag_ask.add_argument(
        "--query-transform",
        choices=["none", "hyde", "mqr", "hyde_mqr"],
        default=None,
        help="Apply query transformation before retrieval (HyDE, MQR, or combined).",
    )
    rag_ask.add_argument(
        "--reranker",
        choices=["none", "lexical", "embedding", "llm"],
        default=None,
        help="Apply reranking stage after broad retrieval.",
    )
    rag_ask.add_argument(
        "--agentic",
        action="store_true",
        help="Enable agentic retrieval loop with evidence evaluation and query repair.",
    )
    rag_ask.add_argument(
        "--filter-year",
        help="Filter retrieval to chunks from a specific report year.",
    )
    rag_ask.add_argument(
        "--filter-pillar",
        choices=["environmental", "social", "governance", "all"],
        help="Filter retrieval to specific ESG pillar.",
    )
    rag_ask.add_argument(
        "--filter-company",
        help="Filter retrieval to a specific company (exact, case-insensitive).",
    )
    rag_ask.add_argument(
        "--filter-speaker-role",
        choices=["executive", "analyst", "management"],
        help="Filter retrieval to earnings-call chunks with this speaker role.",
    )
    rag_ask.add_argument(
        "--filter-chunk-kind",
        choices=["narrative", "table_fact"],
        help="Filter retrieval to narrative chunks or table-extracted facts.",
    )
    rag_ask.add_argument(
        "--filter-doc-type",
        help="Filter retrieval to a specific doc_type (e.g. urd, progress_report, earnings_call).",
    )

    rag_web = subparsers.add_parser("rag-web", help="Launch the local browser UI for ESG RAG")
    rag_web.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the local web app to.",
    )
    rag_web.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port for the local web app.",
    )
    rag_web.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the RAG index used by the web app.",
    )

    rag_eval = subparsers.add_parser("rag-eval", help="Evaluate retrieval quality against the ESG QA dataset")
    rag_eval.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the built RAG index.",
    )
    rag_eval.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="CSV dataset containing ESG evaluation questions and expected contexts.",
    )
    rag_eval.add_argument(
        "--company",
        help="Company to evaluate. If omitted, the command tries to infer it from the indexed PDF paths.",
    )
    rag_eval.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many chunks to retrieve for each evaluation question.",
    )
    rag_eval.add_argument(
        "--embedding-model",
        help="Albert embedding model id for the question embedding.",
    )
    rag_eval.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["dense", "lexical", "hybrid"],
        help="Retrieval mode: dense (semantic embeddings), lexical (BM25-style), or hybrid (RRF fusion).",
    )
    rag_eval.add_argument(
        "--retrieval-architecture",
        default=argparse.SUPPRESS,
        choices=["semantic", "hybrid", "semantic_rerank", "dense", "lexical"],
        help=argparse.SUPPRESS,
    )
    rag_eval.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help="How many candidate chunks to consider before the final top-k selection.",
    )
    rag_eval.add_argument(
        "--search-breadth",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    rag_eval.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT,
        help="Where to save the evaluation JSON output.",
    )
    rag_eval.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )

    rag_tune = subparsers.add_parser("rag-tune", help="Iteratively tune the local ESG RAG pipeline")
    rag_tune.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="CSV dataset containing ESG evaluation questions and expected contexts.",
    )
    rag_tune.add_argument(
        "--sample-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "sample_data",
        help="Directory containing the sample PDF reports used for tuning.",
    )
    rag_tune.add_argument(
        "--company",
        help="Limit tuning to a single company from the dataset.",
    )
    rag_tune.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many chunks to pass into the answer step.",
    )
    rag_tune.add_argument(
        "--chunk-targets",
        type=int,
        nargs="+",
        help="Chunk target token sizes to evaluate during the coarse pass.",
    )
    rag_tune.add_argument(
        "--chunk-overlaps",
        type=int,
        nargs="+",
        help="Chunk overlap sizes to evaluate during the coarse pass.",
    )
    rag_tune.add_argument(
        "--retrieval-modes",
        nargs="+",
        choices=["dense", "lexical", "hybrid", "semantic", "semantic_rerank"],
        help="Retrieval modes to compare after the chunking pass.",
    )
    rag_tune.add_argument(
        "--retrieval-architectures",
        nargs="+",
        choices=["semantic", "hybrid", "semantic_rerank", "dense", "lexical"],
        help=argparse.SUPPRESS,
    )
    rag_tune.add_argument(
        "--search-breadths",
        type=int,
        nargs="+",
        help="Candidate retrieval breadth values to compare after the chunking pass.",
    )
    rag_tune.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        help="Generation temperatures to compare once retrieval is fixed.",
    )
    rag_tune.add_argument(
        "--embedding-model",
        help="Albert embedding model id to use throughout tuning.",
    )
    rag_tune.add_argument(
        "--text-model",
        help="Albert text-generation model id to use throughout tuning.",
    )
    rag_tune.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many chunks to send per embeddings request while building indexes.",
    )
    rag_tune.add_argument(
        "--index-root",
        type=Path,
        default=DEFAULT_INDEX_DIR.parent / "rag_tune_indexes",
        help="Directory where temporary tuning indexes should be written.",
    )
    rag_tune.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TUNE_OUTPUT,
        help="Where to save the tuning JSON output.",
    )
    rag_tune.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )

    rag_eval_grid = subparsers.add_parser(
        "rag-eval-grid",
        help="Grid-search RAG chunking and retrieval parameters against the evaluation dataset.",
    )
    rag_eval_grid.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="CSV dataset containing ESG evaluation questions and expected contexts.",
    )
    rag_eval_grid.add_argument(
        "--company",
        required=True,
        help="Company to evaluate (required).",
    )
    rag_eval_grid.add_argument(
        "--sample-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "sample_data",
        help="Directory containing the sample PDF reports.",
    )
    rag_eval_grid.add_argument(
        "--chunk-targets",
        type=int,
        nargs="+",
        default=[250, 350, 420, 550, 700],
        help="Chunk target token sizes to evaluate.",
    )
    rag_eval_grid.add_argument(
        "--chunk-mins",
        type=int,
        nargs="+",
        default=[100, 200, 300],
        help="Chunk minimum token sizes to evaluate.",
    )
    rag_eval_grid.add_argument(
        "--chunk-maxs",
        type=int,
        nargs="+",
        default=[350, 500, 700, 900],
        help="Chunk maximum token sizes to evaluate.",
    )
    rag_eval_grid.add_argument(
        "--chunk-overlaps",
        type=int,
        nargs="+",
        default=[0, 50, 100, 150],
        help="Chunk overlap token values to evaluate.",
    )
    rag_eval_grid.add_argument(
        "--top-ks",
        type=int,
        nargs="+",
        default=[3, 5, 8, 10],
        help="Top-k retrieval values to evaluate.",
    )
    rag_eval_grid.add_argument(
        "--retrieval-modes",
        nargs="+",
        default=["dense", "hybrid"],
        choices=["dense", "lexical", "hybrid"],
        help="Retrieval modes to evaluate.",
    )
    rag_eval_grid.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="How many candidate chunks to consider before the final top-k selection.",
    )
    rag_eval_grid.add_argument(
        "--section-aware",
        action="store_true",
        help="Enable section/header-aware chunking for all indexes.",
    )
    rag_eval_grid.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many chunks to send per embeddings request while building indexes.",
    )
    rag_eval_grid.add_argument(
        "--embedding-model",
        help="Albert embedding model id to use throughout the grid search.",
    )
    rag_eval_grid.add_argument(
        "--index-root",
        type=Path,
        default=OUTPUT_DIR / "rag_grid_indexes",
        help="Directory where temporary grid indexes should be written.",
    )
    rag_eval_grid.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )

    ragas_eval = subparsers.add_parser("ragas-eval", help="Run RAGAS evaluation metrics on the ESG QA dataset")
    ragas_eval.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the built RAG index.",
    )
    ragas_eval.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="CSV dataset containing ESG evaluation questions.",
    )
    ragas_eval.add_argument(
        "--company",
        help="Company to evaluate (required).",
    )
    ragas_eval.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many chunks to retrieve per question.",
    )
    ragas_eval.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["dense", "lexical", "hybrid"],
        help="Retrieval mode.",
    )
    ragas_eval.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help="Candidate retrieval breadth.",
    )
    ragas_eval.add_argument(
        "--eval-mode",
        default="ragas",
        choices=["retrieval", "answer", "ragas", "all"],
        help="Evaluation mode.",
    )
    ragas_eval.add_argument(
        "--query-transform",
        choices=["none", "hyde", "mqr", "hyde_mqr"],
        default=None,
        help="Query transformation to apply.",
    )
    ragas_eval.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RAGAS_OUTPUT,
        help="Where to save the RAGAS evaluation JSON.",
    )
    ragas_eval.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Albert API base URL.",
    )

    ingest_targets = subparsers.add_parser(
        "ingest-target-reports", help="Download and store the curated ESG report set"
    )
    ingest_targets.add_argument(
        "--output-stem",
        default="target_esg_report_ingest",
        help="Output filename stem for the JSON and CSV summary files.",
    )

    streamlit_ui = subparsers.add_parser("streamlit-ui", help="Launch the Streamlit prompt console")
    streamlit_ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Streamlit app.",
    )
    streamlit_ui.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit app.",
    )
    return parser


def _resolve_retrieval_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "retrieval_mode", None)
    legacy = getattr(args, "retrieval_architecture", None)
    if legacy and legacy != argparse.SUPPRESS:
        return legacy
    if mode:
        return mode
    return "dense"


def _resolve_candidate_k(args: argparse.Namespace) -> int | None:
    ck = getattr(args, "candidate_k", None)
    sb = getattr(args, "search_breadth", None)
    if sb is not None:
        return sb
    if ck is not None:
        return ck
    return 12


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def seed_from_args(args: argparse.Namespace) -> CompanySeed:
    return CompanySeed(
        name=args.company,
        ticker=args.ticker,
        country=args.country,
        issuer_domain=args.issuer_domain,
        ranking_source="manual",
        ranking_url="manual",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    ensure_directories()

    if args.command == "smoke-test":
        rows, json_path, csv_path = run_smoke_test(top_n=args.top_n)
        print(f"Smoke test saved to {json_path} and {csv_path}")
        for row in rows:
            print(
                f"{row.company:20} | {row.chosen_report_title or 'NONE':40.40} | "
                f"{row.report_year or '-':4} | {row.confidence or 0:5.1f} | {row.parse_status or 'failed'}"
            )
        return 0

    if args.command == "rag-build":
        return run_rag_build(args)

    if args.command == "rag-ask":
        return run_rag_ask(args)

    if args.command == "rag-web":
        run_web_app(host=args.host, port=args.port, index_dir=args.index_dir)
        return 0

    if args.command == "rag-eval":
        return run_rag_eval(args)

    if args.command == "rag-eval-grid":
        return run_rag_eval_grid(args)

    if args.command == "ragas-eval":
        return run_ragas_eval(args)

    if args.command == "rag-tune":
        return run_rag_tune(args)

    if args.command == "ingest-target-reports":
        return run_ingest_target_reports(args)

    if args.command == "streamlit-ui":
        streamlit_path = Path(__file__).with_name("streamlit_ui.py")
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(streamlit_path),
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ]
        return subprocess.call(command)

    pipeline = AcquisitionPipeline()
    seed = seed_from_args(args)
    if args.command == "discover":
        profile, candidates = pipeline.discover(seed)
        print(f"Resolved domains: {', '.join(profile.issuer_domains) or 'none'}")
        for candidate in candidates[:15]:
            print(f"{candidate.score:5.1f} | {candidate.document_type:22} | {candidate.url}")
        return 0

    if args.command == "fetch":
        _profile, record, notes = pipeline.fetch_best(seed)
        if not record:
            print("No document fetched.")
            for note in notes:
                print(f"- {note}")
            return 1
        print(record.model_dump_json(indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
