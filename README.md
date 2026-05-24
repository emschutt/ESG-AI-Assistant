# ESG AI

ESG AI is a Python project for two related jobs:

1. discovering and downloading ESG, sustainability, and climate reports from company websites;
2. building a local RAG workflow on top of PDF reports so you can search and question them.

The project is designed for messy real-world company sites where reports may be hard to find, inconsistently named, or spread across different pages.

## What the Project Does

### Document acquisition

For a company name, the pipeline can:

- resolve likely issuer domains;
- discover report candidates from issuer pages and ranking sources;
- rank the candidates with deterministic heuristics;
- download the best matching file;
- extract text from the downloaded document;
- store metadata, candidate lists, and output files locally.

### Local RAG workflow

For one or more ESG PDFs, the project can:

- extract PDF text;
- split it into chunks;
- create embeddings through the Albert API;
- store a local search index;
- retrieve relevant chunks for a question;
- generate a grounded answer from the retrieved context;
- evaluate retrieval quality against the sample dataset;
- launch a small local browser UI for indexing and Q&A.

## Project Structure

```text
repo/
├── app/
│   ├── cli.py
│   ├── pipeline.py
│   ├── rag.py
│   ├── rag_eval.py
│   ├── web.py
│   ├── storage.py
│   ├── downloader.py
│   ├── source_discovery.py
│   ├── company_resolver.py
│   ├── candidate_ranker.py
│   ├── ranking_sources.py
│   └── parsers/
├── sample_data/
├── tests/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- Internet access for live document discovery and smoke tests
- `ALBERT_API_KEY` for RAG embedding and answer generation

## Installation

Run everything from the project root:

```bash
cd /Users/ems/Desktop/ESG_AI/repo
```

Create a virtual environment if you want an isolated setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Quick Start

List available commands:

```bash
python3 -m app.cli --help
```

Discover report candidates for a company:

```bash
python3 -m app.cli discover --company "ASML"
```

Fetch the best report for a company:

```bash
python3 -m app.cli fetch --company "ASML"
```

Build a RAG index from the bundled sample PDF:

```bash
python3 -m app.cli rag-build
```

When `--pdf` is omitted, `rag-build` recursively indexes the local PDF database from `sample_data/`, `outputs/downloaded_reports/`, `data/raw/`, and `esg_scraper/data/pdfs/`.

Ask a question against the built index:

```bash
python3 -m app.cli rag-ask "What climate targets are disclosed for 2030?"
```

## Command Reference

### `discover`

Finds and ranks likely ESG or sustainability report candidates for a company.

```bash
python3 -m app.cli discover --company "ASML"
python3 -m app.cli discover --company "ASML" --ticker ASML --country Netherlands --issuer-domain asml.com
```

### `fetch`

Runs discovery, downloads the best candidate, parses it, and stores the result locally.

```bash
python3 -m app.cli fetch --company "ASML"
```

### `smoke-test`

Runs the full acquisition flow on the top European listed companies by market cap.

```bash
python3 -m app.cli smoke-test
python3 -m app.cli smoke-test --top-n 10
```

### `rag-build`

Builds a local RAG index from one or more PDFs.

```bash
python3 -m app.cli rag-build
python3 -m app.cli rag-build --section-aware --contextual-chunking
python3 -m app.cli rag-build \
  --pdf /absolute/path/to/report.pdf \
  --chunk-target-tokens 420 --chunk-overlap-tokens 50 --section-aware
```

### `rag-ask`

Ask grounded questions with optional advanced retrieval:

```bash
# Basic
python3 -m app.cli rag-ask "What climate targets are disclosed for 2030?"

# HyDE retrieval
python3 -m app.cli rag-ask "Scope 1 emissions target" --query-transform hyde

# Multi-query retrieval with LLM reranking
python3 -m app.cli rag-ask "What is the carbon price?" --query-transform mqr --reranker llm --top-k 5

# Agentic retrieval with evidence evaluation
python3 -m app.cli rag-ask "TRIR 2024" --agentic

# With metadata filtering
python3 -m app.cli rag-ask "emissions targets" --filter-pillar environmental --filter-year 2024
```

The default Q&A path uses hybrid semantic plus lexical retrieval, a wider candidate pool, and deterministic answer generation. Answers are instructed to rely only on retrieved document context, cite chunk IDs, include short evidence excerpts, and flag uncertainty when the retrieved context is incomplete.

### `ragas-eval`

Run RAGAS evaluation metrics (faithfulness, answer relevancy, context precision/recall, hallucination detection):

```bash
python3 -m app.cli ragas-eval --company TotalEnergies
python3 -m app.cli ragas-eval --company TotalEnergies --eval-mode all --query-transform hyde
```

The RAGAS-style evaluator also reports retrieval hit, partial, and miss rates against the expected contexts in `sample_data/rag_evaluation_dataset.csv`, so answer quality and retrieval quality can be diagnosed separately.

### `mcp` — MCP Tool Architecture

Exposed as importable tools following MCP conventions:

```python
from app.mcp.server import list_tools, call_tool

tools = list_tools()  # 8 ESG tools with JSON schemas
result = call_tool("answer_question", {"question": "What is the carbon price?"})
```

Available tools: `search_esg_reports`, `retrieve_chunks`, `answer_question`, `run_ragas_eval`,
`discover_company_reports`, `ingest_report`, `compare_companies`, `extract_esg_metrics`

Use your own PDFs:

```bash
python3 -m app.cli rag-build \
  --pdf /absolute/path/to/report.pdf \
  --pdf /absolute/path/to/another-report.pdf
```

Customize chunking parameters:

```bash
python3 -m app.cli rag-build \
  --chunk-target-tokens 420 \
  --chunk-min-tokens 300 \
  --chunk-max-tokens 500 \
  --chunk-overlap-tokens 50 \
  --section-aware
```

Preview extraction and chunking without calling the API:

```bash
python3 -m app.cli rag-build --dry-run
```

### `rag-ask`

Retrieves relevant chunks from the local index and answers a question.

```bash
python3 -m app.cli rag-ask "What climate targets are disclosed for 2030?"
```

Use a specific retrieval mode:

```bash
python3 -m app.cli rag-ask "What climate targets are disclosed for 2030?" --retrieval-mode hybrid --top-k 8 --candidate-k 20
```

Retrieve chunks only:

```bash
python3 -m app.cli rag-ask "What climate targets are disclosed for 2030?" --search-only
```

### `rag-web`

Starts a small local browser UI for indexing documents and asking questions.

```bash
python3 -m app.cli rag-web
```

Then open [http://127.0.0.1:8787](http://127.0.0.1:8787).

### `streamlit-ui`

Launches the Streamlit prompt console with a CLI-style dark theme.

```bash
python3 -m app.cli streamlit-ui
```

If you prefer to run Streamlit directly:

```bash
streamlit run app/streamlit_ui.py
```

The sidebar lets you select an Albert text-generation model, tune `temperature` and `top_k`, and upload a document to answer from it.

### `rag-eval`

Evaluates retrieval quality against the sample ESG question dataset.

```bash
python3 -m app.cli rag-eval
python3 -m app.cli rag-eval --company TotalEnergies --retrieval-mode hybrid --top-k 8
```

### `rag-eval-grid`

Grid-search optimization that tests many chunking and retrieval parameter combinations and ranks them by hit rate, partial+hit rate, and best match score.

```bash
python3 -m app.cli rag-eval-grid --company TotalEnergies
```

Customize the search space:

```bash
python3 -m app.cli rag-eval-grid \
  --company TotalEnergies \
  --chunk-targets 250 350 420 550 700 \
  --chunk-mins 100 200 300 \
  --chunk-maxs 350 500 700 900 \
  --chunk-overlaps 0 50 100 150 \
  --top-ks 3 5 8 10 \
  --retrieval-modes dense hybrid \
  --section-aware
```

Output files produced:
- `outputs/rag_eval_grid_results.json` — full grid search results with per-question diagnostics
- `outputs/rag_eval_grid_results.csv` — tabular summary of all configurations
- `outputs/best_rag_config.json` — the single best configuration across all metrics

## Environment Variables

Set your Albert API key before running `rag-build`, `rag-ask`, or `rag-eval` without `--dry-run`:

```bash
export ALBERT_API_KEY="your-api-key"
```

## Output Locations

The project creates local output folders automatically.

- `data/raw/`: downloaded original files
- `data/text/`: extracted text files
- `data/candidates/`: saved candidate rankings
- `data/metadata.db`: SQLite metadata database
- `outputs/`: smoke test outputs, RAG index files, and evaluation results

## Tests

Run the test suite with:

```bash
python3 -m pytest
```

## Notes

- The document acquisition flow depends on live websites, so results can change over time.
- `faiss-cpu` is included for vector search, but the code also contains a NumPy fallback when FAISS is unavailable.
- Sample ESG data and a sample PDF are included under `sample_data/`.
