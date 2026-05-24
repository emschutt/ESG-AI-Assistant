"""Lightweight browser UI for the ESG RAG workflow."""

from __future__ import annotations

import html
import os
import re
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from app.llm import ChatModel, load_model_catalog
from app.rag import (
    DEFAULT_INDEX_DIR,
    DEFAULT_SAMPLE_PDF,
    DEFAULT_BASE_URL,
    answer_question,
    build_index,
    get_index_summary,
    parse_pdf_path_lines,
    retrieve_chunks,
    set_api_key,
)


DEFAULT_TOP_K = 5
DEFAULT_QUESTION = "What climate-related targets are disclosed for 2030, and are they science-based?"


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _render_inline_markdown(text: str) -> str:
    rendered = _escape(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    return rendered


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _split_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _render_markdown_table(lines: list[str]) -> str:
    header = _split_table_cells(lines[0])
    body_lines = lines[2:] if len(lines) > 1 and _is_table_separator(lines[1]) else lines[1:]
    header_html = "".join(f"<th>{_render_inline_markdown(cell)}</th>" for cell in header)
    rows_html = []
    for row in body_lines:
        cells = _split_table_cells(row)
        rows_html.append("<tr>" + "".join(f"<td>{_render_inline_markdown(cell)}</td>" for cell in cells) + "</tr>")
    return f"""
<div class="table-wrap">
  <table>
    <thead><tr>{header_html}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
"""


def _render_markdown(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    html_parts: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            html_parts.append(f"<p>{_render_inline_markdown(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
            list_items.clear()

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_list()
            index += 1
            continue

        if _is_table_row(stripped):
            table_lines = [stripped]
            cursor = index + 1
            while cursor < len(lines) and _is_table_row(lines[cursor].strip()):
                table_lines.append(lines[cursor].strip())
                cursor += 1
            if len(table_lines) >= 2 and _is_table_separator(table_lines[1]):
                flush_paragraph()
                flush_list()
                html_parts.append(_render_markdown_table(table_lines))
                index = cursor
                continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_paragraph()
            list_items.append(_render_inline_markdown(stripped[2:].strip()))
            index += 1
            continue

        if re.fullmatch(r"\*\*[^*]+\*\*", stripped):
            flush_paragraph()
            flush_list()
            html_parts.append(f"<h3>{_render_inline_markdown(stripped).replace('<strong>', '').replace('</strong>', '')}</h3>")
            index += 1
            continue

        flush_list()
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    flush_list()
    return "\n".join(html_parts)


def _coerce_top_k(value: Any, default: int = DEFAULT_TOP_K) -> int:
    try:
        return min(max(int(value), 1), 20)
    except (TypeError, ValueError):
        return default


def _refresh_model_catalog(settings: dict[str, Any], *, force: bool = False) -> None:
    api_key = os.environ.get("ALBERT_API_KEY", "").strip()
    if not api_key:
        settings["models"] = []
        settings["model_warnings"] = ["Save an Albert API key to load text models."]
        settings["models_loaded"] = False
        return

    if settings.get("models_loaded") and not force:
        return

    models, warnings = load_model_catalog(api_key, base_url=DEFAULT_BASE_URL)
    settings["models"] = models
    settings["model_warnings"] = warnings
    settings["models_loaded"] = True
    selected_model = str(settings.get("text_model", "")).strip()
    model_ids = {model.model_id for model in models}
    if selected_model not in model_ids:
        settings["text_model"] = models[0].model_id if models else ""


def _render_model_options(models: list[ChatModel], selected_model: str) -> str:
    if not models:
        return '<option value="">Save API key to load models</option>'
    return "\n".join(
        (
            f'<option value="{_escape(model.model_id)}"'
            f'{" selected" if model.model_id == selected_model else ""}>'
            f'{_escape(model.model_id)}</option>'
        )
        for model in models
    )


def _render_settings_panel(settings: dict[str, Any]) -> str:
    _refresh_model_catalog(settings)
    models = settings.get("models", [])
    selected_model = str(settings.get("text_model", "")).strip()
    model_options = _render_model_options(models, selected_model)
    model_warnings = "".join(
        f'<p class="muted">{_escape(warning)}</p>' for warning in settings.get("model_warnings", [])
    )
    key_placeholder = "saved; paste new key to replace" if os.environ.get("ALBERT_API_KEY") else "sk-..."
    return f"""
<section class="panel">
  <h2>Settings</h2>
  <form method="post" action="/settings">
    <label>
      Albert API Key
      <input type="password" name="api_key" placeholder="{_escape(key_placeholder)}">
    </label>
    <label>
      Text model
      <select name="text_model"{" disabled" if not models else ""}>
        {model_options}
      </select>
    </label>
    <label>
      Top K chunks
      <input type="number" name="top_k" value="{_escape(_coerce_top_k(settings.get("top_k")))}" min="1" max="20">
    </label>
    <div class="actions">
      <button type="submit">save settings</button>
    </div>
    {model_warnings}
  </form>
</section>
"""


def _render_page(
    *,
    index_dir: Path,
    settings: dict[str, Any],
    build_result: dict[str, Any] | None = None,
    ask_result: dict[str, Any] | None = None,
    error_message: str | None = None,
    status_message: str | None = None,
) -> str:
    summary = get_index_summary(index_dir)
    summary_html = _render_index_summary(summary, index_dir)
    settings_html = _render_settings_panel(settings)
    build_html = _render_build_result(build_result)
    ask_html = _render_ask_result(ask_result)
    error_html = (
        f'<section class="panel panel-error"><h2>Issue</h2><pre>{_escape(error_message)}</pre></section>'
        if error_message
        else ""
    )
    status_html = (
        f'<section class="panel panel-success"><h2>Status</h2><p>{_escape(status_message)}</p></section>'
        if status_message
        else ""
    )
    sample_value = _escape(str(DEFAULT_SAMPLE_PDF.resolve()))
    question_value = _escape(settings.get("question") or DEFAULT_QUESTION)
    current_key_hint = "present" if os.environ.get("ALBERT_API_KEY") else "missing"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESG RAG Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-deep: #07080e;
      --bg-surface: #0d0f1a;
      --bg-elevated: #131522;
      --bg-input: #0a0b14;
      --border: #1a1c2e;
      --border-hover: #252840;
      --text-primary: #e1e4f0;
      --text-secondary: #8b8fa8;
      --text-muted: #5c607a;
      --accent-green: #10b981;
      --accent-green-strong: #059669;
      --accent-coral: #f43f5e;
      --accent-cyan: #22d3ee;
      --glow-green: rgba(16,185,129,0.18);
      --glow-coral: rgba(244,63,94,0.18);
      --radius: 8px;
      --radius-sm: 6px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "JetBrains Mono", "IBM Plex Mono", "SF Mono", "Menlo", "Cascadia Code", monospace;
      font-size: 13px;
      line-height: 1.65;
      color: var(--text-primary);
      background: var(--bg-deep);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(16,185,129,0.04), transparent),
        radial-gradient(ellipse 60% 40% at 80% 85%, rgba(34,211,238,0.03), transparent);
      pointer-events: none;
      z-index: 0;
    }}
    .shell {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 18px 22px 44px;
      position: relative;
      z-index: 1;
    }}
    .hero {{
      padding: 14px 18px;
      border: 1px solid var(--border);
      background: var(--bg-surface);
      border-radius: var(--radius);
      margin-bottom: 14px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
      opacity: 0.5;
    }}
    .hero h1 {{
      margin: 0 0 4px;
      font-size: 1.25rem;
      font-weight: 600;
      letter-spacing: 0;
      color: var(--text-primary);
    }}
    .hero h1 span {{
      color: var(--accent-green);
    }}
    .hero p {{
      margin: 0;
      max-width: 80ch;
      color: var(--text-secondary);
      font-size: 0.8rem;
    }}
    .hero-badges {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .badge {{
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 5px 10px;
      font-size: 0.7rem;
      background: var(--bg-elevated);
      color: var(--text-secondary);
      letter-spacing: 0.02em;
    }}
    .badge.ok {{ border-color: rgba(16,185,129,0.25); color: var(--accent-green); }}
    .badge.warn {{ border-color: rgba(244,63,94,0.25); color: var(--accent-coral); }}
    .workspace {{
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(0, 2.8fr) minmax(280px, 0.8fr);
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 16px;
    }}
    .panel {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
    }}
    .panel-primary {{
      padding: 20px 22px;
    }}
    .panel h2 {{
      margin: 0 0 8px;
      font-size: 0.9rem;
      font-weight: 600;
      letter-spacing: 0.01em;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .panel h2::before {{
      content: ">";
      color: var(--accent-green);
      font-weight: 400;
    }}
    .panel p, .panel li, .meta {{
      color: var(--text-secondary);
      font-size: 0.75rem;
      margin: 0 0 10px;
    }}
    .panel-error {{
      border-color: rgba(244,63,94,0.3);
      background: rgba(244,63,94,0.06);
    }}
    .panel-error h2::before {{
      color: var(--accent-coral);
    }}
    .panel-success {{
      border-color: rgba(16,185,129,0.28);
      background: rgba(16,185,129,0.06);
    }}
    form {{
      display: grid;
      gap: 12px;
    }}
    .row {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    label {{
      display: grid;
      gap: 5px;
      font-size: 0.7rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    input, textarea, select {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 8px 10px;
      font-family: inherit;
      font-size: 0.78rem;
      color: var(--text-primary);
      background: var(--bg-input);
      outline: none;
      transition: border-color 160ms ease;
    }}
    input:focus, textarea:focus, select:focus {{
      border-color: var(--accent-green);
      box-shadow: 0 0 0 2px var(--glow-green);
    }}
    input:read-only {{
      color: var(--text-muted);
      cursor: default;
    }}
    textarea {{
      min-height: 100px;
      resize: vertical;
      line-height: 1.5;
    }}
    textarea[name="question"] {{
      min-height: 220px;
    }}
    textarea[name="pdf_paths"] {{
      min-height: 80px;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--accent-green);
      border-radius: var(--radius-sm);
      background: rgba(16,185,129,0.1);
      color: var(--accent-green);
      padding: 8px 16px;
      font-family: inherit;
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 140ms ease, box-shadow 140ms ease;
      letter-spacing: 0.02em;
    }}
    button:hover {{
      background: rgba(16,185,129,0.18);
      box-shadow: 0 0 12px var(--glow-green);
    }}
    .muted {{
      color: var(--text-muted);
      font-size: 0.68rem;
    }}
    .stats {{
      display: grid;
      gap: 8px;
      grid-template-columns: 1fr;
      margin-top: 10px;
    }}
    .stat {{
      border: 1px solid var(--border);
      background: var(--bg-elevated);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--accent-green);
      margin-bottom: 2px;
    }}
    .stat span {{
      color: var(--text-muted);
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 12px;
      margin: 0;
      font-family: inherit;
      font-size: 0.78rem;
      line-height: 1.55;
      color: var(--text-secondary);
      overflow-x: auto;
    }}
    .answer-output pre {{
      min-height: 260px;
      color: #f8fafc;
    }}
    .answer-output .meta {{
      color: #cbd5e1;
    }}
    .markdown-body {{
      min-height: 260px;
      color: #f8fafc;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 16px 18px;
      overflow-x: auto;
    }}
    .markdown-body h3 {{
      margin: 18px 0 8px;
      color: #ffffff !important;
      font-size: 0.95rem !important;
      font-weight: 700 !important;
      text-transform: none;
      letter-spacing: 0;
    }}
    .markdown-body h3:first-child {{
      margin-top: 0;
    }}
    .markdown-body p,
    .markdown-body li {{
      color: #f8fafc !important;
      font-size: 0.82rem;
      line-height: 1.65;
    }}
    .markdown-body ul {{
      margin: 6px 0 14px;
      padding-left: 22px;
    }}
    .markdown-body strong {{
      color: #ffffff;
      font-weight: 700;
    }}
    .markdown-body code {{
      color: #d1fae5 !important;
      border-color: rgba(16,185,129,0.18) !important;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin: 12px 0 18px;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
    }}
    .markdown-body table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    .markdown-body th,
    .markdown-body td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      border-right: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
      color: #f8fafc;
      font-size: 0.76rem;
      line-height: 1.5;
    }}
    .markdown-body th {{
      background: var(--bg-elevated);
      color: #ffffff;
      font-weight: 700;
    }}
    .markdown-body tr:last-child td {{
      border-bottom: 0;
    }}
    .markdown-body th:last-child,
    .markdown-body td:last-child {{
      border-right: 0;
    }}
    details.index-tools {{
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--bg-surface);
      overflow: hidden;
    }}
    details.index-tools > summary {{
      cursor: pointer;
      list-style: none;
      padding: 14px 18px;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text-primary);
    }}
    details.index-tools > summary::-webkit-details-marker {{
      display: none;
    }}
    details.index-tools > summary::before {{
      content: ">";
      color: var(--accent-green);
      font-weight: 400;
      margin-right: 8px;
    }}
    details.index-tools .panel {{
      border: 0;
      border-top: 1px solid var(--border);
      border-radius: 0;
    }}
    .chunk {{
      display: grid;
      gap: 6px;
      padding: 10px 12px;
      border-radius: var(--radius-sm);
      background: var(--bg-elevated);
      border: 1px solid var(--border);
    }}
    .chunk + .chunk {{
      margin-top: 10px;
    }}
    .chunk-header {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px 10px;
      font-size: 0.68rem;
      color: var(--text-muted);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      background: rgba(16,185,129,0.1);
      color: var(--accent-green);
      font-size: 0.65rem;
      border: 1px solid rgba(16,185,129,0.15);
    }}
    ul {{
      padding-left: 16px;
      color: var(--text-secondary);
      font-size: 0.75rem;
    }}
    .indexed-files {{
      margin-top: 10px;
      color: var(--text-secondary);
      font-size: 0.72rem;
    }}
    .indexed-files summary {{
      cursor: pointer;
      color: var(--text-muted);
    }}
    .indexed-files ul {{
      max-height: 180px;
      overflow: auto;
      margin-bottom: 0;
    }}
    @media (max-width: 980px) {{
      .workspace, .row, .stats {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        padding: 16px 10px 36px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>ESG RAG Studio <span>v1.0</span></h1>
      <p>Build a searchable ESG knowledge base from PDF reports — index → embed → retrieve → answer.</p>
      <div class="hero-badges">
        <span class="badge {('ok' if current_key_hint == 'present' else 'warn')}">API key: {_escape(current_key_hint)}</span>
        <span class="badge">&gt; index {_escape(str(index_dir))}</span>
      </div>
    </section>

    {error_html}
    {status_html}

    <section class="workspace">
      <div class="stack">
        <section class="panel panel-primary">
          <h2>Ask Question</h2>
          <p>Query the current index and generate an evidenced answer.</p>
          <form method="post" action="/ask">
            <label>
              ESG question
              <textarea name="question">{question_value}</textarea>
            </label>
            <div class="actions">
              <button type="submit">query</button>
            </div>
          </form>
        </section>

        {ask_html}
      </div>

      <aside class="stack">
        {settings_html}
        {summary_html}
        <details class="index-tools">
          <summary>Build Index</summary>
          <section class="panel">
          <p>Paste absolute PDF paths (one per line), or leave blank to index the full local PDF database.</p>
          <form method="post" action="/build">
            <label>
              PDF paths
              <textarea name="pdf_paths" placeholder="{sample_value}"></textarea>
            </label>
            <div class="row">
              <label>
                Chunk target tokens
                <input type="number" name="chunk_target_tokens" value="420" min="100" max="1000">
              </label>
              <label>
                Batch size
                <input type="number" name="batch_size" value="16" min="1" max="128">
              </label>
            </div>
            <div class="row">
              <label>
                Chunk overlap tokens
                <input type="number" name="chunk_overlap_tokens" value="50" min="0" max="300">
              </label>
              <label>
                Section-aware
                <input type="checkbox" name="section_aware" value="1">
              </label>
            </div>
            <div class="actions">
              <button type="submit">build</button>
              <span class="muted">FAISS auto-detected; NumPy fallback available.</span>
            </div>
          </form>
          </section>
        </details>
        {build_html}
      </aside>
    </section>
  </main>
</body>
</html>
"""


def _render_index_summary(summary: dict[str, Any] | None, index_dir: Path) -> str:
    if not summary:
        return f"""
<section class="panel">
  <h2>Index Status</h2>
  <p>No index built yet in <strong>{_escape(str(index_dir))}</strong>. Open <span style="color:var(--accent-green)">Build Index</span> in the side panel to start querying.</p>
</section>
"""

    pdf_list = "".join(f"<li>{_escape(pdf)}</li>" for pdf in summary.get("pdfs", []))
    return f"""
<section class="panel">
  <h2>Index Status</h2>
  <div class="stats">
    <div class="stat"><strong>{_escape(summary.get("chunk_count", 0))}</strong><span>Chunks</span></div>
    <div class="stat"><strong>{_escape(summary.get("vector_backend", "unknown"))}</strong><span>Vector backend</span></div>
    <div class="stat"><strong>{_escape(summary.get("embedding_model", "not built"))}</strong><span>Embedding model</span></div>
    <div class="stat"><strong>{_escape(summary.get("embedding_dimension", "n/a"))}</strong><span>Dimensions</span></div>
  </div>
  <p class="meta">Built: {_escape(summary.get("built_at", "unknown"))}</p>
  <details class="indexed-files">
    <summary>{_escape(len(summary.get("pdfs", [])))} indexed file(s)</summary>
    <ul>{pdf_list}</ul>
  </details>
</section>
"""


def _render_build_result(build_result: dict[str, Any] | None) -> str:
    if not build_result:
        return ""
    return f"""
<section class="panel">
  <h2>Latest Build</h2>
  <pre>{_escape(json_dumps_pretty(build_result))}</pre>
</section>
"""


def _render_ask_result(ask_result: dict[str, Any] | None) -> str:
    if not ask_result:
        return ""

    answer_html = _render_markdown(ask_result["answer"])
    chunks_html = "".join(
        f"""
<div class="chunk">
  <div class="chunk-header">
    <span class="pill">{_escape(chunk["chunk_id"])}</span>
    <span>{_escape(chunk["source_file"])}</span>
    <span>pages {_escape(chunk["page_start"])}-{_escape(chunk["page_end"])}</span>
    <span>score {_escape(f'{chunk["score"]:.4f}')}</span>
  </div>
  <pre>{_escape(chunk["text"])}</pre>
</div>
"""
        for chunk in ask_result["retrieved_chunks"]
    )
    return f"""
<section class="panel answer-output">
  <h2>Answer</h2>
  <p class="meta">Embedding model: {_escape(ask_result["embedding_model"])} | Text model: {_escape(ask_result["text_model"])}</p>
  <div class="markdown-body">{answer_html}</div>
</section>
<details class="index-tools retrieved-chunks">
  <summary>Retrieved Chunks ({_escape(len(ask_result["retrieved_chunks"]))})</summary>
  <section class="panel">
    {chunks_html}
  </section>
</details>
"""


def json_dumps_pretty(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)


def _read_form(environ: dict[str, Any]) -> dict[str, str]:
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
    except ValueError:
        length = 0
    body = environ["wsgi.input"].read(length).decode("utf-8")
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def _build_handler(form: dict[str, str], index_dir: Path) -> dict[str, Any]:
    pdf_paths = parse_pdf_path_lines(form.get("pdf_paths", ""))
    return build_index(
        pdf_paths=pdf_paths,
        index_dir=index_dir,
        target_tokens=int(form.get("chunk_target_tokens", "420") or 420),
        min_tokens=300,
        max_tokens=500,
        overlap_tokens=int(form.get("chunk_overlap_tokens", "50") or 0),
        section_aware=form.get("section_aware") == "1",
        batch_size=int(form.get("batch_size", "16") or 16),
        base_url=DEFAULT_BASE_URL,
    )


def _settings_handler(form: dict[str, str], settings: dict[str, Any]) -> str:
    api_key = form.get("api_key", "").strip()
    if api_key:
        set_api_key(api_key)
        settings["models_loaded"] = False

    settings["top_k"] = _coerce_top_k(form.get("top_k"), default=_coerce_top_k(settings.get("top_k")))
    selected_text_model = form.get("text_model", "").strip()
    if selected_text_model:
        settings["text_model"] = selected_text_model

    _refresh_model_catalog(settings, force=bool(api_key))
    return "Settings saved. Future queries will reuse this API key, text model, and Top K value."


def _ask_handler(form: dict[str, str], index_dir: Path, settings: dict[str, Any]) -> dict[str, Any]:
    question = form.get("question", "").strip()
    if not question:
        raise RuntimeError("Please enter a question before submitting.")

    settings["question"] = question
    top_k = _coerce_top_k(settings.get("top_k"))
    text_model = str(settings.get("text_model", "")).strip() or None
    retrieved_chunks, embedding_model = retrieve_chunks(
        index_dir=index_dir,
        question=question,
        top_k=top_k,
        base_url=DEFAULT_BASE_URL,
    )
    answer, text_model = answer_question(
        question=question,
        retrieved_chunks=retrieved_chunks,
        text_model=text_model,
        base_url=DEFAULT_BASE_URL,
    )
    return {
        "question": question,
        "embedding_model": embedding_model,
        "text_model": text_model,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }


def create_app(index_dir: Path):
    settings: dict[str, Any] = {
        "top_k": DEFAULT_TOP_K,
        "question": DEFAULT_QUESTION,
        "text_model": "",
        "models": [],
        "model_warnings": [],
        "models_loaded": False,
    }

    def app(environ: dict[str, Any], start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()
        status = "200 OK"
        build_result: dict[str, Any] | None = None
        ask_result: dict[str, Any] | None = None
        error_message: str | None = None
        status_message: str | None = None

        try:
            if path == "/settings" and method == "POST":
                form = _read_form(environ)
                status_message = _settings_handler(form, settings)
            elif path == "/build" and method == "POST":
                form = _read_form(environ)
                build_result = _build_handler(form, index_dir)
            elif path == "/ask" and method == "POST":
                form = _read_form(environ)
                ask_result = _ask_handler(form, index_dir, settings)
            elif path != "/":
                status = "404 Not Found"
                error_message = f"Unknown route: {path}"
        except Exception as exc:  # noqa: BLE001
            error_message = f"{exc}\n\n{traceback.format_exc(limit=2)}"
            status = "500 Internal Server Error"

        body = _render_page(
            index_dir=index_dir,
            settings=settings,
            build_result=build_result,
            ask_result=ask_result,
            error_message=error_message,
            status_message=status_message,
        ).encode("utf-8")
        headers = [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))]
        start_response(status, headers)
        return [body]

    return app


def run_web_app(host: str, port: int, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
    app = create_app(index_dir=index_dir)
    print(f"Serving ESG RAG Studio on http://{host}:{port}")
    with make_server(host, port, app) as server:
        server.serve_forever()
