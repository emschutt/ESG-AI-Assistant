"""Verification for fine-grained filters + numbered-citation answer path.

Runs the TE-CEO unblock test from the prompt, asserts every emitted [n]
maps to a real chunk_id, asserts each fine filter is exactly satisfied by
the returned chunks, exercises the impossible-filter empty path, and
records everything to outputs/filters_citations_smoke_report.md.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EMBED_DIR = REPO.parent / "esg_scraper" / "data" / "embeddings"
EMBED_KEY_PATH = REPO.parent / "esg_scraper" / ".env"
INDEX_DIR = REPO / "outputs" / "rag_index"
REPORT_PATH = REPO / "outputs" / "filters_citations_smoke_report.md"
TEXT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def _load_dotenv(p: Path) -> None:
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    _load_dotenv(EMBED_KEY_PATH)
    api_key = os.environ.get("ALBERT_API_KEY", "")
    if not api_key:
        raise SystemExit("ALBERT_API_KEY missing")

    from app.rag import (  # noqa: E402
        _load_chunks,
        answer_question_cited,
        retrieve_chunks,
        set_api_key,
    )
    set_api_key(api_key)

    results: list[tuple[str, bool, str]] = []
    record: dict[str, object] = {}

    # 1) Backward-compat load + spot-check new fields populated
    chunks = _load_chunks(INDEX_DIR)
    spot = []
    for i in (0, 50_000, 146_000):
        if i < len(chunks):
            c = chunks[i]
            spot.append({
                "i": i, "chunk_id": c.chunk_id, "company": c.company,
                "chunk_kind": c.chunk_kind, "speaker_role": c.speaker_role,
                "doc_type": c.doc_type,
            })
    non_empty = sum(1 for s in spot if s["company"] and s["chunk_kind"])
    results.append((
        "load_with_new_fields", non_empty == len(spot) and len(chunks) == 146920,
        f"N={len(chunks)} spot={spot}",
    ))

    # 2) TE-CEO unblock — the headline test
    question = "most recent vision from TotalEnergies CEO on ESG"

    te_executive_results, used_embed_a = retrieve_chunks(
        index_dir=INDEX_DIR,
        question=question,
        top_k=8,
        retrieval_architecture="dense",
        search_breadth=50,
        filter_company="TotalEnergies",
        filter_speaker_role="executive",
    )
    te_progress_results, used_embed_b = retrieve_chunks(
        index_dir=INDEX_DIR,
        question=question,
        top_k=8,
        retrieval_architecture="dense",
        search_breadth=50,
        filter_company="TotalEnergies",
        filter_doc_type="progress_report",
    )

    # Filter correctness assertions (Task 4)
    all_te_a = all(c.get("company") == "TotalEnergies" for c in te_executive_results)
    all_exec_a = all(c.get("speaker_role") == "executive" for c in te_executive_results)
    all_te_b = all(c.get("company") == "TotalEnergies" for c in te_progress_results)
    all_progress_b = all(c.get("doc_type") == "progress_report" for c in te_progress_results)
    results.append((
        "filter_correctness_te_executive",
        bool(te_executive_results) and all_te_a and all_exec_a,
        f"n={len(te_executive_results)} all_TE={all_te_a} all_executive={all_exec_a}",
    ))
    results.append((
        "filter_correctness_te_progress_report",
        bool(te_progress_results) and all_te_b and all_progress_b,
        f"n={len(te_progress_results)} all_TE={all_te_b} all_progress_report={all_progress_b}",
    ))

    # Cited answer over the executive set
    answer_a, model_a, cit_map_a, sources_a = answer_question_cited(
        question=question,
        retrieved_chunks=te_executive_results,
        text_model=TEXT_MODEL,
        temperature=0.0,
    )
    answer_b, model_b, cit_map_b, sources_b = answer_question_cited(
        question=question,
        retrieved_chunks=te_progress_results,
        text_model=TEXT_MODEL,
        temperature=0.0,
    )

    # 3) Citation integrity — every [n] emitted by the model maps to a real
    # chunk_id in chunk_ids.json, and there are no bare suffix labels.
    real_ids = set(json.loads((EMBED_DIR / "chunk_ids.json").read_text(encoding="utf-8")))
    BRACKET_NUM = re.compile(r"\[(\d{1,3})\]")
    BAD_LABEL = re.compile(r"\[(chunk-\d+|t\d+|n\d+)\]", re.IGNORECASE)

    def _citation_check(answer: str, cit_map: dict, label: str):
        emitted = sorted({int(m) for m in BRACKET_NUM.findall(answer)})
        bad = BAD_LABEL.findall(answer)
        unresolved = [n for n in emitted if n not in cit_map]
        valid_chunk_ids = [
            cit_map[n] for n in emitted if n in cit_map and cit_map[n] in real_ids
        ]
        ok = (
            len(emitted) > 0
            and not bad
            and not unresolved
            and len(valid_chunk_ids) == len(emitted)
        )
        return ok, {
            "emitted_numbers": emitted,
            "bad_suffix_labels": bad,
            "unresolved_numbers": unresolved,
            "resolved_chunk_ids": valid_chunk_ids,
        }

    ok_a, det_a = _citation_check(answer_a, cit_map_a, "executive")
    ok_b, det_b = _citation_check(answer_b, cit_map_b, "progress_report")
    results.append((
        "citation_integrity_executive_run", ok_a,
        json.dumps(det_a, ensure_ascii=False),
    ))
    results.append((
        "citation_integrity_progress_run", ok_b,
        json.dumps(det_b, ensure_ascii=False),
    ))

    # Answer engaged with the context: a "blanket refusal" is an answer with
    # the formal refusal phrase AND zero emitted citations. An answer that
    # emits ≥1 citation has substantively used the retrieved chunks, even if
    # it also disclaims a narrow part of the question (e.g. "no direct CEO
    # quote here, but the doc says X[1] Y[2]") — which is correct behaviour.
    def _engaged(answer: str, det: dict) -> bool:
        emitted = det.get("emitted_numbers") or []
        return bool(answer) and len(emitted) >= 1
    results.append((
        "te_ceo_unblock_executive_run", _engaged(answer_a, det_a),
        f"len={len(answer_a or '')} citations={det_a.get('emitted_numbers')} "
        f"excerpt={(answer_a or '')[:180]!r}",
    ))
    results.append((
        "te_ceo_unblock_progress_run", _engaged(answer_b, det_b),
        f"len={len(answer_b or '')} citations={det_b.get('emitted_numbers')} "
        f"excerpt={(answer_b or '')[:180]!r}",
    ))

    # 4) Impossible-filter empty path
    impossible, _ = retrieve_chunks(
        index_dir=INDEX_DIR,
        question="anything at all",
        top_k=5,
        retrieval_architecture="dense",
        search_breadth=50,
        filter_company="TotalEnergies",
        filter_doc_type="vigilance_plan",  # TE has no vigilance_plan
    )
    results.append((
        "impossible_filter_returns_empty", impossible == [],
        f"n_results={len(impossible)}",
    ))

    # 5) Default no-filter path still works (caller arity)
    plain_results, _ = retrieve_chunks(
        index_dir=INDEX_DIR,
        question="Scope 1 and Scope 2 greenhouse gas emissions",
        top_k=5,
        retrieval_architecture="dense",
    )
    plain_ans, plain_model, plain_map, plain_src = answer_question_cited(
        question="Scope 1 and Scope 2 greenhouse gas emissions",
        retrieved_chunks=plain_results,
        text_model=TEXT_MODEL,
        temperature=0.0,
    )
    ok_plain, det_plain = _citation_check(plain_ans, plain_map, "no_filter")
    results.append((
        "default_no_filter_path", bool(plain_ans) and ok_plain,
        f"len={len(plain_ans)} citation_ok={ok_plain} details={det_plain}",
    ))

    # Single-doc Streamlit path AST integrity (we didn't touch it but verify still parses)
    import ast
    try:
        ast.parse((REPO / "app" / "streamlit_ui.py").read_text(encoding="utf-8"))
        results.append(("streamlit_ui_ast", True, "parses ok"))
    except Exception as e:
        results.append(("streamlit_ui_ast", False, f"{type(e).__name__}: {e}"))

    record["te_executive_answer"] = answer_a
    record["te_executive_citation_map"] = cit_map_a
    record["te_executive_sources"] = [
        {
            "chunk_id": s["chunk_id"], "company": s["company"],
            "doc_type": s["doc_type"], "speaker_role": s["speaker_role"],
            "report_year": s["report_year"], "page_start": s["page_start"],
            "score": round(s["score"], 4),
        }
        for s in sources_a
    ]
    record["te_progress_answer"] = answer_b
    record["te_progress_citation_map"] = cit_map_b

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# Filters + citations smoke report")
    L.append("")
    L.append(f"- index_dir: `{INDEX_DIR}`")
    L.append(f"- N chunks: {len(chunks)}")
    L.append(f"- chat model: `{TEXT_MODEL}`")
    L.append("")
    L.append("## Verification")
    all_pass = all(p for _, p, _ in results)
    L.append(f"- Overall: **{'PASS' if all_pass else 'FAIL'}**")
    for name, p, d in results:
        L.append(f"  - {name}: {'PASS' if p else 'FAIL'} — {d}")
    L.append("")

    L.append("## TE-CEO unblock — executive-filter run")
    L.append("")
    L.append("### Citation map")
    for n, cid in sorted(cit_map_a.items()):
        s = sources_a[n - 1]
        L.append(
            f"- **[{n}]** `{cid}` — {s.get('company')} · {s.get('doc_type')} · "
            f"FY{s.get('report_year')} · `{s['source_file']}` p.{s['page_start']} "
            f"(score {s['score']:.4f})"
        )
    L.append("")
    L.append("### Answer")
    L.append("")
    L.append("```")
    L.append(answer_a)
    L.append("```")
    L.append("")
    L.append("## TE-CEO unblock — progress_report filter run")
    L.append("")
    L.append("### Citation map")
    for n, cid in sorted(cit_map_b.items()):
        s = sources_b[n - 1]
        L.append(
            f"- **[{n}]** `{cid}` — {s.get('company')} · {s.get('doc_type')} · "
            f"FY{s.get('report_year')} · `{s['source_file']}` p.{s['page_start']} "
            f"(score {s['score']:.4f})"
        )
    L.append("")
    L.append("### Answer")
    L.append("")
    L.append("```")
    L.append(answer_b)
    L.append("```")
    L.append("")
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"[verify] {'PASS' if all_pass else 'FAIL'}")
    for name, p, d in results:
        print(f"  - {name}: {'PASS' if p else 'FAIL'} — {d[:160]}")
    print(f"\n[report] wrote {REPORT_PATH}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
