"""End-to-end Streamlit provenance test.

Drives `app/streamlit_ui.py` headlessly via `streamlit.testing.v1.AppTest` for
two scenarios:
  (a) no-filter prompt asking about Engie's Scope 1 emissions
  (b) filtered prompt asking about TotalEnergies CEO's ESG vision, with
      company=TotalEnergies and speaker_role=executive

For each scenario it captures the answer + sources + citation map from
`session_state` and asserts six things, the core one being that **every
chunk_id in the sources is a member of the embedded dataset**
(`esg_scraper/data/embeddings/chunk_ids.json`). That subset assertion is the
direct evidence that the answer is grounded in the indexed corpus rather than
model memory or the single-document path.

Writes outputs/streamlit_provenance_report.md.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Inject ALBERT_API_KEY from esg_scraper/.env BEFORE importing AppTest, so the
# UI's sidebar text_input default + the load_model_catalog network call have it.
EMBED_KEY_PATH = REPO.parent / "esg_scraper" / ".env"
if EMBED_KEY_PATH.exists():
    for raw in EMBED_KEY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from streamlit.testing.v1 import AppTest  # noqa: E402

EMBEDDED_CHUNK_IDS_PATH = REPO.parent / "esg_scraper" / "data" / "embeddings" / "chunk_ids.json"
REPORT_PATH = REPO / "outputs" / "streamlit_provenance_report.md"
TARGET_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
BAD_LABEL_RE = re.compile(r"\[(?:chunk-\d+|t\d+|n\d+)\]", re.IGNORECASE)
# gpt-oss-120b cycles between three citation formats under this structured
# prompt: `[5]` (ASCII brackets), `【5】` (lenticular brackets, no suffix), and
# `【5†L1-L4】` (lenticular with web-search line range). All three encode the
# same citation_map index — match the leading integer of any form.
BRACKET_NUM_RE = re.compile(r"\[(\d{1,3})\]|【(\d{1,3})(?:†[^】]*)?】")


def _ss_get(ss, key, default=None):
    """AppTest.session_state proxy routes attribute access to keys, so .get
    doesn't work directly — read via `in` + bracket access."""
    try:
        if key in ss:
            return ss[key]
    except Exception:
        pass
    return default


def _find_execute(at: AppTest):
    for accessor in ("form_submit_button", "button"):
        widgets = getattr(at, accessor, None)
        if not widgets:
            continue
        for w in widgets:
            label = getattr(w, "label", "") or ""
            if label.strip().lower() == "execute":
                return w
    return None


def _pin_chat_model(at: AppTest, target_id: str) -> str:
    """Pin the sidebar model selectbox to target_id if available; return whatever ends up selected."""
    for sb in at.sidebar.selectbox:
        try:
            options = list(sb.options) if hasattr(sb, "options") else []
        except Exception:
            options = []
        # ChatModel objects expose .model_id and .label
        for opt in options:
            mid = getattr(opt, "model_id", None)
            if mid == target_id:
                sb.set_value(opt)
                return mid
    current = None
    for sb in at.sidebar.selectbox:
        val = getattr(sb, "value", None)
        mid = getattr(val, "model_id", None)
        if mid:
            current = mid
            break
    return current or "(unknown)"


def _set_text_input(at: AppTest, label_substr: str, value: str) -> bool:
    """Set the value of a sidebar text_input whose label contains the substring."""
    for ti in at.sidebar.text_input:
        label = (getattr(ti, "label", "") or "").lower()
        if label_substr.lower() in label:
            ti.set_value(value)
            return True
    return False


def _set_selectbox_value(at: AppTest, label_substr: str, value: str) -> bool:
    """Set the value of a sidebar selectbox whose label contains the substring."""
    for sb in at.sidebar.selectbox:
        label = (getattr(sb, "label", "") or "").lower()
        if label_substr.lower() in label:
            sb.set_value(value)
            return True
    return False


def _find_main_prompt(at: AppTest):
    for ta in at.text_area:
        if (getattr(ta, "label", "") or "") == "Prompt":
            return ta
    return at.text_area[-1] if at.text_area else None


def _run_scenario(name: str, prompt: str, filters: dict) -> dict:
    """Boot a fresh AppTest, set filters/prompt, click Execute, return state snapshot."""
    print(f"\n=== Scenario {name}: {prompt!r}  filters={filters}")
    at = AppTest.from_file(str(REPO / "app" / "streamlit_ui.py"), default_timeout=600)
    at.run()
    if at.exception:
        raise SystemExit(f"[{name}] initial render exception: {at.exception}")

    used_model = _pin_chat_model(at, TARGET_MODEL_ID)
    print(f"  pinned chat model: {used_model}")

    # Sidebar caption inspection (index loaded summary)
    sidebar_caption_text = " ".join(
        (getattr(c, "value", "") or "") for c in at.sidebar.caption
    )

    # Apply filters
    if filters.get("company"):
        ok = _set_text_input(at, "filter company", filters["company"])
        print(f"  set company filter: {ok}")
    if filters.get("speaker_role"):
        ok = _set_selectbox_value(at, "filter speaker role", filters["speaker_role"])
        print(f"  set speaker_role filter: {ok}")

    # Type the prompt
    pta = _find_main_prompt(at)
    if pta is None:
        raise SystemExit(f"[{name}] could not find main prompt text_area")
    pta.set_value(prompt)

    # Click Execute
    exe = _find_execute(at)
    if exe is None:
        raise SystemExit(f"[{name}] could not find Execute button")
    exe.click()
    t0 = time.monotonic()
    at.run()
    elapsed = time.monotonic() - t0

    if at.exception:
        raise SystemExit(f"[{name}] post-submit exception: {at.exception}")

    ss = at.session_state
    snapshot = {
        "name": name,
        "prompt": prompt,
        "filters": filters,
        "used_model": used_model,
        "sidebar_caption": sidebar_caption_text,
        "last_output": (_ss_get(ss, "last_output", "") or ""),
        "last_corpus_sources": _ss_get(ss, "last_corpus_sources", None),
        "last_corpus_citation_map": _ss_get(ss, "last_corpus_citation_map", None),
        "last_prompt_with_context": (_ss_get(ss, "last_prompt_with_context", "") or ""),
        "submit_elapsed_s": elapsed,
        "success_messages": [s.value for s in at.success] if at.success else [],
    }
    print(f"  elapsed: {elapsed:.1f}s  answer_chars: {len(snapshot['last_output'])}  "
          f"sources: {len(snapshot['last_corpus_sources']) if isinstance(snapshot['last_corpus_sources'], list) else 'NONE'}")
    return snapshot


def _check_provenance(snapshot: dict, embedded_ids: set) -> tuple[bool, dict]:
    sources = snapshot["last_corpus_sources"]
    if not isinstance(sources, list) or not sources:
        return False, {"reason": "no sources in session_state"}
    src_ids = {s["chunk_id"] for s in sources}
    missing = src_ids - embedded_ids
    return (len(missing) == 0 and len(src_ids) > 0), {
        "n_sources": len(sources),
        "n_distinct_chunk_ids": len(src_ids),
        "subset_holds": len(missing) == 0,
        "missing_ids": sorted(missing)[:5],
    }


def _check_citation_resolution(snapshot: dict, embedded_ids: set) -> tuple[bool, dict]:
    answer = snapshot["last_output"] or ""
    cit_map = snapshot["last_corpus_citation_map"]
    if not isinstance(cit_map, dict):
        cit_map = {}
    # BRACKET_NUM_RE has two alternations; each match yields a 2-tuple where one
    # group is the captured number and the other is empty. Take whichever is non-empty.
    emitted_numbers = sorted({
        int(g) for matchpair in BRACKET_NUM_RE.findall(answer) for g in matchpair if g
    })
    bad_labels = BAD_LABEL_RE.findall(answer)
    unresolved = [n for n in emitted_numbers if n not in cit_map]
    resolved_chunk_ids = [cit_map[n] for n in emitted_numbers if n in cit_map]
    resolved_in_corpus = [cid for cid in resolved_chunk_ids if cid in embedded_ids]
    ok = (
        len(emitted_numbers) > 0
        and not bad_labels
        and not unresolved
        and len(resolved_in_corpus) == len(resolved_chunk_ids)
    )
    return ok, {
        "emitted_numbers": emitted_numbers,
        "bad_suffix_labels": bad_labels,
        "unresolved_numbers": unresolved,
        "resolved_chunk_ids": resolved_chunk_ids,
        "all_resolved_in_embedded_set": len(resolved_in_corpus) == len(resolved_chunk_ids),
    }


def _check_filter_honored(snapshot: dict, expected: dict) -> tuple[bool, dict]:
    sources = snapshot["last_corpus_sources"] or []
    violations = []
    for s in sources:
        for k, v in expected.items():
            actual = (s.get(k) or "").strip().lower()
            if actual != v.strip().lower():
                violations.append({"chunk_id": s.get("chunk_id"), "field": k, "actual": actual, "expected": v})
    return len(violations) == 0 and len(sources) > 0, {
        "n_sources": len(sources),
        "violations": violations[:5],
    }


def _check_attribution_regression(snapshot: dict) -> tuple[bool, dict]:
    answer_lower = (snapshot["last_output"] or "").lower()
    forbidden = ["ceo jean-pierre sbraire", "ceo sbraire"]
    hits = [phrase for phrase in forbidden if phrase in answer_lower]
    return len(hits) == 0, {"forbidden_phrases_present": hits}


def _check_single_doc_mode_intact() -> tuple[bool, dict]:
    """Switch the mode radio to 'Single document' and confirm clean rerun."""
    at = AppTest.from_file(str(REPO / "app" / "streamlit_ui.py"), default_timeout=300)
    at.run()
    if at.exception:
        return False, {"initial_render_exception": str(at.exception)}
    # mode radio is the first sidebar radio
    try:
        at.sidebar.radio[0].set_value("Single document")
        at.run()
    except Exception as e:
        return False, {"switch_exception": f"{type(e).__name__}: {e}"}
    if at.exception:
        return False, {"post_switch_exception": str(at.exception)}
    return True, {"note": "mode radio switched to Single document, app rerun clean"}


def main() -> int:
    if not os.environ.get("ALBERT_API_KEY"):
        raise SystemExit("ALBERT_API_KEY missing — E2E test requires it")

    embedded_ids = set(json.loads(EMBEDDED_CHUNK_IDS_PATH.read_text(encoding="utf-8")))
    print(f"[setup] embedded id set size: {len(embedded_ids)}")

    snap_a = _run_scenario(
        name="a_no_filter",
        prompt="What are Engie's Scope 1 emissions?",
        filters={},
    )
    snap_b = _run_scenario(
        name="b_filtered_te_executive",
        prompt="most recent vision from TotalEnergies CEO on ESG",
        filters={"company": "TotalEnergies", "speaker_role": "executive"},
    )

    findings: list[tuple[str, bool, str]] = []

    # Check 1: mode default + index summary caption
    # (corpus mode is the default; the caption should contain '146920 chunks' and 'BAAI/bge-m3')
    cap_text = snap_a["sidebar_caption"]
    mode_ok = ("146920 chunks" in cap_text and "BAAI/bge-m3" in cap_text)
    findings.append((
        "1_mode_and_index_loaded", mode_ok,
        f"sidebar_caption={cap_text[:200]!r}",
    ))

    # Check 2: PROVENANCE — every source chunk_id ⊆ embedded set, for both scenarios
    ok_a, det_a = _check_provenance(snap_a, embedded_ids)
    ok_b, det_b = _check_provenance(snap_b, embedded_ids)
    findings.append((
        "2a_provenance_no_filter", ok_a,
        f"n_sources={det_a.get('n_sources')} subset_holds={det_a.get('subset_holds')} "
        f"missing={det_a.get('missing_ids')}",
    ))
    findings.append((
        "2b_provenance_filtered", ok_b,
        f"n_sources={det_b.get('n_sources')} subset_holds={det_b.get('subset_holds')} "
        f"missing={det_b.get('missing_ids')}",
    ))

    # Check 3: citation resolution + no bad suffix labels
    ok_ca, det_ca = _check_citation_resolution(snap_a, embedded_ids)
    ok_cb, det_cb = _check_citation_resolution(snap_b, embedded_ids)
    findings.append((
        "3a_citation_resolution_no_filter", ok_ca,
        f"emitted={det_ca['emitted_numbers']} bad_labels={det_ca['bad_suffix_labels']} "
        f"unresolved={det_ca['unresolved_numbers']} in_corpus={det_ca['all_resolved_in_embedded_set']}",
    ))
    findings.append((
        "3b_citation_resolution_filtered", ok_cb,
        f"emitted={det_cb['emitted_numbers']} bad_labels={det_cb['bad_suffix_labels']} "
        f"unresolved={det_cb['unresolved_numbers']} in_corpus={det_cb['all_resolved_in_embedded_set']}",
    ))

    # Check 4: filter honored on scenario b
    ok_f, det_f = _check_filter_honored(snap_b, {"company": "TotalEnergies", "speaker_role": "executive"})
    findings.append((
        "4_filter_honored_scenario_b", ok_f,
        f"n_sources={det_f['n_sources']} violations={det_f['violations']}",
    ))

    # Check 5: attribution regression — no fabricated CEO title for Sbraire
    ok_attr, det_attr = _check_attribution_regression(snap_b)
    findings.append((
        "5_attribution_regression_scenario_b", ok_attr,
        f"forbidden_present={det_attr['forbidden_phrases_present']}",
    ))

    # Check 6: single-doc mode intact
    ok_sd, det_sd = _check_single_doc_mode_intact()
    findings.append((
        "6_single_document_mode_intact", ok_sd,
        json.dumps(det_sd, ensure_ascii=False),
    ))

    all_pass = all(p for _, p, _ in findings)

    # --- Write report ---
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append("# Streamlit provenance + attribution-guardrail report")
    L.append("")
    L.append(f"- streamlit version: {__import__('streamlit').__version__}")
    L.append(f"- embedded id set: {len(embedded_ids)} ids (from `esg_scraper/data/embeddings/chunk_ids.json`)")
    L.append(f"- pinned chat model target: `{TARGET_MODEL_ID}`")
    L.append(f"- scenario a — used model: `{snap_a['used_model']}`, elapsed {snap_a['submit_elapsed_s']:.1f}s")
    L.append(f"- scenario b — used model: `{snap_b['used_model']}`, elapsed {snap_b['submit_elapsed_s']:.1f}s")
    L.append("")
    L.append("## Verification")
    L.append(f"- Overall: **{'PASS' if all_pass else 'FAIL'}**")
    for name, p, d in findings:
        L.append(f"  - `{name}`: {'PASS' if p else 'FAIL'} — {d}")
    L.append("")

    for snap, det_prov, det_cit in (
        (snap_a, det_a, det_ca),
        (snap_b, det_b, det_cb),
    ):
        L.append(f"## Scenario `{snap['name']}`")
        L.append("")
        L.append(f"- prompt: `{snap['prompt']!r}`")
        L.append(f"- filters: `{snap['filters']}`")
        L.append(f"- last_prompt_with_context: `{snap['last_prompt_with_context']}`")
        L.append(f"- success message(s): `{snap['success_messages']}`")
        L.append(f"- sources count: {det_prov.get('n_sources')}; "
                 f"distinct chunk_ids: {det_prov.get('n_distinct_chunk_ids')}; "
                 f"provenance subset holds: {det_prov.get('subset_holds')}")
        L.append("")
        L.append("### Citation map (n → chunk_id + source meta)")
        cmap = snap.get("last_corpus_citation_map") or {}
        sources = snap.get("last_corpus_sources") or []
        for idx, src in enumerate(sources, start=1):
            cid = cmap.get(idx) if isinstance(cmap, dict) else None
            cid = cid or src.get("chunk_id", "")
            L.append(
                f"- **[{idx}]** `{cid}` — "
                f"{src.get('company') or '?'} · {src.get('doc_type') or src.get('section_title') or '?'} · "
                f"FY{src.get('report_year') or '?'} · speaker `{src.get('speaker_role') or '—'}` · "
                f"`{src.get('source_file', '?')}` p.{src.get('page_start', '?')} "
                f"(score {src.get('score', 0):.4f})"
            )
        L.append("")
        L.append("### Answer (verbatim)")
        L.append("")
        L.append("```")
        L.append(snap["last_output"] or "(empty)")
        L.append("```")
        L.append("")

    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n[verify] {'PASS' if all_pass else 'FAIL'}")
    for name, p, d in findings:
        print(f"  - {name}: {'PASS' if p else 'FAIL'} — {d[:200]}")
    print(f"\n[report] wrote {REPORT_PATH}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
