"""READ-ONLY diagnostic for table_fact chunk quality.

Streams the canonical `esg_scraper/data/chunks/chunks.jsonl`, flags degenerate
`chunk_kind == "table_fact"` rows by four conservative predicates, computes
unit availability, and writes three outputs in `outputs/`:
  - `tablefact_quality_report.md`  — human report with counts/examples
  - `degenerate_table_facts.json`  — the chunk_id drop-list for Task 2
  - `table_fact_units.json`        — {chunk_id: raw_unit} for non-empty units

NEVER modifies the canonical chunks.jsonl. The 10% sanity gate flips the
recommendation between "drop by default" and "systemic upstream fix".
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHUNKS_PATH = REPO.parent / "esg_scraper" / "data" / "chunks" / "chunks.jsonl"
OUT_DIR = REPO / "outputs"
REPORT_PATH = OUT_DIR / "tablefact_quality_report.md"
DEGEN_PATH = OUT_DIR / "degenerate_table_facts.json"
UNITS_PATH = OUT_DIR / "table_fact_units.json"

PLACEHOLDER_RE = re.compile(r"^\(?\d{1,3}\)?$")
DIGIT_RE = re.compile(r"\d")
THOUSANDS_RE = re.compile(r"\d[, \s]\d")
DECIMAL_RE = re.compile(r"\d[. ]\d")
UNIT_HINT_RE = re.compile(r"\d\s*[a-zA-Z%€$£°]")


def is_placeholder_value(raw_value) -> bool:
    """Detect bare/parenthesised ≤3-digit footnote markers like '(1)' or '31'.

    Conservative — bail out on any hint of a real figure (thousands separator,
    decimal, unit suffix, or more than 3 digits)."""
    if raw_value is None:
        return False
    v = str(raw_value).strip()
    if not v:
        return False
    if THOUSANDS_RE.search(v) or DECIMAL_RE.search(v) or UNIT_HINT_RE.search(v):
        return False
    if len(re.findall(r"\d", v)) > 3:
        return False
    return bool(PLACEHOLDER_RE.fullmatch(v))


def classify(obj: dict) -> set:
    raw_label = obj.get("raw_label") or ""
    raw_value = obj.get("raw_value") or ""
    cats: set = set()

    if not str(raw_value).strip():
        cats.add("empty_value")

    if is_placeholder_value(raw_value):
        cats.add("placeholder_value")

    v_strip = str(raw_value).strip()
    l_strip = str(raw_label).strip()
    if v_strip and l_strip and v_strip.lower() == l_strip.lower():
        cats.add("label_equals_value")

    # value_in_label_only: non-numeric value, a substring of label, not equal
    if (
        v_strip
        and not DIGIT_RE.search(v_strip)
        and v_strip.lower() != l_strip.lower()
        and v_strip.lower() in l_strip.lower()
    ):
        cats.add("value_in_label_only")

    return cats


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"missing canonical source: {CHUNKS_PATH}")

    total_chunks = 0
    n_table_fact = 0
    n_with_unit = 0
    counts: Counter = Counter()
    examples: dict = {
        "empty_value": [],
        "placeholder_value": [],
        "label_equals_value": [],
        "value_in_label_only": [],
    }
    degen_ids: set = set()
    units_map: dict = {}

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            total_chunks += 1
            if obj.get("chunk_kind") != "table_fact":
                continue
            n_table_fact += 1

            unit = obj.get("raw_unit")
            unit_str = str(unit).strip() if unit is not None else ""
            if unit_str:
                n_with_unit += 1
                units_map[obj["chunk_id"]] = unit_str

            cats = classify(obj)
            for c in cats:
                counts[c] += 1
                if len(examples[c]) < 10:
                    txt = (obj.get("text") or "")
                    if len(txt) > 220:
                        txt = txt[:220] + "..."
                    examples[c].append({
                        "chunk_id": obj["chunk_id"],
                        "raw_label": obj.get("raw_label"),
                        "raw_value": obj.get("raw_value"),
                        "text": txt,
                    })
            if cats:
                degen_ids.add(obj["chunk_id"])

    n_degen = len(degen_ids)
    pct_of_tf = (n_degen / n_table_fact * 100) if n_table_fact else 0.0
    pct_of_corpus = (n_degen / total_chunks * 100) if total_chunks else 0.0
    unit_availability = (n_with_unit / n_table_fact * 100) if n_table_fact else 0.0
    sanity_gate_tripped = pct_of_tf > 10.0

    _atomic_write(
        DEGEN_PATH,
        json.dumps({
            "n_degenerate": n_degen,
            "n_table_fact": n_table_fact,
            "pct_of_table_fact": round(pct_of_tf, 4),
            "sanity_gate_tripped": sanity_gate_tripped,
            "chunk_ids": sorted(degen_ids),
        }, ensure_ascii=False),
    )
    _atomic_write(UNITS_PATH, json.dumps(units_map, ensure_ascii=False))

    L: list = []
    L.append("# table_fact quality diagnostic")
    L.append("")
    L.append(f"- canonical source: `{CHUNKS_PATH}`")
    L.append(f"- corpus total chunks: **{total_chunks}**")
    L.append(f"- table_fact chunks: **{n_table_fact}** "
             f"({n_table_fact/total_chunks*100:.2f}% of corpus)" if total_chunks else "")
    L.append(f"- table_fact with non-empty `raw_unit`: **{n_with_unit}** "
             f"({unit_availability:.2f}% of table_fact)")
    L.append("")
    L.append("## Degeneracy counts (a chunk may hit multiple categories)")
    L.append("")
    L.append("| category | count | % of table_fact | % of corpus |")
    L.append("|---|---|---|---|")
    for cat in ("empty_value", "placeholder_value", "label_equals_value", "value_in_label_only"):
        c = counts[cat]
        L.append(
            f"| `{cat}` | {c} | "
            f"{(c/n_table_fact*100) if n_table_fact else 0:.2f}% | "
            f"{(c/total_chunks*100) if total_chunks else 0:.2f}% |"
        )
    L.append(
        f"| **distinct degenerate chunk_ids (union)** | **{n_degen}** | "
        f"**{pct_of_tf:.2f}%** | **{pct_of_corpus:.2f}%** |"
    )
    L.append("")

    if sanity_gate_tripped:
        L.append("## ⚠ SANITY GATE TRIPPED")
        L.append("")
        L.append(
            f"Degenerate table_fact rate **{pct_of_tf:.2f}%** exceeds the 10% threshold. "
            "Treat this as a **systemic extraction bug in the upstream pipeline**, "
            "not a mass-drop case. Investigate the table parser before mass-dropping "
            f"{n_degen} chunks from the index. If you still want to drop, Task 2 will "
            "require an explicit `--drop-degenerate` opt-in."
        )
    else:
        L.append("## Sanity gate: OK")
        L.append("")
        L.append(
            f"Degenerate rate **{pct_of_tf:.2f}%** is below 10%. "
            "Task 2 (`build_index_from_embeddings.py`) will drop these by default."
        )
    L.append("")

    for cat in ("empty_value", "placeholder_value", "label_equals_value", "value_in_label_only"):
        L.append(f"## Examples — `{cat}`  (showing up to 10)")
        L.append("")
        if not examples[cat]:
            L.append("- _(none)_")
        for ex in examples[cat]:
            L.append(
                f"- `{ex['chunk_id']}`  "
                f"raw_label={ex['raw_label']!r}  raw_value={ex['raw_value']!r}"
            )
            L.append(f"  text: {ex['text']!r}")
        L.append("")

    _atomic_write(REPORT_PATH, "\n".join(L) + "\n")

    print(f"[diag] corpus_total={total_chunks}  table_fact={n_table_fact}")
    print(f"[diag] degenerate distinct: {n_degen} "
          f"({pct_of_tf:.2f}% of table_fact, {pct_of_corpus:.2f}% of corpus)")
    print(f"[diag] unit-availability: {n_with_unit}/{n_table_fact} ({unit_availability:.2f}%)")
    print(f"[diag] per-category counts: {dict(counts)}")
    print(f"[diag] sanity_gate_tripped: {sanity_gate_tripped}")
    print(f"[diag] wrote {REPORT_PATH}")
    print(f"[diag] wrote {DEGEN_PATH}")
    print(f"[diag] wrote {UNITS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
