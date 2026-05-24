"""Convert the esg_scraper embeddings dir into the cloned repo's rag_index layout.

Reads the pipeline outputs (`vectors.npy`, `chunk_ids.json`, `metadata.jsonl`) and
emits the four artifacts the repo's `app.rag` expects: `chunks.json` (with EXACTLY
the 13 ChunkRecord fields), `vectors.npy`, `index.faiss` (`IndexFlatIP`), and
`manifest.json`. No API calls; no re-embedding.

Run from <REPO> root:
    python scripts/build_index_from_embeddings.py \
        --embeddings-dir "C:\\...\\esg_scraper\\data\\embeddings"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from app.rag import DEFAULT_INDEX_DIR

DEFAULT_EMBEDDINGS_DIR = Path(
    r"C:\Users\azalizniak\Downloads\ESG_RAG\Script\esg_scraper\data\embeddings"
)
# Diagnostic outputs produced by scripts/diagnose_tablefact_quality.py
DEFAULT_DEGEN_PATH = Path(__file__).resolve().parents[1] / "outputs" / "degenerate_table_facts.json"
DEFAULT_UNITS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "table_fact_units.json"

CHUNKRECORD_FIELDS = (
    "chunk_id", "source_file", "source_path", "page_start", "page_end",
    "token_count", "text", "contextual_summary", "esg_pillar",
    "section_title", "report_year", "contains_table", "contains_targets",
    "company", "speaker_role", "chunk_kind", "doc_type",
)

TOKEN_RE = re.compile(r"\S+")
_PILLAR_PRIORITY = ("environmental", "social", "governance")


def _derive_esg_pillar(esrs_topics) -> str:
    counts: Counter = Counter()
    for t in esrs_topics or []:
        if not t:
            continue
        head = t[0].upper()
        if head == "E":
            counts["environmental"] += 1
        elif head == "S":
            counts["social"] += 1
        elif head == "G":
            counts["governance"] += 1
    if not counts:
        return ""
    return sorted(
        counts.items(),
        key=lambda kv: (-kv[1], _PILLAR_PRIORITY.index(kv[0])),
    )[0][0]


def _atomic_replace(tmp: Path, dst: Path) -> None:
    os.replace(tmp, dst)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    tmp.write_text(content, encoding="utf-8")
    _atomic_replace(tmp, path)


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    np.save(tmp, arr)
    _atomic_replace(tmp, path)


def _atomic_write_faiss(path: Path, index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    faiss.write_index(index, str(tmp))
    _atomic_replace(tmp, path)


def _stream_write_chunks_json(path: Path, records) -> None:
    """Write a JSON array of records without materialising the full string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    with tmp.open("w", encoding="utf-8") as f:
        f.write("[")
        for i, rec in enumerate(records):
            if i > 0:
                f.write(",")
            f.write(json.dumps(rec, ensure_ascii=False))
        f.write("]")
    _atomic_replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing populated index_dir")
    p.add_argument("--drop-degenerate", action="store_true",
                   help="drop chunks listed in outputs/degenerate_table_facts.json "
                        "(auto-on if the diagnostic's sanity gate is NOT tripped; "
                        "opt-in required if the gate IS tripped, i.e. >10 pct)")
    p.add_argument("--no-drop-degenerate", action="store_true",
                   help="explicitly do NOT drop any chunks even if the drop-list exists")
    p.add_argument("--enrich-units", action="store_true",
                   help="for kept table_fact chunks whose chunk_id has a known raw_unit, "
                        "append the unit to the text written into chunks.json. NOTE: this "
                        "creates a deliberate text/vector divergence: the embedding still "
                        "reflects the original unit-less text (no re-embedding).")
    p.add_argument("--degenerate-list", type=Path, default=DEFAULT_DEGEN_PATH,
                   help="path to the drop-list JSON produced by diagnose_tablefact_quality.py")
    p.add_argument("--units-map", type=Path, default=DEFAULT_UNITS_PATH,
                   help="path to the {chunk_id: raw_unit} JSON produced by diagnose_tablefact_quality.py")
    return p.parse_args()


def _load_drop_list(path: Path) -> tuple[set, bool, float]:
    """Return (degen_id_set, sanity_gate_tripped, pct_of_table_fact). If missing, empty set."""
    if not path.exists():
        return set(), False, 0.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        set(payload.get("chunk_ids", [])),
        bool(payload.get("sanity_gate_tripped", False)),
        float(payload.get("pct_of_table_fact", 0.0)),
    )


def _load_units_map(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build(
    embeddings_dir: Path,
    index_dir: Path,
    force: bool,
    *,
    drop_degenerate_flag: bool = False,
    no_drop_degenerate_flag: bool = False,
    enrich_units_flag: bool = False,
    degenerate_list_path: Path = DEFAULT_DEGEN_PATH,
    units_map_path: Path = DEFAULT_UNITS_PATH,
) -> None:
    vectors_in = embeddings_dir / "vectors.npy"
    chunk_ids_in = embeddings_dir / "chunk_ids.json"
    metadata_in = embeddings_dir / "metadata.jsonl"
    for p in (vectors_in, chunk_ids_in, metadata_in):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    has_core = any(
        (index_dir / f).exists()
        for f in ("chunks.json", "vectors.npy", "index.faiss", "manifest.json")
    )
    if has_core and not force:
        raise SystemExit(
            f"{index_dir} already contains an index; pass --force to overwrite"
        )

    print(f"[load] {vectors_in}")
    vectors = np.load(vectors_in)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 1024:
        raise SystemExit(f"unexpected vectors shape: {vectors.shape}")
    N = vectors.shape[0]

    print(f"[load] {chunk_ids_in}")
    chunk_ids = json.loads(chunk_ids_in.read_text(encoding="utf-8"))
    if len(chunk_ids) != N:
        raise SystemExit(f"chunk_ids len {len(chunk_ids)} != vectors rows {N}")

    print(f"[load] {metadata_in}")
    metadata = []
    with metadata_in.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            metadata.append(json.loads(raw))
    if len(metadata) != N:
        raise SystemExit(f"metadata len {len(metadata)} != vectors rows {N}")

    # Hard alignment guarantee — row i of vectors corresponds to chunk_ids[i]
    # which must equal metadata[i].chunk_id. A drift here would silently scramble
    # retrieval, so fail loudly.
    for i in range(N):
        if metadata[i]["chunk_id"] != chunk_ids[i]:
            raise SystemExit(
                f"row alignment failure at i={i}: "
                f"metadata={metadata[i]['chunk_id']!r} chunk_ids={chunk_ids[i]!r}"
            )

    # ----- Drop-list + units map (Task 2 mitigations) ----------------------
    degen_ids, sanity_gate_tripped, pct_of_tf = _load_drop_list(degenerate_list_path)
    units_map = _load_units_map(units_map_path) if enrich_units_flag else {}

    if no_drop_degenerate_flag:
        drop_active = False
        drop_reason = "--no-drop-degenerate passed"
    elif not degen_ids:
        drop_active = False
        drop_reason = f"no drop-list at {degenerate_list_path}"
    elif sanity_gate_tripped:
        # >10% degeneracy → require explicit opt-in
        drop_active = bool(drop_degenerate_flag)
        drop_reason = (
            f"sanity gate TRIPPED (pct={pct_of_tf:.2f}%); "
            + ("--drop-degenerate passed, dropping" if drop_active
               else "no --drop-degenerate, keeping all")
        )
    else:
        # ≤10% → drop by default
        drop_active = True
        drop_reason = f"sanity gate OK (pct={pct_of_tf:.2f}%); dropping by default"
    print(f"[drop] {drop_reason}  (drop-list size: {len(degen_ids)})")
    if enrich_units_flag:
        print(f"[enrich-units] loaded {len(units_map)} unit mappings; "
              "text/vector divergence — embeddings unchanged")

    print(f"[build] converting {N} rows to ChunkRecord format")
    chunks_out: list[dict] = []
    sources_seen: set[str] = set()
    keep_mask = np.ones(N, dtype=bool)
    n_dropped = 0
    n_enriched = 0
    for i in range(N):
        m = metadata[i]
        cid = chunk_ids[i]

        if drop_active and cid in degen_ids:
            keep_mask[i] = False
            n_dropped += 1
            continue

        text = m.get("text") or ""
        orig_src = m.get("source_file") or ""
        company = m.get("company") or ""
        doc_type = m.get("doc_type") or ""
        fy = m.get("fiscal_year")
        page_raw = m.get("page")
        try:
            page_int = int(page_raw) if page_raw is not None else 0
        except (TypeError, ValueError):
            page_int = 0
        topics = m.get("esrs_topic") or []
        speaker = m.get("speaker_role") or ""
        authority = m.get("source_authority")
        kind = m.get("chunk_kind")

        # Opt-in unit enrichment: append the raw_unit to text if missing.
        # The vector remains the embedding of the ORIGINAL unit-less text —
        # this is a deliberate text/vector divergence flagged in the manifest.
        if enrich_units_flag and cid in units_map:
            unit = (units_map[cid] or "").strip()
            if unit and unit.lower() not in text.lower():
                text = f"{text.rstrip(' .')} {unit}".strip() if text else unit
                n_enriched += 1

        # company prefix keeps the year substring intact so repo's filter_year
        # (which substring-matches against source_file) still works
        prefixed_source_file = (
            f"{company} · {orig_src}" if company and orig_src else (orig_src or "")
        )
        contextual_summary = (
            f"{company} | {doc_type} | FY{fy} | tier{authority} | "
            f"{','.join(t for t in topics if t)} | {speaker or ''}"
        )

        rec = {
            "chunk_id": cid,
            "source_file": prefixed_source_file,
            "source_path": orig_src,
            "page_start": page_int,
            "page_end": page_int,
            "token_count": len(TOKEN_RE.findall(text)),
            "text": text,
            "contextual_summary": contextual_summary,
            "esg_pillar": _derive_esg_pillar(topics),
            "section_title": doc_type,
            "report_year": str(fy) if fy is not None else "",
            "contains_table": kind == "table_fact",
            "contains_targets": False,
            # Direct fields (also reflected in contextual_summary for display)
            # so the repo's filter logic can do exact-match without parsing.
            "company": company or "",
            "speaker_role": speaker or "",
            "chunk_kind": kind or "",
            "doc_type": doc_type or "",
        }
        # safety: exact field-set match — _load_chunks does ChunkRecord(**item)
        # and extras would raise TypeError
        assert set(rec.keys()) == set(CHUNKRECORD_FIELDS), (
            f"unexpected fields: {set(rec.keys()) ^ set(CHUNKRECORD_FIELDS)}"
        )
        chunks_out.append(rec)
        if orig_src:
            sources_seen.add(orig_src)

    n_kept = int(keep_mask.sum())
    assert len(chunks_out) == n_kept, (
        f"chunks_out len {len(chunks_out)} != keep_mask sum {n_kept}"
    )
    if drop_active and n_dropped:
        vectors_out = vectors[keep_mask]
    else:
        vectors_out = vectors
    print(f"[build] N before={N}  kept={n_kept}  dropped={n_dropped}  "
          f"enriched={n_enriched}  (vectors shape: {vectors_out.shape})")

    print(f"[write] {index_dir / 'chunks.json'}")
    _stream_write_chunks_json(index_dir / "chunks.json", chunks_out)

    print(f"[write] {index_dir / 'vectors.npy'}")
    _atomic_write_npy(index_dir / "vectors.npy", vectors_out)

    print(f"[write] {index_dir / 'index.faiss'}")
    faiss_index = faiss.IndexFlatIP(vectors_out.shape[1])
    faiss_index.add(vectors_out)
    assert faiss_index.ntotal == n_kept
    _atomic_write_faiss(index_dir / "index.faiss", faiss_index)

    print(f"[write] {index_dir / 'manifest.json'}")
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pdfs": sorted(sources_seen),
        "chunk_count": n_kept,
        "original_chunk_count": N,
        "dropped_degenerate_count": n_dropped,
        "drop_list_source": str(degenerate_list_path) if drop_active else None,
        "enrich_units_active": bool(enrich_units_flag),
        "enriched_units_count": n_enriched,
        "text_vector_divergence_note": (
            "Embeddings reflect the original unit-less text; chunks.json text "
            "may include appended units (--enrich-units). Vectors were NOT recomputed."
        ) if enrich_units_flag and n_enriched else None,
        "target_tokens": None,
        "min_tokens": None,
        "max_tokens": None,
        "overlap_tokens": 0,
        "section_aware": False,
        "contextual_chunking": False,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": int(vectors_out.shape[1]),
        "vector_backend": "faiss",
    }
    _atomic_write_text(
        index_dir / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False),
    )

    print()
    print(f"[done] N={n_kept}  dim={vectors_out.shape[1]}  index_dir={index_dir}")
    print(f"       distinct source files in manifest.pdfs: {len(sources_seen)}")
    print(f"       before -> after:  {N} -> {n_kept}  (dropped {n_dropped}, enriched {n_enriched})")


def main() -> int:
    args = parse_args()
    build(
        args.embeddings_dir.resolve(),
        args.index_dir.resolve(),
        args.force,
        drop_degenerate_flag=args.drop_degenerate,
        no_drop_degenerate_flag=args.no_drop_degenerate,
        enrich_units_flag=args.enrich_units,
        degenerate_list_path=args.degenerate_list.resolve(),
        units_map_path=args.units_map.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
