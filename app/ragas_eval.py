"""RAGAS-style evaluation metrics using Albert API as LLM judge.

Implements: faithfulness, answer_relevancy, context_precision, context_recall,
hallucination detection, citation coverage, unsupported claim detection.

All metrics are computed via Albert API without external paid services.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.rag import (
    AlbertClient,
    DEFAULT_BASE_URL,
    answer_question,
    require_api_key,
    retrieve_chunks,
    retrieve_with_transform,
)
from app.rag_eval import (
    DEFAULT_EVAL_DATASET,
    _infer_company,
    _load_rows,
    _parse_expected_contexts,
    _score_retrieval,
    compute_eval_summary,
)
from app.utils import OUTPUT_DIR

DEFAULT_RAGAS_OUTPUT = OUTPUT_DIR / "ragas_eval_results.json"


def _call_judge(client: AlbertClient, model: str, prompt: str, temperature: float = 0.0) -> str:
    messages = [{"role": "user", "content": prompt}]
    return client.chat_completion(model, messages, temperature=temperature)


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(raw[start : end + 1])

    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected a JSON object", raw, 0)
    return parsed


def extract_cited_chunk_ids(answer: str) -> set[str]:
    normalized = (
        answer.replace("\u2011", "-")
        .replace("\u2010", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return set(re.findall(r"[\[【]\s*(chunk-\d+)\s*[\]】]", normalized))


def score_faithfulness(question: str, answer: str, context_chunks: list[dict[str, Any]], client: AlbertClient, model: str) -> dict[str, Any]:
    if not answer or not answer.strip():
        return {"score": 0.0, "verdicts": [], "reasoning": "No answer provided."}

    contexts_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in context_chunks
    )

    prompt = f"""You are evaluating the factual faithfulness of an ESG answer against provided context.

Question: {question}

Answer: {answer}

Context:
{contexts_text}

For each factual claim in the answer, determine if it is:
- "supported": directly stated in the context
- "contradicted": explicitly contradicted by the context
- "unsupported": not mentioned in the context at all

Output as JSON:
{{"verdicts": [{{"claim": "<claim text>", "verdict": "supported|contradicted|unsupported", "reason": "<brief reason>"}}], "overall_reasoning": "<summary>"}}

Only output valid JSON. No markdown."""

    try:
        raw = _call_judge(client, model, prompt)
        parsed = _parse_json_object(raw)
        verdicts = parsed.get("verdicts", [])
        supported = sum(1 for v in verdicts if v.get("verdict") == "supported")
        total = len(verdicts)
        score = round(supported / total, 4) if total > 0 else 0.0
        return {"score": score, "verdicts": verdicts, "reasoning": parsed.get("overall_reasoning", ""), "raw": raw}
    except (json.JSONDecodeError, Exception) as exc:
        return {"score": 0.0, "verdicts": [], "reasoning": f"Judge evaluation failed: {exc}", "raw": None}


def score_answer_relevancy(question: str, answer: str, client: AlbertClient, model: str) -> dict[str, Any]:
    if not answer or not answer.strip():
        return {"score": 0.0, "reasoning": "No answer provided."}

    prompt = f"""Evaluate how relevant this ESG answer is to the question.

Question: {question}

Answer: {answer}

Score relevance from 0.0 to 1.0 where:
- 1.0: Fully answers the question with specific, relevant information
- 0.7-0.9: Mostly relevant but partially incomplete or slightly off-topic
- 0.4-0.6: Somewhat relevant but misses key aspects
- 0.1-0.3: Mostly irrelevant
- 0.0: Completely irrelevant or non-responsive

Output as JSON:
{{"score": <float>, "reasoning": "<brief explanation>"}}

Only output valid JSON. No markdown."""

    try:
        raw = _call_judge(client, model, prompt)
        parsed = _parse_json_object(raw)
        return {"score": float(parsed.get("score", 0.0)), "reasoning": parsed.get("reasoning", ""), "raw": raw}
    except (json.JSONDecodeError, Exception) as exc:
        return {"score": 0.0, "reasoning": f"Judge evaluation failed: {exc}", "raw": None}


def score_context_precision(
    question: str, answer: str, retrieved_chunks: list[dict[str, Any]], client: AlbertClient, model: str
) -> dict[str, Any]:
    if not retrieved_chunks:
        return {"score": 0.0, "relevant_count": 0, "total": 0}

    chunk_summaries = "\n".join(
        f"[{c['chunk_id']}] (pages {c['page_start']}-{c['page_end']}): {c['text'][:200]}..."
        for c in retrieved_chunks
    )

    prompt = f"""Evaluate how precisely the retrieved ESG chunks address the question.

Question: {question}

Answer: {answer}

Retrieved chunks (in order):
{chunk_summaries}

For each chunk, classify as:
- "highly_relevant": Contains specific information directly answering the question
- "somewhat_relevant": Related topic but not directly addressing the question
- "not_relevant": Unrelated or tangential

Output as JSON:
{{"verdicts": [{{"chunk_id": "<id>", "verdict": "highly_relevant|somewhat_relevant|not_relevant", "reason": "<brief>"}}], "overall": "<summary>"}}

Only output valid JSON. No markdown."""

    try:
        raw = _call_judge(client, model, prompt)
        parsed = _parse_json_object(raw)
        verdicts = parsed.get("verdicts", [])
        relevant = sum(1 for v in verdicts if v.get("verdict") in ("highly_relevant", "somewhat_relevant"))
        highly = sum(1 for v in verdicts if v.get("verdict") == "highly_relevant")
        total = len(verdicts)
        precision = round(relevant / total, 4) if total > 0 else 0.0
        return {"score": precision, "highly_relevant": highly, "relevant_count": relevant, "total": total, "verdicts": verdicts, "raw": raw}
    except (json.JSONDecodeError, Exception) as exc:
        return {"score": 0.0, "relevant_count": 0, "total": len(retrieved_chunks), "verdicts": [], "reasoning": f"Failed: {exc}", "raw": None}


def score_context_recall(
    question: str, answer: str, retrieved_chunks: list[dict[str, Any]], expected_contexts: list[str], client: AlbertClient, model: str
) -> dict[str, Any]:
    if not expected_contexts:
        return {"score": 0.0, "reasoning": "No expected contexts provided."}

    retrieved_text = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in retrieved_chunks
    )
    expected_text = "\n\n".join(f"Expected #{i}: {ec}" for i, ec in enumerate(expected_contexts))

    prompt = f"""Evaluate how well the retrieved chunks cover the expected ESG context.

Question: {question}

Expected key information:
{expected_text}

Retrieved chunks:
{retrieved_text}

For each expected piece of information, determine if it is:
- "covered": The retrieved chunks contain this information
- "partially_covered": Some aspects are present but details are incomplete  
- "not_covered": The information is absent from retrieved chunks

Output as JSON:
{{"verdicts": [{{"expected_id": <index>, "verdict": "covered|partially_covered|not_covered", "reason": "<brief>"}}], "overall": "<summary>"}}

Only output valid JSON. No markdown."""

    try:
        raw = _call_judge(client, model, prompt)
        parsed = _parse_json_object(raw)
        verdicts = parsed.get("verdicts", [])
        covered = sum(1 for v in verdicts if v.get("verdict") in ("covered", "partially_covered"))
        fully = sum(1 for v in verdicts if v.get("verdict") == "covered")
        total = len(verdicts) or len(expected_contexts)
        recall = round(covered / total, 4) if total > 0 else 0.0
        return {"score": recall, "fully_covered": fully, "covered_count": covered, "total": total, "verdicts": verdicts, "raw": raw}
    except (json.JSONDecodeError, Exception) as exc:
        return {"score": 0.0, "covered_count": 0, "total": len(expected_contexts), "verdicts": [], "reasoning": f"Failed: {exc}", "raw": None}


def score_hallucination(question: str, answer: str, context_chunks: list[dict[str, Any]], client: AlbertClient, model: str) -> dict[str, Any]:
    if not answer or not answer.strip():
        return {"score": 0.0, "hallucinated_claims": [], "reasoning": "No answer provided."}

    contexts_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in context_chunks
    )

    prompt = f"""You are a hallucination detector for ESG answers. Identify claims in the answer that are NOT supported by the context.

Question: {question}

Answer: {answer}

Context:
{contexts_text}

List any claims that appear in the answer but CANNOT be verified from the context:
- Flag fabricated numbers/metrics
- Flag invented entity names
- Flag unsupported causal claims
- Flag hallucinated dates/years

Output as JSON:
{{"hallucinated_claims": [{{"claim": "<text>", "type": "fabricated_number|invented_entity|unsupported_causal|hallucinated_date", "reason": "<why it's unsupported>"}}], "hallucination_free": true|false, "overall": "<summary>"}}

If no hallucinations are found, set hallucination_free to true. Only output valid JSON."""

    try:
        raw = _call_judge(client, model, prompt)
        parsed = _parse_json_object(raw)
        claims = parsed.get("hallucinated_claims", [])
        is_free = parsed.get("hallucination_free", len(claims) == 0)
        score = 1.0 if is_free else max(0.0, 1.0 - (len(claims) * 0.25))
        return {"score": round(score, 4), "hallucinated_claims": claims, "hallucination_free": is_free, "raw": raw}
    except (json.JSONDecodeError, Exception) as exc:
        return {"score": 0.5, "hallucinated_claims": [], "reasoning": f"Failed: {exc}", "raw": None}


def score_citation_coverage(answer: str, cited_chunk_ids: set[str], context_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    available_ids = {c["chunk_id"] for c in context_chunks}
    valid_citations = cited_chunk_ids & available_ids
    invalid_citations = cited_chunk_ids - available_ids

    score = round(len(valid_citations) / len(cited_chunk_ids), 4) if cited_chunk_ids else 0.0

    return {
        "score": score,
        "cited_count": len(cited_chunk_ids),
        "valid_cited": len(valid_citations),
        "invalid_cited": len(invalid_citations),
        "invalid_ids": sorted(invalid_citations),
    }


def detect_unsupported_claims(
    question: str, answer: str, context_chunks: list[dict[str, Any]], client: AlbertClient, model: str
) -> dict[str, Any]:
    if not answer or not answer.strip():
        return {"score": 0.0, "claims": [], "reasoning": "No answer provided."}

    contexts_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in context_chunks
    )

    prompt = f"""Detect ESG claims in the answer that cannot be supported by the context.

Question: {question}

Answer: {answer}

Context:
{contexts_text}

Identify claims that:
1. Assert facts not present in the context
2. Make quantitative claims without source
3. Draw conclusions beyond what the context supports
4. Attribute actions/policies without evidence

Output as JSON:
{{"unsupported": [{{"claim": "<text>", "severity": "minor|moderate|major", "reason": "<why unsupported>"}}], "supported_count": <int>, "unsupported_count": <int>, "overall": "<summary>"}}

Only output valid JSON. No markdown."""

    try:
        raw = _call_judge(client, model, prompt)
        parsed = _parse_json_object(raw)
        unsupported = parsed.get("unsupported", [])
        total = parsed.get("supported_count", 0) + len(unsupported)
        score = round(1.0 - (len(unsupported) / max(total, 1)), 4)
        return {"score": score, "claims": unsupported, "unsupported_count": len(unsupported), "raw": raw}
    except (json.JSONDecodeError, Exception) as exc:
        return {"score": 0.5, "claims": [], "reasoning": f"Failed: {exc}", "raw": None}


def run_full_ragas_eval(
    *,
    index_dir: Path,
    dataset_path: Path = DEFAULT_EVAL_DATASET,
    company: str | None = None,
    top_k: int = 5,
    retrieval_mode: str = "dense",
    candidate_k: int = 15,
    eval_mode: str = "all",
    query_transform: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    text_model = client.get_text_generation_model()

    rows = _load_rows(dataset_path)
    selected_company = company or _infer_company(index_dir, rows)
    if not selected_company:
        raise RuntimeError("Could not infer which company to evaluate. Pass --company explicitly.")

    company_rows = [row for row in rows if row.get("company") == selected_company]
    if not company_rows:
        raise RuntimeError(f"No evaluation rows found for company '{selected_company}'.")

    results: list[dict[str, Any]] = []
    retrieval_results: list[dict[str, Any]] = []
    for row in company_rows:
        question = row["question"]
        ground_truth = row.get("ground_truth", "")
        expected_contexts = _parse_expected_contexts(row)

        diagnostics: list[dict[str, Any]] = []
        if query_transform and query_transform != "none":
            retrieved_chunks, _embedding_model, diagnostics = retrieve_with_transform(
                index_dir=index_dir,
                question=question,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                candidate_k=candidate_k,
                query_transform=query_transform,
                base_url=base_url,
            )
        else:
            retrieved_chunks, _embedding_model = retrieve_chunks(
                index_dir=index_dir,
                question=question,
                top_k=top_k,
                retrieval_architecture=retrieval_mode,
                search_breadth=candidate_k,
                base_url=base_url,
            )

        retrieval_scoring = _score_retrieval(expected_contexts, retrieved_chunks)
        retrieval_row = {
            "status": retrieval_scoring["status"],
            "best_match_score": retrieval_scoring["best_match_score"],
            "best_chunk_id": retrieval_scoring["best_chunk_id"],
            "top_score": retrieved_chunks[0]["score"] if retrieved_chunks else None,
        }
        retrieval_results.append(retrieval_row)

        answer = ""
        if eval_mode in ("answer", "ragas", "all"):
            answer, _ = answer_question(
                question=question,
                retrieved_chunks=retrieved_chunks,
                text_model=text_model,
                temperature=0.0,
                base_url=base_url,
            )

        result_entry: dict[str, Any] = {
            "company": selected_company,
            "question": question,
            "topic": row.get("topic", ""),
            "ground_truth": ground_truth,
            "answer": answer,
            "retrieved_count": len(retrieved_chunks),
            "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
            "retrieval_diagnostics": diagnostics,
        }

        cited_chunk_ids: set[str] = set()
        if answer:
            cited_chunk_ids = extract_cited_chunk_ids(answer)

        if eval_mode in ("retrieval", "ragas", "all"):
            result_entry["retrieval"] = retrieval_row

        if eval_mode in ("ragas", "all") and answer:
            faithfulness = score_faithfulness(question, answer, retrieved_chunks, client, text_model)
            relevancy = score_answer_relevancy(question, answer, client, text_model)
            precision = score_context_precision(question, answer, retrieved_chunks, client, text_model)
            recall = score_context_recall(question, answer, retrieved_chunks, expected_contexts, client, text_model)
            hallucination = score_hallucination(question, answer, retrieved_chunks, client, text_model)
            citation = score_citation_coverage(answer, cited_chunk_ids, retrieved_chunks)
            unsupported = detect_unsupported_claims(question, answer, retrieved_chunks, client, text_model)

            result_entry["ragas"] = {
                "faithfulness": faithfulness["score"],
                "answer_relevancy": relevancy["score"],
                "context_precision": precision["score"],
                "context_recall": recall["score"],
                "hallucination_free": hallucination["score"],
                "citation_coverage": citation["score"],
                "unsupported_claims_score": unsupported["score"],
            }
            result_entry["ragas_details"] = {
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
                "context_precision": precision,
                "context_recall": recall,
                "hallucination": hallucination,
                "citation": citation,
                "unsupported_claims": unsupported,
            }

        results.append(result_entry)

    aggregated: dict[str, Any] = {
        "company": selected_company,
        "question_count": len(results),
        "eval_mode": eval_mode,
        "text_model": text_model,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "query_transform": query_transform or "none",
    }

    if eval_mode in ("retrieval", "ragas", "all"):
        aggregated["retrieval_summary"] = compute_eval_summary(retrieval_results)

    if eval_mode in ("ragas", "all") and results:
        ragas_keys = [
            "faithfulness", "answer_relevancy", "context_precision", "context_recall",
            "hallucination_free", "citation_coverage", "unsupported_claims_score",
        ]
        ragas_avg: dict[str, float] = {}
        for key in ragas_keys:
            scores = [r["ragas"][key] for r in results if "ragas" in r and key in r.get("ragas", {})]
            ragas_avg[key] = round(sum(scores) / len(scores), 4) if scores else 0.0
        aggregated["ragas_summary"] = ragas_avg

    aggregated["results"] = results
    return aggregated


def run_ragas_eval(args: Any) -> int:
    payload = run_full_ragas_eval(
        index_dir=args.index_dir,
        dataset_path=args.dataset,
        company=args.company,
        top_k=args.top_k,
        retrieval_mode=getattr(args, "retrieval_mode", "dense"),
        candidate_k=getattr(args, "candidate_k", 15),
        eval_mode=args.eval_mode,
        query_transform=getattr(args, "query_transform", None),
        base_url=args.base_url,
    )

    output_path = getattr(args, "output", DEFAULT_RAGAS_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"RAGAS evaluation complete for {payload['company']} ({payload['question_count']} questions)")
    print(f"Eval mode: {payload['eval_mode']} | Model: {payload['text_model']}")

    if "retrieval_summary" in payload:
        r = payload["retrieval_summary"]
        print(
            f"Retrieval hit rate: {r['hit_rate']:.1%} | "
            f"Partial+hit: {r['partial_or_hit_rate']:.1%} | "
            f"Avg match: {r['average_best_match_score']:.2f}"
        )

    if "ragas_summary" in payload:
        s = payload["ragas_summary"]
        print(f"Faithfulness:      {s['faithfulness']:.3f}")
        print(f"Answer relevancy:  {s['answer_relevancy']:.3f}")
        print(f"Context precision: {s['context_precision']:.3f}")
        print(f"Context recall:    {s['context_recall']:.3f}")
        print(f"Hallucination:     {s['hallucination_free']:.3f}")
        print(f"Citation coverage: {s['citation_coverage']:.3f}")
        print(f"Unsupported score: {s['unsupported_claims_score']:.3f}")

    print(f"\nSaved to {output_path}")

    failures = [
        r for r in payload.get("results", [])
        if r.get("ragas", {}).get("faithfulness", 1.0) < 0.5
    ]
    if failures:
        print(f"\n{len(failures)} questions with low faithfulness (<0.5):")
        for r in failures[:5]:
            print(f"  - [{r['topic']}] {r['question'][:80]}...")

    return 0
