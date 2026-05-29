# RAGAS Lap 2 scorecard

Status: **computed end-to-end on the new 24-row golden dataset**.

- Dataset: `sample_data/ragas_esg_eval_dataset.csv`
- Index: `outputs/rag_index` built from the four golden PDFs in `eval/pdfs`
- Retrieval: `semantic_rerank`, candidate_k=100, top_k=8
- Rows: **24**
- Companies: Danone, Enel, Schneider Electric, BNP Paribas
- Generators: Qwen vs gpt-oss
- Judge: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`

## Overall metrics

| metric | Qwen | gpt-oss |
|---|---:|---:|
| context_relevance | 0.198 | 0.198 |
| faithfulness | 0.891 | 0.907 |
| answer_relevancy | 0.779 | 0.838 |
| answer_correctness | 0.625 | 0.667 |
| abstention_rate | 8.3% | 4.2% |
| gold_chunk_id_hit_rate | 0.0% | 0.0% |

## Per-company metrics

| company | n | context_rel | expected_hit | faithfulness (Q\|g) | answer_relevancy (Q\|g) | answer_correctness (Q\|g) | abstention_rate (Q\|g) |
|---|---:|---:|---:|---|---|---|---|
| BNP Paribas | 6 | 0.250 | 0% | 0.907 \| 1.000 | 0.767 \| 0.867 | 0.583 \| 0.750 | 0% \| 0% |
| Danone | 6 | 0.146 | 0% | 0.875 \| 0.845 | 0.750 \| 0.833 | 0.583 \| 0.583 | 17% \| 17% |
| Enel | 6 | 0.292 | 0% | 1.000 \| 0.854 | 0.867 \| 0.850 | 1.000 \| 0.667 | 0% \| 0% |
| Schneider Electric | 6 | 0.104 | 0% | 0.783 \| 0.929 | 0.733 \| 0.800 | 0.333 \| 0.667 | 17% \| 0% |

## Read

This run fixes the previous caveat: contexts now come from the live RAG retrieval index rather than from the dataset gold contexts. The benchmark is much cleaner than the old 49-row run, but retrieval is now the bottleneck: `context_relevance` is only 0.198 overall, so `answer_correctness` drops below the earlier gold-context-only scorecard.

Note: `gold_chunk_id_hit_rate` is not a reliable recall metric here. The new golden dataset uses human-authored chunk IDs such as `danone-p02-001`, while the live index creates parser-derived chunk IDs. The 0% value means the ID namespaces do not match, not necessarily that the exact evidence text is always absent from retrieved chunks.
