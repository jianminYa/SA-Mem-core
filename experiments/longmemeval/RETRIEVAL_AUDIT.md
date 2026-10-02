# LongMemEval-S Retrieval Audit

This is a post-run audit of the frozen 50-question phase-1 comparison. It does
not change either baseline implementation or rerun any question.

## What text was embedded?

Both phase-1 retrievers used the same representation pattern for memory
ranking:

```text
content_text + topic_kw_text + event_text
```

Here `content_text` is the preserved original dialogue for the memory unit.
Therefore, raw dialogue was included in the embedding input for both MemBox
and SA-Mem. It is not an SA-Mem-only behavior and cannot, by itself, explain
the accuracy gap.

The raw dialogue is also used again in the final QA prompt: the selected Top-10
memory units are concatenated as their original `content_text`. This creates a
separate long-context effect from the embedding-ranking effect.

## Observed 50-question results

| Metric | MemBox | SA-Mem |
|---|---:|---:|
| QA accuracy | 18/50 (36%) | 16/50 (32%) |
| Evidence Hit@1 | 88% | 82% |
| Evidence Hit@5 | 96% | 88% |
| Evidence Hit@10 | 100% | 96% |
| Evidence Hit@20 | 100% | 98% |
| MRR | 0.9152 | 0.8532 |

Extraction loss was not observed for either system in the aggregate audit.
MemBox placed the gold session in the final Top-10 for all 50 questions; SA-Mem
did so for 48/50. Thus retrieval explains some difference, but not the whole
four-point QA gap.

## Important protocol difference

The MemBox run used its plain vector retriever. The SA-Mem LongMemEval adapter
ran the native enhanced path (`use_enhanced=True`), which can perform query
parsing, query rewriting, temporal filtering, and then vector ranking.

Observed in the saved SA-Mem logs:

- 25/50 query embedding inputs differed from the original question after
  query rewriting.
- 3/50 questions actually reduced the candidate pool through temporal
  filtering; the other questions retained the full pool after fallback or had
  no effective filter.
- Some rewrites removed important constraints, for example location,
  date/month, comparison clauses, or the second item in a multi-part question.

This means the phase-1 result is a valid comparison of the configured native
baselines, but it is not a pure representation-only retrieval ablation.

## Long QA context effect

The actual Top-10 QA contexts were built from raw memory dialogue. With the
same local tokenizer used for diagnostics, the context statistics were:

| Metric | MemBox | SA-Mem |
|---|---:|---:|
| Mean context tokens | 9,118 | 9,178 |
| Median context tokens | 8,879 | 8,494 |
| Maximum context tokens | 15,607 | 16,719 |

Among wrong answers, the mean context was longer than among correct answers
for both systems. This is consistent with distractor/attention pressure, but
it is observational evidence rather than a causal ablation.

## Conclusion

The current evidence does **not** support the claim that SA-Mem underperformed
because it alone embedded raw context. Both systems did that. The strongest
explanations supported by the artifacts are:

1. SA-Mem's enhanced query rewrite/filter path is not matched by MemBox's
   plain vector path and sometimes removes useful question constraints.
2. SA-Mem's structured event/topic representation and ranking inputs differ in
   content from MemBox's generated event/topic fields, even though the outer
   concatenation pattern is the same.
3. Both systems pass long raw-dialogue Top-10 contexts to QA; the majority of
   errors occur even when the gold session is present, so QA context usage and
   generation remain a major error source.

A causal follow-up should separately compare: original-question vs rewritten
query, enhanced vs pure vector ranking, raw-dialogue vs structured-only
embedding text, and compacted vs raw Top-10 QA context. Those are ablations for
the next phase, not changes to this baseline.
