# LongMemEval baseline instrumentation

This branch contains the phase-1, sidecar-only construction token
instrumentation and cleaned-dataset filename compatibility used by the
isolated LongMemEval-S first-question run.

- Native two-pass MemBlock construction remains enabled.
- `MEMBLOCK_MERGED_EXTRACTION=0` and graph expansion remain disabled.
- Split checks, Pass 1, temporal/tool follow-ups, and Pass 2 are recorded in
  the runner-provided `CONSTRUCTION_CALLS_FILE`.
- The sidecar is observational and does not enter prompts or output schemas.

The runner, post-processing scripts, and smoke report live in the separate
`longmemeval-baselines` experiment workspace. The executed question is
`e47becba` only.
