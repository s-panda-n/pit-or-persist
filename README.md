# pit-or-persist

**Reasoning Under Uncertainty: Evaluating LLM Robustness for F1 Pit-Stop Decisions**

Investigates how LLM decision accuracy degrades as telemetry reliability decreases (r=1.0 → 0.4),
and whether prompting strategies (CoT, self-verification) mitigate this degradation.

## Structure
- `data/` — FastF1 snapshots and raw cache
- `mcp_server/` — MCP telemetry server with noise injection
- `eval/` — prompt templates, eval loop, metrics
- `slurm/` — HPC job scripts
- `results/` — experiment outputs (gitignored)
- `paper/` — LaTeX writeup
