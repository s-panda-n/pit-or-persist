# pit-or-persist

**Reasoning Under Uncertainty: Evaluating LLM Robustness for Formula 1 Pit-Stop Decisions**

Spandan Patil · NYU Courant Institute · Building LLM Reasoners (CSCI-GA 3033) · Spring 2026

---

## Overview

This project studies how LLM decision-making degrades as tool reliability
decreases, using Formula 1 pit-stop strategy as a controlled testbed with
verifiable ground truth. We build an MCP-style tool server, inject configurable
noise at four reliability levels, and evaluate Qwen2.5-7B-Instruct and Claude
Haiku 3.5 under zero-shot and chain-of-thought prompting.

**Key finding:** Corrupted tool outputs cause systematic pit-bias, not random
errors. As reliability drops from r=1.0 to r=0.4, pit recall rises from 42%
to 78%. Plausible noise fools both models equally; CoT helps on anomalous
noise but not plausible noise.

---

### Repository Structure

```
pit-or-persist/
├── data/
│   ├── build_snapshots.py       # extract lap-level snapshots from FastF1
│   ├── balance_snapshots.py     # balance pit/stay class distribution
│   └── explore.py               # exploratory data analysis
├── mcp_server/
│   ├── server.py                # MCP-style tool server with noise injection
│   └── __init__.py
├── eval/
│   ├── prompts.py               # zero-shot and CoT prompt templates
│   ├── run_eval.py              # Qwen evaluation on HPC (GPU)
│   ├── run_eval_api.py          # Claude Haiku evaluation via Anthropic API
│   ├── compute_metrics.py       # compute F1, precision, recall from results
│   └── faithfulness.py          # CoT scratchpad faithfulness analysis
├── paper/
│   └── plot_results.py          # generate all paper figures
├── slurm/
│   ├── run_full.slurm           # full Qwen sweep (8 conditions)
│   ├── run_ablation.slurm       # per-field ablation jobs
│   └── run_pitwindow_cot.slurm  # pit_window CoT ablation
└── results/                     # gitignored — generated on HPC
```

---

## Setup

```bash
# on NYU HPC Greene
python -m venv ~/envs/pit
source ~/envs/pit/bin/activate
pip install fastf1 pandas numpy transformers accelerate torch anthropic matplotlib
```

Set your Anthropic API key as an environment variable:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

---

## Data Pipeline

```bash
# 1. Extract snapshots from FastF1 (downloads ~500MB of telemetry data)
python data/build_snapshots.py
# Output: data/snapshots/snapshots.jsonl (9,753 raw snapshots)

# 2. Balance the dataset
python data/balance_snapshots.py
# Output: data/snapshots/snapshots_balanced.jsonl (1,705 balanced snapshots)
```

The dataset covers 10 races from the 2022 and 2023 F1 seasons:
Bahrain, Australia, Spain, Monaco, Italy for each season.
Ground truth labels are derived from PitInTime fields in FastF1 data.

---

## MCP Tool Server

`mcp_server/server.py` implements six tool functions:

| Tool | Returns | Noise type |
|---|---|---|
| `get_tyre_age` | Tyre age in laps | Integer additive |
| `get_compound` | SOFT / MEDIUM / HARD | Categorical replacement |
| `get_gap` | Gap to leader in seconds | Float Gaussian |
| `get_deg_rate` | Lap time degradation s/lap | Float Gaussian |
| `get_weather` | Air temp + rainfall boolean | Float + bit flip |
| `get_pit_window` | PIT/STAY recommendation | Derived from above |

Noise injection at reliability `r`:
- With probability `r`: tool returns true value
- With probability `(1-r)`: tool returns corrupted value
- **Plausible noise**: realistic ranges, indistinguishable from true values
- **Anomalous noise**: physically impossible constants (tyre_age=999, gap=-999)

---

## Running Experiments

### Qwen2.5-7B on HPC (SLURM)

```bash
# full sweep — zero-shot and CoT, all 4 reliability levels
sbatch slurm/run_full.slurm

# per-field ablation
sbatch slurm/run_ablation.slurm

# single run example
python eval/run_eval.py \
    --mode zero_shot \
    --r 1.0 \
    --noise plausible \
    --n 500 \
    --out results/zeroshot_r1.jsonl
```

Arguments:
- `--mode`: `zero_shot` or `cot`
- `--r`: reliability level (1.0, 0.8, 0.6, 0.4)
- `--noise`: `plausible` or `anomalous`
- `--ablation_field`: corrupt only one field (tyre_age, compound, position,
  gap_to_leader, deg_rate, rainfall, pit_window)
- `--n`: number of examples

### Claude Haiku via API

```bash
python eval/run_eval_api.py \
    --mode zero_shot \
    --r 1.0 \
    --noise plausible \
    --n 200 \
    --out results/haiku_zeroshot_r1.jsonl
```

---

## Computing Metrics

```bash
# compute F1, precision, recall on existing result files
python eval/compute_metrics.py

# faithfulness analysis on CoT outputs
python eval/faithfulness.py
```

---

## Generating Figures

```bash
# generates degradation_curves.png, faithfulness_curve.png, ablation_chart.png
python paper/plot_results.py
```

Figures are saved to `paper/`. Upload to Overleaf for the paper.

---

## Results Summary

### Qwen2.5-7B-Instruct (500 examples per condition)

| Condition | r | Overall | Pit Recall | Pit F1 |
|---|---|---|---|---|
| Zero-shot + tool | 1.0 | 58.0% | 41.8% | 0.305 |
| Zero-shot + tool | 0.8 | 55.6% | 55.5% | 0.355 |
| Zero-shot + tool | 0.6 | 45.4% | 63.6% | 0.339 |
| Zero-shot + tool | 0.4 | 43.6% | 78.2% | 0.379 |
| CoT + tool | 1.0 | 53.2% | 65.5% | 0.381 |
| CoT + tool | 0.8 | 47.0% | 65.5% | 0.352 |
| CoT + tool | 0.6 | 44.2% | 74.5% | 0.370 |
| CoT + tool | 0.4 | 44.8% | 79.1% | 0.387 |
| Zero-shot anomalous | 0.6 | 30.8% | 92.7% | 0.371 |
| CoT anomalous | 0.6 | 37.8% | 84.5% | 0.374 |

### Claude Haiku 3.5 (200 examples per condition)

| Condition | r | Overall | Pit Recall |
|---|---|---|---|
| Zero-shot | 1.0 | 70.0% | 46.2% |
| Zero-shot | 0.4 | 47.5% | 74.4% |
| CoT | 1.0 | 69.0% | 56.4% |
| CoT | 0.4 | 60.0% | 53.8% |
| Zero-shot anomalous | 0.6 | 29.0% | 92.3% |
| CoT anomalous | 0.6 | 67.0% | 46.2% |

---

## Key Findings

1. **Systematic pit-bias**: pit recall rises 42% to 78% as r drops — failure
   has a direction, not random noise
2. **CoT robustness**: CoT drops 8.4pt vs 14.4pt for zero-shot under plausible
   noise, but faithfulness degrades from 0.608 to 0.305
3. **Anomalous vs plausible**: CoT helps 29pt on anomalous noise (Haiku) but
   near zero on plausible noise — detectability is the key difference
4. **Frontier models**: Haiku starts higher (70% vs 58%) but drops more
   (22.5pt vs 14.4pt) — stronger models are not more robust to plausible noise
5. **Ablation**: rainfall and get_pit_window are the most damaging individual
   fields — corrupting either alone nearly reproduces full all-fields damage

---

## HPC Details

- **Cluster**: NYU Greene HPC
- **GPU**: NVIDIA A100 40GB
- **Account**: csci_ga_3033_131-2026sp
- **Partition**: c12m85-a100-1
- **Environment**: Python 3.9, PyTorch 2.x, HuggingFace Transformers

---

## Citation

If you use this benchmark or code, please cite:
@article{patil2026pit,
title={Reasoning Under Uncertainty: Evaluating LLM Robustness
for Formula 1 Pit-Stop Decisions},
author={Patil, Spandan},
institution={New York University},
year={2026}
}

---

## License

MIT