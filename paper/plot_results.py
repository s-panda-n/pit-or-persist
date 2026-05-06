import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

r_levels = [0.4, 0.6, 0.8, 1.0]

# ── Data loading ──────────────────────────────────────────────

def load(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    acc = sum(r['correct'] for r in rows) / len(rows)
    pit = [r for r in rows if r['label'] == 1]
    stay = [r for r in rows if r['label'] == 0]
    pred_pit = [r for r in rows if r['pred'] == 1]
    precision = sum(r['correct'] for r in pred_pit) / len(pred_pit) if pred_pit else 0
    recall = sum(r['correct'] for r in pit) / len(pit) if pit else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    stay_acc = sum(r['correct'] for r in stay) / len(stay) if stay else 0
    return {'overall': acc, 'pit': recall, 'stay': stay_acc, 'f1': f1,
            'precision': precision, 'n': len(rows)}

def load_safe(path):
    try:
        return load(path)
    except:
        return None

# Qwen results
qwen_zs = {r: load_safe(f'results/zeroshot_plausible_r{r}.jsonl') for r in r_levels}
qwen_cot = {r: load_safe(f'results/cot_plausible_r{r}.jsonl') for r in r_levels}
qwen_zs_anom = load_safe('results/zeroshot_anomalous_r0.6.jsonl')
qwen_cot_anom = load_safe('results/cot_anomalous_r0.6.jsonl')

# Haiku results
haiku_zs = {r: load_safe(f'results/haiku_zero_shot_plausible_r{r}.jsonl') for r in r_levels}
haiku_cot = {r: load_safe(f'results/haiku_cot_plausible_r{r}.jsonl') for r in r_levels}
haiku_zs_anom = load_safe('results/haiku_zeroshot_anomalous_r0.6.jsonl')
haiku_cot_anom = load_safe('results/haiku_cot_anomalous_r0.6.jsonl')

# Faithfulness
faithfulness_qwen = {1.0: 0.608, 0.8: 0.511, 0.6: 0.414, 0.4: 0.305}

# Ablation (zero-shot F1 at r=0.6, corrupting one field at a time)
ablation_fields = ['tyre_age', 'compound', 'position', 'gap_to_leader',
                   'deg_rate', 'rainfall', 'pit_window']
ablation_labels = ['Tyre Age', 'Compound', 'Position', 'Gap to\nLeader',
                   'Deg Rate', 'Rainfall', 'Pit Window\n(tool)']

def load_ablation(field, mode):
    path = f'results/ablation_{field}_{mode}.jsonl'
    r = load_safe(path)
    return r['overall'] if r else None

# ── Figure 1: Degradation curves (2x2) ───────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

colors = {
    'qwen_zs':   '#2563EB',   # blue
    'qwen_cot':  '#DC2626',   # red
    'haiku_zs':  '#0891B2',   # cyan
    'haiku_cot': '#D97706',   # amber
}

# ── Left: Overall accuracy ────────────────────────────────────
ax = axes[0]
ax.axhline(0.80, color='gray', linestyle='--', linewidth=1,
           label='Always-STAY baseline', alpha=0.6)

def plot_line(ax, data, r_levels, color, marker, label, linestyle='-'):
    vals = [data[r]['overall'] if data[r] else None for r in r_levels]
    valid_r = [r for r, v in zip(r_levels, vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    ax.plot(valid_r, valid_v, marker=marker, color=color, linewidth=1.8,
            markersize=5, label=label, linestyle=linestyle)

plot_line(ax, qwen_zs,   r_levels, colors['qwen_zs'],   'o', 'Qwen zero-shot')
plot_line(ax, qwen_cot,  r_levels, colors['qwen_cot'],  's', 'Qwen CoT')
plot_line(ax, haiku_zs,  r_levels, colors['haiku_zs'],  'o', 'Haiku zero-shot', '--')
plot_line(ax, haiku_cot, r_levels, colors['haiku_cot'], 's', 'Haiku CoT', '--')

# Anomalous points
if qwen_zs_anom:
    ax.scatter([0.6], [qwen_zs_anom['overall']], marker='^', color=colors['qwen_zs'],
               s=80, zorder=5, edgecolors='white', linewidth=0.5)
if qwen_cot_anom:
    ax.scatter([0.6], [qwen_cot_anom['overall']], marker='^', color=colors['qwen_cot'],
               s=80, zorder=5, edgecolors='white', linewidth=0.5)
if haiku_zs_anom:
    ax.scatter([0.6], [haiku_zs_anom['overall']], marker='^', color=colors['haiku_zs'],
               s=80, zorder=5, edgecolors='white', linewidth=0.5)
if haiku_cot_anom:
    ax.scatter([0.6], [haiku_cot_anom['overall']], marker='^', color=colors['haiku_cot'],
               s=80, zorder=5, edgecolors='white', linewidth=0.5)

# Annotation for anomalous
ax.annotate('▲ = anomalous noise', xy=(0.6, 0.31), fontsize=8.5, color='gray',
            ha='center')

ax.set_xlabel('Reliability $r$')
ax.set_ylabel('Overall accuracy')
ax.set_title('(a) Overall accuracy vs. reliability')
ax.set_xlim(0.35, 1.05)
ax.set_ylim(0.25, 0.90)
ax.set_xticks(r_levels)
ax.legend(fontsize=8.5, framealpha=0.3, loc='upper left')
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

# ── Right: Pit recall ─────────────────────────────────────────
ax = axes[1]

def plot_line_pit(ax, data, r_levels, color, marker, label, linestyle='-'):
    vals = [data[r]['pit'] if data[r] else None for r in r_levels]
    valid_r = [r for r, v in zip(r_levels, vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    ax.plot(valid_r, valid_v, marker=marker, color=color, linewidth=1.8,
            markersize=5, label=label, linestyle=linestyle)

plot_line_pit(ax, qwen_zs,   r_levels, colors['qwen_zs'],   'o', 'Qwen zero-shot')
plot_line_pit(ax, qwen_cot,  r_levels, colors['qwen_cot'],  's', 'Qwen CoT')
plot_line_pit(ax, haiku_zs,  r_levels, colors['haiku_zs'],  'o', 'Haiku zero-shot', '--')
plot_line_pit(ax, haiku_cot, r_levels, colors['haiku_cot'], 's', 'Haiku CoT', '--')

if qwen_zs_anom:
    ax.scatter([0.6], [qwen_zs_anom['pit']], marker='^', color=colors['qwen_zs'],
               s=80, zorder=5, edgecolors='white', linewidth=0.5)
if qwen_cot_anom:
    ax.scatter([0.6], [qwen_cot_anom['pit']], marker='^', color=colors['qwen_cot'],
               s=80, zorder=5, edgecolors='white', linewidth=0.5)
if haiku_zs_anom:
    ax.scatter([0.6], [haiku_zs_anom['pit']], marker='^', color=colors['haiku_zs'],
               s=80, zorder=5, edgecolors='white', linewidth=0.5)
if haiku_cot_anom:
    ax.scatter([0.6], [haiku_cot_anom['pit']], marker='^', color=colors['haiku_cot'],
               s=80, zorder=5, edgecolors='white', linewidth=0.5)

ax.set_xlabel('Reliability $r$')
ax.set_ylabel('Pit recall')
ax.set_title('(b) Pit recall vs. reliability')
ax.set_xlim(0.35, 1.05)
ax.set_ylim(0.0, 1.05)
ax.set_xticks(r_levels)
ax.legend(fontsize=8.5, framealpha=0.3, loc='upper right')
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('paper/degradation_curves.pdf', bbox_inches='tight')
plt.savefig('paper/degradation_curves.png', bbox_inches='tight', dpi=300)
print("Saved degradation_curves.pdf/png")
plt.close()

# ── Figure 2: Faithfulness degradation ───────────────────────

fig, ax = plt.subplots(figsize=(6, 4))

faith_r = [0.4, 0.6, 0.8, 1.0]
faith_vals = [faithfulness_qwen[r] for r in faith_r]

ax.plot(faith_r, faith_vals, 'o-', color=colors['qwen_cot'], linewidth=1.8,
        markersize=6, label='Qwen CoT faithfulness')
ax.axhline(faith_vals[-1], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Annotate specific values
for r, v in zip(faith_r, faith_vals):
    ax.annotate(f'{v:.2f}', (r, v), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=9)

ax.set_xlabel('Reliability $r$')
ax.set_ylabel('Faithfulness score')
ax.set_title('CoT reasoning faithfulness vs. telemetry reliability')
ax.set_xlim(0.35, 1.05)
ax.set_ylim(0.0, 0.80)
ax.set_xticks(faith_r)
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('paper/faithfulness_curve.pdf', bbox_inches='tight')
plt.savefig('paper/faithfulness_curve.png', bbox_inches='tight', dpi=300)
print("Saved faithfulness_curve.pdf/png")
plt.close()

# ── Figure 3: Ablation bar chart ─────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
baseline_zs_overall = 0.454
baseline_cot_overall = 0.442

zs_vals = [load_ablation(f, 'zero_shot') for f in ablation_fields]
cot_vals = [load_ablation(f, 'cot') for f in ablation_fields]

x = np.arange(len(ablation_fields))
width = 0.35

for ax_idx, (ax, vals, baseline, title, color) in enumerate(zip(
    axes,
    [zs_vals, cot_vals],
    [baseline_zs_overall, baseline_cot_overall],
    ['(a) Zero-shot: accuracy when corrupting one field', '(b) CoT: accuracy when corrupting one field'],
    [colors['qwen_zs'], colors['qwen_cot']]
)):
    plot_vals = [v if v is not None else 0 for v in vals]
    bars = ax.bar(x, plot_vals, width=0.5, color=color, alpha=0.75, label='Single-field corruption')
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1.2,
               label=f'All-fields baseline ({baseline:.3f})')

    # Color bars that are below baseline red
    for bar, val in zip(bars, vals):
        if val is not None and val < baseline:
            bar.set_color('#DC2626')
            bar.set_alpha(0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ablation_labels, fontsize=9)
    ax.set_ylabel('Overall accuracy')
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.35, 0.65)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Annotate values on bars
    for bar, val in zip(bars, vals):
        if val is not None and val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Per-field ablation: effect of corrupting individual telemetry fields (r=0.6)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('paper/ablation_chart.pdf', bbox_inches='tight')
plt.savefig('paper/ablation_chart.png', bbox_inches='tight', dpi=300)
print("Saved ablation_chart.pdf/png")
plt.close()

print("\nAll figures saved to paper/")
