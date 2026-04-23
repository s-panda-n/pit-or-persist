import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

r_levels = [0.4, 0.6, 0.8, 1.0]

# Load results
def load(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    acc = sum(r['correct'] for r in rows) / len(rows)
    pit = [r for r in rows if r['label'] == 1]
    stay = [r for r in rows if r['label'] == 0]
    pit_acc = sum(r['correct'] for r in pit) / len(pit) if pit else 0
    stay_acc = sum(r['correct'] for r in stay) / len(stay) if stay else 0
    return {'overall': acc, 'pit': pit_acc, 'stay': stay_acc, 'n': len(rows)}

zeroshot = {
    1.0: load('results/zeroshot_plausible_r1.0.jsonl'),
    0.8: load('results/zeroshot_plausible_r0.8.jsonl'),
    0.6: load('results/zeroshot_plausible_r0.6.jsonl'),
    0.4: load('results/zeroshot_plausible_r0.4.jsonl'),
}

# CoT — fill in as results arrive
cot = {
    1.0: load('results/cot_r1.0.jsonl'),
    0.8: load('results/cot_plausible_r0.8.jsonl'),
    0.6: load('results/cot_plausible_r0.6.jsonl'),
    0.4: load('results/cot_plausible_r0.4.jsonl'),
}
cot_anomalous_r06 = load('results/cot_anomalous_r0.6.jsonl')
anomalous_r06 = load('results/zeroshot_anomalous_test.jsonl')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ── Plot 1: Overall accuracy ──────────────────────────────────
ax = axes[0]
ax.axhline(0.78, color='gray', linestyle='--', linewidth=1, label='Always-STAY baseline', alpha=0.7)

zs_overall = [zeroshot[r]['overall'] for r in r_levels]
ax.plot(r_levels, zs_overall, 'o-', color='#2563EB', linewidth=1.8,
        markersize=5, label='Zero-shot + tool')

if len(cot) >= 4:
    cot_overall = [cot[r]['overall'] for r in r_levels]
    ax.plot(r_levels, cot_overall, 's-', color='#DC2626', linewidth=1.8,
            markersize=5, label='CoT + tool')
else:
    # plot what we have
    cot_r = sorted(cot.keys())
    ax.plot(cot_r, [cot[r]['overall'] for r in cot_r], 's--', color='#DC2626',
            linewidth=1.5, markersize=5, label='CoT + tool (partial)', alpha=0.6)

ax.scatter([0.6], [anomalous_r06['overall']], marker='^', color='#D97706',
           s=60, zorder=5, label='Zero-shot anomalous (r=0.6)')
ax.scatter([0.6], [cot_anomalous_r06['overall']], marker='^', color='#DC2626',
           s=60, zorder=5, label='CoT anomalous (r=0.6)')

ax.set_xlabel('Reliability r')
ax.set_ylabel('Overall accuracy')
ax.set_title('(a) Overall accuracy vs. reliability')
ax.set_xlim(0.35, 1.05)
ax.set_ylim(0.25, 0.90)
ax.set_xticks(r_levels)
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

# ── Plot 2: Pit recall ────────────────────────────────────────
ax = axes[1]

zs_pit = [zeroshot[r]['pit'] for r in r_levels]
ax.plot(r_levels, zs_pit, 'o-', color='#2563EB', linewidth=1.8,
        markersize=5, label='Zero-shot + tool')

if len(cot) >= 4:
    cot_pit = [cot[r]['pit'] for r in r_levels]
    ax.plot(r_levels, cot_pit, 's-', color='#DC2626', linewidth=1.8,
            markersize=5, label='CoT + tool')
else:
    cot_r = sorted(cot.keys())
    ax.plot(cot_r, [cot[r]['pit'] for r in cot_r], 's--', color='#DC2626',
            linewidth=1.5, markersize=5, label='CoT + tool (partial)', alpha=0.6)

ax.scatter([0.6], [anomalous_r06['pit']], marker='^', color='#D97706',
           s=60, zorder=5, label='Zero-shot anomalous (r=0.6)')

ax.set_xlabel('Reliability r')
ax.set_ylabel('Pit recall')
ax.set_title('(b) Pit recall vs. reliability')
ax.set_xlim(0.35, 1.05)
ax.set_ylim(0.0, 1.05)
ax.set_xticks(r_levels)
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('paper/degradation_curves.pdf', bbox_inches='tight')
plt.savefig('paper/degradation_curves.png', bbox_inches='tight', dpi=300)
print("Saved paper/degradation_curves.pdf and .png")