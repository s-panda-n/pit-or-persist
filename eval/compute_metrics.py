import json
import os
from pathlib import Path

def compute_metrics(path):
    with open(path) as f:
        results = [json.loads(l) for l in f]
    
    acc = sum(r['correct'] for r in results) / len(results)
    pit = [r for r in results if r['label'] == 1]
    stay = [r for r in results if r['label'] == 0]
    pit_acc = sum(r['correct'] for r in pit) / len(pit) if pit else 0
    stay_acc = sum(r['correct'] for r in stay) / len(stay) if stay else 0
    pred_pit = [r for r in results if r['pred'] == 1]
    precision = sum(r['correct'] for r in pred_pit) / len(pred_pit) if pred_pit else 0
    recall = pit_acc
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'file': Path(path).name,
        'n': len(results),
        'overall': round(acc, 3),
        'pit_recall': round(recall, 3),
        'pit_precision': round(precision, 3),
        'pit_f1': round(f1, 3),
        'stay_acc': round(stay_acc, 3),
    }

files = [
    'results/zeroshot_plausible_r1.0.jsonl',
    'results/zeroshot_plausible_r0.8.jsonl',
    'results/zeroshot_plausible_r0.6.jsonl',
    'results/zeroshot_plausible_r0.4.jsonl',
    'results/cot_plausible_r1.0.jsonl',
    'results/cot_plausible_r0.8.jsonl',
    'results/cot_plausible_r0.6.jsonl',
    'results/cot_plausible_r0.4.jsonl',
    'results/zeroshot_anomalous_r0.6.jsonl',
    'results/cot_anomalous_r0.6.jsonl',
]

print(f"{'Condition':<35} {'Overall':>8} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Stay':>6}")
print('-' * 75)
all_metrics = []
for f in files:
    try:
        m = compute_metrics(f)
        print(f"{m['file']:<35} {m['overall']:>8.3f} {m['pit_precision']:>6.3f} {m['pit_recall']:>7.3f} {m['pit_f1']:>6.3f} {m['stay_acc']:>6.3f}")
        all_metrics.append(m)
    except Exception as e:
        print(f"{f}: ERROR {e}")

with open('results/all_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("\nSaved to results/all_metrics.json")