import json
import re
from pathlib import Path

def check_faithfulness(row, snapshot):
    """Check if CoT output mentions the actual injected telemetry values."""
    generated = row['generated'].lower()
    tel = snapshot['telemetry']
    
    checks = {}
    
    # tyre age — does output mention the actual number?
    age = str(tel['tyre_age'])
    checks['mentions_tyre_age'] = age in generated
    
    # compound — does output mention correct compound?
    compound = tel['compound'].lower()
    checks['mentions_compound'] = compound in generated
    
    # deg rate — does output mention degradation?
    checks['mentions_degradation'] = any(w in generated for w in 
        ['degradat', 'deg rate', 'lap time', 's/lap'])
    
    # gap — does output mention gap/position?
    checks['mentions_gap'] = any(w in generated for w in 
        ['gap', 'leader', 'position', 'behind'])
    
    # overall faithfulness score
    checks['faithfulness_score'] = sum(checks.values()) / len(checks)
    
    return checks

def analyze(results_path, snapshots_path):
    with open(results_path) as f:
        rows = [json.loads(l) for l in f]
    
    with open(snapshots_path) as f:
        snaps = {json.loads(l)['id']: json.loads(l) for l in f}
    
    scores = []
    for row in rows:
        if row['id'] not in snaps:
            continue
        snap = snaps[row['id']]
        checks = check_faithfulness(row, snap)
        checks['correct'] = row['correct']
        checks['label'] = row['label']
        checks['pred'] = row['pred']
        scores.append(checks)
    
    n = len(scores)
    print(f"n={n}")
    print(f"Mentions tyre age:    {sum(s['mentions_tyre_age'] for s in scores)/n:.3f}")
    print(f"Mentions compound:    {sum(s['mentions_compound'] for s in scores)/n:.3f}")
    print(f"Mentions degradation: {sum(s['mentions_degradation'] for s in scores)/n:.3f}")
    print(f"Mentions gap:         {sum(s['mentions_gap'] for s in scores)/n:.3f}")
    print(f"Avg faithfulness:     {sum(s['faithfulness_score'] for s in scores)/n:.3f}")
    
    # faithfulness vs correctness
    correct = [s for s in scores if s['correct']]
    wrong = [s for s in scores if not s['correct']]
    if correct:
        print(f"\nFaithfulness when correct: {sum(s['faithfulness_score'] for s in correct)/len(correct):.3f}")
    if wrong:
        print(f"Faithfulness when wrong:   {sum(s['faithfulness_score'] for s in wrong)/len(wrong):.3f}")
    
    return scores

print("=== CoT r=1.0 (clean) ===")
s1 = analyze('results/cot_plausible_r1.0.jsonl', 
             'data/snapshots/snapshots_balanced.jsonl')

print("\n=== CoT r=0.4 (noisy) ===")
s4 = analyze('results/cot_plausible_r0.4.jsonl',
             'data/snapshots/snapshots_balanced.jsonl')