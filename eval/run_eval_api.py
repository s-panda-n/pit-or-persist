import json
import argparse
import re
import anthropic
from pathlib import Path
from mcp_server.server import serve_snapshot
from eval.prompts import zero_shot_prompt, cot_prompt

def parse_decision(text, mode):
    text_upper = text.upper()
    if mode == "cot":
        match = re.search(r'DECISION:\s*(PIT|STAY)', text_upper)
        if match:
            return 1 if match.group(1) == "PIT" else 0
    if "PIT" in text_upper:
        return 1
    return 0

def load_snapshots(path, n=None):
    with open(path) as f:
        snaps = [json.loads(l) for l in f]
    if n:
        snaps = snaps[:n]
    return snaps

def evaluate(snapshots, mode, r, noise_type, max_tokens=16):
    client = anthropic.Anthropic()
    results = []
    prompt_fn = zero_shot_prompt if mode == "zero_shot" else cot_prompt
    if mode == "cot":
        max_tokens = 400

    for i, snap in enumerate(snapshots):
        tel = serve_snapshot(snap, r=r, noise_type=noise_type)
        prompt = prompt_fn(tel)

        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        generated = message.content[0].text.strip()
        pred = parse_decision(generated, mode)
        label = snap["label"]

        results.append({
            "id": snap["id"],
            "label": label,
            "pred": pred,
            "correct": int(pred == label),
            "generated": generated[:300],
            "r": r,
            "mode": mode,
            "noise": noise_type,
            "model": "claude-haiku-4-5"
        })

        if (i + 1) % 10 == 0:
            acc = sum(x['correct'] for x in results) / len(results)
            print(f"  [{i+1}/{len(snapshots)}] acc={acc:.3f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default="data/snapshots/snapshots_balanced.jsonl")
    parser.add_argument("--mode", default="zero_shot", choices=["zero_shot", "cot"])
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--noise", default="plausible", choices=["plausible", "anomalous"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default="results/haiku_eval.jsonl")
    args = parser.parse_args()

    snapshots = load_snapshots(args.snapshots, n=args.n)
    print(f"Loaded {len(snapshots)} snapshots | mode={args.mode} | r={args.r} | noise={args.noise}")

    results = evaluate(snapshots, args.mode, args.r, args.noise)

    acc = sum(r['correct'] for r in results) / len(results)
    pit = [r for r in results if r['label'] == 1]
    stay = [r for r in results if r['label'] == 0]
    pred_pit = [r for r in results if r['pred'] == 1]
    precision = sum(r['correct'] for r in pred_pit) / len(pred_pit) if pred_pit else 0
    recall = sum(r['correct'] for r in pit) / len(pit) if pit else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== RESULTS ===")
    print(f"Overall:   {acc:.3f}")
    print(f"Pit F1:    {f1:.3f} (P={precision:.3f} R={recall:.3f})")
    print(f"Stay acc:  {sum(r['correct'] for r in stay)/len(stay):.3f}")

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()