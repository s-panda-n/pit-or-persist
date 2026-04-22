import json
import argparse
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mcp_server.server import serve_snapshot
from eval.prompts import zero_shot_prompt, cot_prompt

def load_snapshots(path, n=None):
    with open(path) as f:
        snaps = [json.loads(l) for l in f]
    if n:
        snaps = snaps[:n]
    return snaps

def parse_decision(text, mode):
    text = text.upper()
    if mode == "zero_shot":
        if "PIT" in text:
            return 1
        return 0
    elif mode == "cot":
        match = re.search(r'DECISION:\s*(PIT|STAY)', text)
        if match:
            return 1 if match.group(1) == "PIT" else 0
        # fallback
        if "PIT" in text:
            return 1
        return 0

def evaluate(model, tokenizer, snapshots, mode, r, device, max_new_tokens=16):
    results = []
    prompt_fn = zero_shot_prompt if mode == "zero_shot" else cot_prompt
    if mode == "cot":
        max_new_tokens = 300

    for i, snap in enumerate(snapshots):
        tel = serve_snapshot(snap, r=r)
        prompt = prompt_fn(tel)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        pred = parse_decision(generated, mode)
        label = snap["label"]

        results.append({
            "id": snap["id"],
            "label": label,
            "pred": pred,
            "correct": int(pred == label),
            "generated": generated[:200],  # truncate for storage
            "r": r,
            "mode": mode
        })

        if (i + 1) % 10 == 0:
            acc = sum(x['correct'] for x in results) / len(results)
            print(f"  [{i+1}/{len(snapshots)}] acc={acc:.3f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--snapshots", default="data/snapshots/snapshots_balanced.jsonl")
    parser.add_argument("--mode", default="zero_shot", choices=["zero_shot", "cot"])
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="results/eval_results.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    snapshots = load_snapshots(args.snapshots, n=args.n)
    print(f"Loaded {len(snapshots)} snapshots | mode={args.mode} | r={args.r}")

    results = evaluate(model, tokenizer, snapshots, args.mode, args.r, device)

    # metrics
    acc = sum(r['correct'] for r in results) / len(results)
    pit_snaps = [r for r in results if r['label'] == 1]
    stay_snaps = [r for r in results if r['label'] == 0]
    pit_acc = sum(r['correct'] for r in pit_snaps) / len(pit_snaps) if pit_snaps else 0
    stay_acc = sum(r['correct'] for r in stay_snaps) / len(stay_snaps) if stay_snaps else 0

    print(f"\n=== RESULTS ===")
    print(f"Overall accuracy: {acc:.3f}")
    print(f"Pit accuracy:     {pit_acc:.3f} ({len(pit_snaps)} examples)")
    print(f"Stay accuracy:    {stay_acc:.3f} ({len(stay_snaps)} examples)")

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()