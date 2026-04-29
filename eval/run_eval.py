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

def evaluate(model, tokenizer, snapshots, mode, r, device, max_new_tokens=16, noise_type="plausible", ablation_field=None):
    results = []
    prompt_fn = zero_shot_prompt if mode == "zero_shot" else cot_prompt
    if mode == "cot":
        max_new_tokens = 300

    for i, snap in enumerate(snapshots):
        tel = serve_snapshot(snap, r=r, noise_type=noise_type, ablation_field=ablation_field)
        prompt = prompt_fn(tel)

        # Use chat template for instruct models
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

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
            "generated": generated,
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
    parser.add_argument("--noise", default="plausible", choices=["plausible", "anomalous"])
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--ablation_field", default=None,
    help="If set, corrupt only this field. Others stay clean.")
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

    results = evaluate(model, tokenizer, snapshots, args.mode, args.r, device,noise_type=args.noise, ablation_field=args.ablation_field)

    # metrics
    def compute_metrics(results):
        acc = sum(r['correct'] for r in results) / len(results)
        pit = [r for r in results if r['label'] == 1]
        stay = [r for r in results if r['label'] == 0]
        pit_acc = sum(r['correct'] for r in pit) / len(pit) if pit else 0
        stay_acc = sum(r['correct'] for r in stay) / len(stay) if stay else 0
    
        # precision: of all predicted PIT, how many were actually PIT
        pred_pit = [r for r in results if r['pred'] == 1]
        precision = sum(r['correct'] for r in pred_pit) / len(pred_pit) if pred_pit else 0
        recall = pit_acc  # same as pit accuracy
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
        return {
            'overall': acc,
            'pit_recall': pit_acc,
            'pit_precision': precision,
            'pit_f1': f1,
            'stay_acc': stay_acc,
            'n': len(results),
            'n_pit': len(pit),
            'n_stay': len(stay),
            'n_pred_pit': len(pred_pit),
        }

    metrics = compute_metrics(results)
    print(f"\n=== RESULTS ===")
    print(f"Overall accuracy:  {metrics['overall']:.3f}")
    print(f"Pit recall:        {metrics['pit_recall']:.3f} ({metrics['n_pit']} examples)")
    print(f"Pit precision:     {metrics['pit_precision']:.3f} ({metrics['n_pred_pit']} predicted pit)")
    print(f"Pit F1:            {metrics['pit_f1']:.3f}")
    print(f"Stay accuracy:     {metrics['stay_acc']:.3f} ({metrics['n_stay']} examples)")

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    # save metrics alongside results
    metrics_path = args.out.replace('.jsonl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({**metrics, 'mode': args.mode, 'r': args.r, 'noise': args.noise}, f, indent=2)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()