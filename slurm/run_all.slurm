#!/bin/bash
#SBATCH --job-name=pit-eval-full
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=results/full_%j.log
#SBATCH --error=results/full_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4

source ~/envs/pit/bin/activate
cd /scratch/spp9400/pit-or-persist
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Zero-shot + tool across all reliability levels
for r in 1.0 0.8 0.6 0.4; do
    echo "=== Zero-shot r=$r ==="
    python eval/run_eval.py --mode zero_shot --r $r --n 200 \
        --out results/zeroshot_r${r}.jsonl
done

# CoT + tool across all reliability levels
for r in 1.0 0.8 0.6 0.4; do
    echo "=== CoT r=$r ==="
    python eval/run_eval.py --mode cot --r $r --n 200 \
        --out results/cot_r${r}.jsonl
done

echo "=== ALL DONE ==="