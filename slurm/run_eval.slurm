#!/bin/bash
#SBATCH --job-name=pit-eval
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=results/eval_%j.log
#SBATCH --error=results/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4

source ~/envs/pit/bin/activate
cd /scratch/spp9400/pit-or-persist   # ← change this

pip install transformers accelerate -q

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=== Zero-shot r=1.0 ==="
python eval/run_eval.py --mode zero_shot --r 1.0 --n 100 --out results/zeroshot_r1.jsonl

echo "=== Zero-shot r=0.6 ==="
python eval/run_eval.py --mode zero_shot --r 0.6 --n 100 --out results/zeroshot_r06.jsonl