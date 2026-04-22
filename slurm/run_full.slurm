#!/bin/bash
#SBATCH --job-name=pit-full
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=results/full_%j.log
#SBATCH --error=results/full_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4

source ~/envs/pit/bin/activate
cd /scratch/spp9400/pit-or-persist
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Plausible noise — zero-shot
for r in 1.0 0.8 0.6 0.4; do
    echo "=== Zero-shot plausible r=$r ==="
    python eval/run_eval.py --mode zero_shot --r $r --noise plausible --n 500 \
        --out results/zeroshot_plausible_r${r}.jsonl
done

# Plausible noise — CoT
for r in 1.0 0.8 0.6 0.4; do
    echo "=== CoT plausible r=$r ==="
    python eval/run_eval.py --mode cot --r $r --noise plausible --n 500 \
        --out results/cot_plausible_r${r}.jsonl
done

# Anomalous noise — zero-shot and CoT at r=0.6 only
echo "=== Zero-shot anomalous r=0.6 ==="
python eval/run_eval.py --mode zero_shot --r 0.6 --noise anomalous --n 500 \
    --out results/zeroshot_anomalous_r0.6.jsonl

echo "=== CoT anomalous r=0.6 ==="
python eval/run_eval.py --mode cot --r 0.6 --noise anomalous --n 500 \
    --out results/cot_anomalous_r0.6.jsonl

echo "=== ALL DONE ==="