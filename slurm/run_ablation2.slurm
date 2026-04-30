#!/bin/bash
#SBATCH --job-name=pit-abla2
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=results/ablation2_%j.log
#SBATCH --error=results/ablation2_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4

source ~/envs/pit/bin/activate
cd /scratch/spp9400/pit-or-persist
export PYTHONPATH=$PYTHONPATH:$(pwd)

for field in deg_rate rainfall pit_window; do
    for mode in zero_shot cot; do
        echo "=== Ablation: field=$field mode=$mode ==="
        python eval/run_eval.py \
            --mode $mode --r 0.6 --noise plausible \
            --ablation_field $field --n 500 \
            --out results/ablation_${field}_${mode}.jsonl
    done
done

echo "=== ALL DONE ==="
