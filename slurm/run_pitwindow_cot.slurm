#!/bin/bash
#SBATCH --job-name=pit-window-cot
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=results/pitwindow_cot_%j.log
#SBATCH --error=results/pitwindow_cot_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4

source ~/envs/pit/bin/activate
cd /scratch/spp9400/pit-or-persist
export PYTHONPATH=$PYTHONPATH:$(pwd)

python eval/run_eval.py --mode cot --r 0.6 --noise plausible --ablation_field pit_window --n 500 --out results/ablation_pit_window_cot.jsonl

echo "DONE"
