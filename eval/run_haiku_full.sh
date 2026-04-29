#!/bin/bash
source ~/envs/pit/bin/activate
cd /scratch/spp9400/pit-or-persist
export PYTHONPATH=$PYTHONPATH:$(pwd)
export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
for mode in zero_shot cot; do
    for r in 1.0 0.8 0.6 0.4; do
        echo "=== Haiku $mode r=$r ==="
        python eval/run_eval_api.py \
            --mode $mode --r $r --noise plausible --n 200 \
            --out results/haiku_${mode}_plausible_r${r}.jsonl
    done
done
echo "=== Haiku zero_shot anomalous r=0.6 ==="
python eval/run_eval_api.py --mode zero_shot --r 0.6 --noise anomalous --n 200 --out results/haiku_zeroshot_anomalous_r0.6.jsonl
echo "=== Haiku cot anomalous r=0.6 ==="
python eval/run_eval_api.py --mode cot --r 0.6 --noise anomalous --n 200 --out results/haiku_cot_anomalous_r0.6.jsonl
echo "=== ALL DONE ==="
