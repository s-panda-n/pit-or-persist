import json
import random
from pathlib import Path

random.seed(42)

with open('data/snapshots/snapshots.jsonl') as f:
    snapshots = [json.loads(l) for l in f]

pit = [s for s in snapshots if s['label'] == 1]
stay = [s for s in snapshots if s['label'] == 0]

print(f"Pit: {len(pit)}, Stay: {len(stay)}")

# subsample stay to 3x pit (still realistic but not degenerate)
# gives ~920 pit, ~690 stay -> roughly balanced enough
stay_sampled = random.sample(stay, min(len(pit) * 4, len(stay)))

# combine and shuffle
balanced = pit + stay_sampled
random.shuffle(balanced)

print(f"Balanced dataset: {len(balanced)} total")
print(f"Pit: {sum(s['label']==1 for s in balanced)} | Stay: {sum(s['label']==0 for s in balanced)}")

with open('data/snapshots/snapshots_balanced.jsonl', 'w') as f:
    for s in balanced:
        f.write(json.dumps(s) + '\n')

print("Saved to data/snapshots/snapshots_balanced.jsonl")