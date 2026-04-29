"""
MCP-style telemetry server for F1 pit strategy decisions.
Tools are called during snapshot construction to serve clean telemetry.
Noise is injected at the point of serving based on reliability level r.
"""

import json
import random
import math
from pathlib import Path

random.seed(42)

# ── Noise injection ──────────────────────────────────────────────────────────

def inject_noise(value, field_type, r):
    """
    Corrupt a telemetry value with probability (1 - r).
    field_type: 'int', 'float', 'compound', 'bool'
    """
    if random.random() < r:
        return value  # clean

    if field_type == 'int':
        noise = random.randint(-8, 8)
        return max(1, value + noise)

    elif field_type == 'float':
        noise = random.gauss(0, abs(value) * 0.3 + 0.5)
        return round(value + noise, 3)

    elif field_type == 'compound':
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        others = [c for c in compounds if c != value]
        return random.choice(others)

    elif field_type == 'bool':
        return not value

    return value

def inject_anomalous_noise(value, field_type):
    """Obviously wrong values — outside any realistic F1 range."""
    if field_type == 'int':
        return random.choice([0, 150, 999])
    elif field_type == 'float':
        return random.choice([-999.0, 999.0, 0.0])
    elif field_type == 'compound':
        return 'UNKNOWN'
    elif field_type == 'bool':
        return not value
    return value


def corrupt_telemetry_anomalous(telemetry):
    """Corrupt all fields with obviously anomalous values."""
    return {
        "tyre_age":      inject_anomalous_noise(telemetry["tyre_age"],      "int"),
        "compound":      inject_anomalous_noise(telemetry["compound"],       "compound"),
        "position":      inject_anomalous_noise(telemetry["position"],       "int"),
        "gap_to_leader": inject_anomalous_noise(telemetry["gap_to_leader"],  "float"),
        "deg_rate":      inject_anomalous_noise(telemetry["deg_rate"],       "float"),
        "air_temp":      inject_anomalous_noise(telemetry["air_temp"],       "float"),
        "rainfall":      inject_anomalous_noise(telemetry["rainfall"],       "bool"),
    }


def corrupt_telemetry(telemetry, r):
    """Apply noise at reliability level r to all telemetry fields."""
    if r == 1.0:
        return telemetry.copy()

    return {
        "tyre_age":      inject_noise(telemetry["tyre_age"],      "int",      r),
        "compound":      inject_noise(telemetry["compound"],       "compound", r),
        "position":      inject_noise(telemetry["position"],       "int",      r),
        "gap_to_leader": inject_noise(telemetry["gap_to_leader"],  "float",    r),
        "deg_rate":      inject_noise(telemetry["deg_rate"],       "float",    r),
        "air_temp":      inject_noise(telemetry["air_temp"],       "float",    r),
        "rainfall":      inject_noise(telemetry["rainfall"],       "bool",     r),
    }


# ── MCP Tools ────────────────────────────────────────────────────────────────

def get_tyre_age(telemetry, r=1.0):
    """Return tyre age in laps, optionally corrupted."""
    return inject_noise(telemetry["tyre_age"], "int", r)

def get_gap(telemetry, r=1.0):
    """Return gap to race leader in seconds."""
    return inject_noise(telemetry["gap_to_leader"], "float", r)

def get_deg_rate(telemetry, r=1.0):
    """Return lap time degradation rate in seconds/lap."""
    return inject_noise(telemetry["deg_rate"], "float", r)

def get_weather(telemetry, r=1.0):
    """Return weather context: air temp and rainfall."""
    return {
        "air_temp": inject_noise(telemetry["air_temp"], "float", r),
        "rainfall": inject_noise(telemetry["rainfall"], "bool", r),
    }

def get_compound(telemetry, r=1.0):
    """Return current tyre compound."""
    return inject_noise(telemetry["compound"], "compound", r)

def get_pit_window(telemetry, r=1.0):
    """
    Compute a strategic pit recommendation based on telemetry.
    Returns: dict with 'recommended' (PIT/STAY) and 'confidence' (high/medium/low)
    This tool can also be corrupted at reliability r.
    """
    tyre_age = inject_noise(telemetry["tyre_age"], "int", r)
    deg_rate = inject_noise(telemetry["deg_rate"], "float", r)
    gap = inject_noise(telemetry["gap_to_leader"], "float", r)
    compound = inject_noise(telemetry["compound"], "compound", r)

    # Domain rules for pit recommendation
    max_age = {"SOFT": 20, "MEDIUM": 32, "HARD": 42}.get(compound, 30)
    age_score = tyre_age / max_age          # >1.0 means overdue
    deg_score = max(0, deg_rate) / 0.15    # normalized degradation
    gap_ok = gap > 3.0                     # enough gap to pit safely

    score = 0.5 * age_score + 0.5 * deg_score

    if score > 0.85 and gap_ok:
        recommended, confidence = "PIT", "high"
    elif score > 0.65 and gap_ok:
        recommended, confidence = "PIT", "medium"
    elif score > 0.85 and not gap_ok:
        recommended, confidence = "PIT", "low"  # needs to pit but risky
    else:
        recommended, confidence = "STAY", "high" if score < 0.4 else "medium"

    return {
        "recommended": recommended,
        "confidence": confidence,
        "tyre_life_pct": round(tyre_age / max_age * 100, 1),
        "deg_normalized": round(deg_score, 3)
    }

def _field_type(field):
    types = {
        "tyre_age": "int",
        "compound": "compound",
        "position": "int",
        "gap_to_leader": "float",
        "deg_rate": "float",
        "air_temp": "float",
        "rainfall": "bool",
        "pit_window": "float",
    }
    return types.get(field, "float")
    

# ── Snapshot serving ─────────────────────────────────────────────────────────

def serve_snapshot(snapshot, r=1.0, noise_type="plausible", ablation_field=None):
    tel = snapshot["telemetry"]

    if ablation_field is not None:
        # only corrupt the specified field, everything else clean
        corrupted = tel.copy()
        if noise_type == "anomalous":
            corrupted[ablation_field] = inject_anomalous_noise(
                tel[ablation_field], _field_type(ablation_field))
        else:
            corrupted[ablation_field] = inject_noise(
                tel[ablation_field], _field_type(ablation_field), r)
    elif noise_type == "anomalous" and r < 1.0:
        corrupted = corrupt_telemetry_anomalous(tel)
    else:
        corrupted = corrupt_telemetry(tel, r)

    pit_window = get_pit_window(tel, r if noise_type == "plausible" else 0.4)

    return {
        "tyre_age":      corrupted["tyre_age"],
        "compound":      corrupted["compound"],
        "position":      corrupted["position"],
        "gap_to_leader": corrupted["gap_to_leader"],
        "deg_rate":      corrupted["deg_rate"],
        "air_temp":      corrupted.get("air_temp", tel["air_temp"]),
        "rainfall":      corrupted.get("rainfall", tel["rainfall"]),
        "pit_window":    pit_window,
        "lap":           snapshot["lap"],
        "total_laps":    snapshot["total_laps"],
        "race":          snapshot["race"],
        "year":          snapshot["year"],
    }


if __name__ == "__main__":
    # quick smoke test
    with open("data/snapshots/snapshots_balanced.jsonl") as f:
        snap = json.loads(f.readline())

    print("=== Clean (r=1.0) ===")
    print(json.dumps(serve_snapshot(snap, r=1.0), indent=2))

    print("\n=== Noisy (r=0.4) ===")
    print(json.dumps(serve_snapshot(snap, r=0.4), indent=2))