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


# ── Snapshot serving ─────────────────────────────────────────────────────────

def serve_snapshot(snapshot, r=1.0):
    """
    Simulate MCP tool calls for a snapshot at reliability r.
    Returns corrupted telemetry ready for prompt injection.
    """
    tel = snapshot["telemetry"]
    return {
        "tyre_age":      get_tyre_age(tel, r),
        "compound":      get_compound(tel, r),
        "position":      inject_noise(tel["position"], "int", r),
        "gap_to_leader": get_gap(tel, r),
        "deg_rate":      get_deg_rate(tel, r),
        **get_weather(tel, r),
        # metadata (never corrupted)
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