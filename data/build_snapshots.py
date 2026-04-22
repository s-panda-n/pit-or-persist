import fastf1
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

fastf1.Cache.enable_cache('data/raw/cache')

RACES = [
    (2022, 'Bahrain'), (2022, 'Australia'), (2022, 'Spain'),
    (2022, 'Monaco'), (2022, 'Italy'),
    (2023, 'Bahrain'), (2023, 'Australia'), (2023, 'Spain'),
    (2023, 'Monaco'), (2023, 'Italy'),
]

def compute_deg_rate(driver_laps, current_lap_num, window=3):
    """Lap time degradation over last N laps in seconds/lap."""
    recent = driver_laps[driver_laps['LapNumber'] < current_lap_num].tail(window)
    if len(recent) < 2:
        return None
    times = recent['LapTime'].dt.total_seconds()
    if times.isna().any():
        return None
    diffs = times.diff().dropna()
    return round(float(diffs.mean()), 4)

def get_gap_to_leader(laps, driver, lap_num):
    """Gap in seconds to the race leader at this lap."""
    lap_row = laps[(laps['Driver'] == driver) & (laps['LapNumber'] == lap_num)]
    if lap_row.empty:
        return None
    pos = lap_row.iloc[0]['Position']
    if pd.isna(pos) or pos == 1:
        return 0.0
    # find leader lap time on same lap
    leader_lap = laps[(laps['LapNumber'] == lap_num) & (laps['Position'] == 1)]
    if leader_lap.empty:
        return None
    driver_time = lap_row.iloc[0]['LapTime']
    leader_time = leader_lap.iloc[0]['LapTime']
    if pd.isna(driver_time) or pd.isna(leader_time):
        return None
    return round((driver_time - leader_time).total_seconds(), 3)

def build_snapshots_for_session(year, race_name):
    print(f"Processing {year} {race_name}...")
    session = fastf1.get_session(year, race_name, 'R')
    session.load(weather=True)
    laps = session.laps
    weather = session.weather_data
    snapshots = []

    # get avg weather (simplification)
    avg_temp = round(float(weather['AirTemp'].mean()), 1) if weather is not None else 25.0
    avg_rain = bool(weather['Rainfall'].any()) if weather is not None else False

    drivers = laps['Driver'].unique()

    for driver in drivers:
        driver_laps = laps[laps['Driver'] == driver].copy()
        driver_laps = driver_laps.sort_values('LapNumber')

        # find pit laps for ground truth
        pit_lap_nums = set(
            driver_laps[driver_laps['PitInTime'].notna()]['LapNumber'].tolist()
        )

        total_laps = int(driver_laps['LapNumber'].max())

        for _, row in driver_laps.iterrows():
            lap_num = row['LapNumber']
            if pd.isna(lap_num):
                continue
            lap_num = int(lap_num)

            # skip first 3 laps (no deg rate) and last 3 (no decision to make)
            if lap_num < 4 or lap_num > total_laps - 3:
                continue

            tyre_life = row['TyreLife']
            compound = row['Compound']
            position = row['Position']

            if pd.isna(tyre_life) or pd.isna(compound) or pd.isna(position):
                continue

            deg_rate = compute_deg_rate(driver_laps, lap_num)
            if deg_rate is None:
                continue

            gap = get_gap_to_leader(laps, driver, lap_num)
            if gap is None:
                continue

            # ground truth: did they pit THIS lap?
            label = 1 if lap_num in pit_lap_nums else 0

            snapshot = {
                "id": f"{year}_{race_name}_{driver}_lap{lap_num}",
                "year": year,
                "race": race_name,
                "driver": driver,
                "lap": lap_num,
                "total_laps": total_laps,
                "telemetry": {
                    "tyre_age": int(tyre_life),
                    "compound": str(compound),
                    "position": int(position),
                    "gap_to_leader": float(gap),
                    "deg_rate": float(deg_rate),
                    "air_temp": avg_temp,
                    "rainfall": avg_rain,
                },
                "label": label  # 1 = pitted, 0 = stayed out
            }
            snapshots.append(snapshot)

    return snapshots

def main():
    all_snapshots = []
    for year, race in RACES:
        try:
            snaps = build_snapshots_for_session(year, race)
            all_snapshots.extend(snaps)
            print(f"  -> {len(snaps)} snapshots")
        except Exception as e:
            print(f"  ERROR: {e}")

    # class balance check
    labels = [s['label'] for s in all_snapshots]
    n_pit = sum(labels)
    n_stay = len(labels) - n_pit
    print(f"\nTotal snapshots: {len(all_snapshots)}")
    print(f"Pit: {n_pit} ({100*n_pit/len(labels):.1f}%) | Stay: {n_stay} ({100*n_stay/len(labels):.1f}%)")

    out_path = Path('data/snapshots/snapshots.jsonl')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        for snap in all_snapshots:
            f.write(json.dumps(snap) + '\n')
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()