import fastf1
import pandas as pd

fastf1.Cache.enable_cache('data/raw/cache')

session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps
print("=== COLUMNS ===")
print(laps.columns.tolist())

print("\n=== SAMPLE LAPS ===")
print(laps[['LapNumber', 'Driver', 'Compound', 'TyreLife',
            'LapTime', 'PitInTime', 'PitOutTime']].head(30).to_string())

print("\n=== PIT LAPS ONLY ===")
pit_laps = laps[laps['PitInTime'].notna()]
print(pit_laps[['LapNumber', 'Driver', 'Compound', 'TyreLife']].to_string())