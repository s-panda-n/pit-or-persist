def zero_shot_prompt(tel):
    pw = tel['pit_window']
    return f"""You are an F1 race strategist. Based on the telemetry and strategic analysis below, decide whether the driver should PIT this lap or STAY OUT.

Race: {tel['race']} {tel['year']}
Lap: {tel['lap']} / {tel['total_laps']}
Tyre compound: {tel['compound']}
Tyre age: {tel['tyre_age']} laps
Position: P{tel['position']}
Gap to leader: {tel['gap_to_leader']:.2f}s
Lap time degradation: {tel['deg_rate']:+.3f}s/lap
Air temperature: {tel['air_temp']:.1f}C
Rainfall: {'Yes' if tel['rainfall'] else 'No'}

Strategic tool recommendation: {pw['recommended']} (confidence: {pw['confidence']})
Tyre life used: {pw['tyre_life_pct']}%

Respond with exactly one word: PIT or STAY."""


def cot_prompt(tel):
    pw = tel['pit_window']
    return f"""You are an F1 race strategist. Based on the telemetry and strategic analysis below, decide whether the driver should PIT this lap or STAY OUT.

Race: {tel['race']} {tel['year']}
Lap: {tel['lap']} / {tel['total_laps']}
Tyre compound: {tel['compound']}
Tyre age: {tel['tyre_age']} laps
Position: P{tel['position']}
Gap to leader: {tel['gap_to_leader']:.2f}s
Lap time degradation: {tel['deg_rate']:+.3f}s/lap
Air temperature: {tel['air_temp']:.1f}C
Rainfall: {'Yes' if tel['rainfall'] else 'No'}

Strategic tool recommendation: {pw['recommended']} (confidence: {pw['confidence']})
Tyre life used: {pw['tyre_life_pct']}%

Note: Most laps do NOT involve a pit stop. Consider whether the evidence and tool recommendation strongly support pitting now.

Think step by step about tyre condition, track position, tool recommendation, and race strategy. Then on the final line write exactly: DECISION: PIT or DECISION: STAY."""