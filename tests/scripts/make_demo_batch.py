import json
import pandas as pd

CSV = "data/sensor.csv"
OUT = "samples/batch.json"

df = pd.read_csv(CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
df["machine_status"] = df["machine_status"].astype(str).str.upper()

def make_window(idx, k=30):
    start = max(0, idx - k + 1)
    w = df.iloc[start:idx+1]
    window = []
    for _, r in w.iterrows():
        item = {"ts": r["timestamp"].isoformat()}
        for c in sensor_cols:
            v = r[c]
            if pd.notna(v):
                item[c] = float(v)
        window.append(item)
    return window

def pick_index(status):
    sub = df.index[df["machine_status"] == status].to_list()
    if not sub:
        return None
    return sub[len(sub)//2]  # stable mid point

examples = []
for status in ["NORMAL", "RECOVERING", "BROKEN"]:
    idx = pick_index(status)
    if idx is None:
        continue
    examples.append({
        "asset_id": f"pump_{status.lower()}",
        "timestamp": df.loc[idx, "timestamp"].isoformat(),
        "sensor_window": make_window(idx, k=30)
    })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2)

print(f"Wrote {OUT} with {len(examples)} realistic examples.")
