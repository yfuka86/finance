# fins/summary cache

## Overview
J-Quants `/v2/fins/summary` API responses cached as gzip-compressed JSON.
Original file `fins_20260417.json` (112MB) is split into parts to stay within GitHub's 100MB limit.

## Files
- `fins_20260424_part00.json.gz` ~ `fins_20260424_part05.json.gz`
  - 2974 stock codes, ~496 codes per part
  - Each part is a JSON object: `{ "code": [records...], ... }`

## Restore (reconstruct fins_YYYYMMDD.json)

```python
import json, gzip, glob

# Auto-detect latest date
parts = sorted(glob.glob("data/cache/fins_*_part00.json.gz"))
date = parts[-1].split("fins_")[1].split("_part")[0]

merged = {}
for path in sorted(glob.glob(f"data/cache/fins_{date}_part*.json.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        merged.update(json.load(f))

with open(f"data/cache/fins_{date}.json", "w") as f:
    json.dump(merged, f, ensure_ascii=False, indent=1)

print(f"Restored {len(merged)} codes")
```

## Usage by screener
The screener's `_load_fins_cache()` automatically restores from `.gz` parts
if the full JSON doesn't exist. No manual restore needed.
