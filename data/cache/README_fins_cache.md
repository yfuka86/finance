# fins/summary cache

## Overview
J-Quants `/v2/fins/summary` API responses cached as gzip-compressed JSON.
Original file `fins_20260417.json` (112MB) is split into parts to stay within GitHub's 100MB limit.

## Files
- `fins_20260417_part00.json.gz` ~ `fins_20260417_part05.json.gz`
  - 2927 stock codes, ~500 codes per part
  - Each part is a JSON object: `{ "code": [records...], ... }`

## Restore (reconstruct fins_YYYYMMDD.json)

```python
import json, gzip, glob

merged = {}
for path in sorted(glob.glob("data/cache/fins_20260417_part*.json.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        merged.update(json.load(f))

with open("data/cache/fins_20260417.json", "w") as f:
    json.dump(merged, f, ensure_ascii=False, indent=1)

print(f"Restored {len(merged)} codes")
```

## Usage by screener
The screener's `_load_fins_cache("20260417")` reads `fins_20260417.json`.
If only the `.gz` parts exist, run the restore script above first, or
the screener will re-fetch from J-Quants API (takes ~7 min for ~3000 codes).
