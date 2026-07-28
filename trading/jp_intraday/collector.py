from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def _to_jquants_codes(symbols: list[str]) -> set[str]:
    """J-Quants bulk files key by 5-digit codes (4-digit tickers gain a trailing 0)."""
    return {s if len(s) == 5 else f"{s}0" for s in map(str, symbols)}


def _strip_jquants_suffix(codes: pd.Series) -> pd.Series:
    """Reverse of the 4->5 digit padding: drop the trailing 0."""
    return codes.astype(str).str.removesuffix("0")


def _download_bulk_file(client, key: str, raw_path: Path, timeout: int) -> None:
    """Stream a J-Quants bulk file to disk if it is not already cached."""
    if raw_path.exists():
        return
    url = client.get_bulk(key=key)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with raw_path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)


def collect_jquants_minutes(
    symbols: list[str], start: str, end: str, output: str | Path,
    pause_seconds: float = 0.35, max_retries: int = 6,
) -> pd.DataFrame:
    """Collect official J-Quants minute bars with resumable per-symbol cache."""
    try:
        import jquantsapi
    except ImportError as exc:
        raise RuntimeError("install jquants-api-client from requirements.txt") from exc
    from data.collectors.config import JQUANTS_API_KEY
    key = JQUANTS_API_KEY
    if not key:
        raise RuntimeError("JQUANTS_API_KEY environment variable is required")
    output = Path(output)
    cache = output / "by_symbol"
    cache.mkdir(parents=True, exist_ok=True)
    client = jquantsapi.ClientV2(api_key=key)
    frames = []
    for index, symbol in enumerate(symbols, start=1):
        code = str(symbol).replace(".T", "")
        target = cache / f"{code}_{start}_{end}.parquet"
        if target.exists():
            frame = pd.read_parquet(target)
        else:
            for attempt in range(max_retries):
                try:
                    frame = client.get_eq_bars_minute(
                        code=code, from_yyyymmdd=start.replace("-", ""),
                        to_yyyymmdd=end.replace("-", ""),
                    )
                    break
                except Exception as exc:
                    if attempt + 1 == max_retries:
                        raise RuntimeError(f"minute download failed for {code}") from exc
                    time.sleep(min(2 ** attempt, 30))
            if frame.empty:
                continue
            frame.to_parquet(target, index=False)
            time.sleep(pause_seconds)
        frames.append(frame)
        if index % 10 == 0 or index == len(symbols):
            print(f"minute collection: {index}/{len(symbols)} symbols", flush=True)
    if not frames:
        raise ValueError("no minute bars returned; check add-on entitlement and date range")
    raw = pd.concat(frames, ignore_index=True)
    rename = {"Code": "symbol", "O": "open", "H": "high", "L": "low",
              "C": "close", "Vo": "volume"}
    raw = raw.rename(columns=rename)
    if "timestamp" not in raw:
        date_col = "Date" if "Date" in raw else "date"
        time_col = next((c for c in ("Time", "time", "Minute") if c in raw), None)
        if not time_col:
            raise ValueError(f"unknown minute-bar time columns: {list(raw.columns)}")
        raw["timestamp"] = pd.to_datetime(
            raw[date_col].astype(str) + " " + raw[time_col].astype(str)
        ).dt.tz_localize("Asia/Tokyo")
    canonical = raw[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    target = output / f"jp_1m_{start}_{end}.parquet"
    canonical.to_parquet(target, index=False)
    return canonical


def collect_jquants_minutes_bulk(
    symbols: list[str], start: str, end: str, output: str | Path,
) -> pd.DataFrame:
    """Download official bulk minute CSVs and retain only requested symbols."""
    try:
        import jquantsapi
        from jquantsapi.enums import BulkEndpoint
    except ImportError as exc:
        raise RuntimeError("install jquants-api-client from requirements.txt") from exc
    from data.collectors.config import JQUANTS_API_KEY
    if not JQUANTS_API_KEY:
        raise RuntimeError("JQUANTS_API_KEY is required")
    output = Path(output)
    raw_dir = output / "bulk_raw"
    filtered_dir = output / "bulk_filtered"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    client = jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    listing = client.get_bulk_list(
        endpoint=BulkEndpoint.EQ_BARS_MINUTE,
        from_date=start.replace("-", ""), to_date=end.replace("-", ""),
    )
    if listing.empty:
        raise ValueError("no bulk minute files available for the date range")
    wanted = _to_jquants_codes(symbols)
    pieces = []
    for index, key in enumerate(listing["Key"], start=1):
        name = Path(key).name
        raw_path = raw_dir / name
        filtered_path = filtered_dir / name.replace(".csv.gz", ".parquet")
        if not filtered_path.exists():
            _download_bulk_file(client, key, raw_path, timeout=120)
            selected = []
            for chunk in pd.read_csv(raw_path, compression="gzip", dtype={"Code": str},
                                     chunksize=500_000):
                code = chunk["Code"].astype(str)
                selected.append(chunk.loc[code.isin(wanted)])
            filtered = pd.concat(selected, ignore_index=True)
            filtered.to_parquet(filtered_path, index=False)
        pieces.append(pd.read_parquet(filtered_path))
        print(f"bulk minute collection: {index}/{len(listing)} files", flush=True)
    raw = pd.concat(pieces, ignore_index=True)
    raw = raw.loc[raw["Code"].astype(str).isin(wanted)].copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    keep_date = raw["Date"].between(pd.Timestamp(start), pd.Timestamp(end))
    raw = raw.loc[keep_date].copy()
    raw["timestamp"] = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str)
    ).dt.tz_localize("Asia/Tokyo")
    canonical = raw.rename(columns={
        "Code": "symbol", "O": "open", "H": "high", "L": "low",
        "C": "close", "Vo": "volume",
    })[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    canonical["symbol"] = _strip_jquants_suffix(canonical["symbol"])
    target = output / f"jp_1m_{start}_{end}.parquet"
    canonical.to_parquet(target, index=False)
    return canonical


def collect_tick_imbalance_bulk(
    symbols: list[str], start: str, end: str, output: str | Path,
) -> pd.DataFrame:
    """Stream J-Quants trades and aggregate tick-rule order flow to five minutes."""
    import jquantsapi
    from jquantsapi.enums import BulkEndpoint
    from data.collectors.config import JQUANTS_API_KEY
    output = Path(output)
    raw_dir, aggregate_dir = output / "bulk_raw", output / "aggregated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    wanted = _to_jquants_codes(symbols)
    client = jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    listing = client.get_bulk_list(
        endpoint=BulkEndpoint.EQ_TRADES,
        from_date=start.replace("-", ""), to_date=end.replace("-", ""),
    )
    pieces = []
    for file_index, key in enumerate(listing["Key"], start=1):
        name = Path(key).name
        raw_path, aggregate_path = raw_dir / name, aggregate_dir / name.replace(".csv.gz", ".parquet")
        if not aggregate_path.exists():
            _download_bulk_file(client, key, raw_path, timeout=180)
            partial, last_price, last_sign = [], {}, {}
            for chunk in pd.read_csv(raw_path, compression="gzip", dtype={"Code": str},
                                     chunksize=1_000_000):
                chunk = chunk[chunk["Code"].isin(wanted)].copy()
                if chunk.empty:
                    continue
                chunk["Price"] = pd.to_numeric(chunk["Price"])
                chunk["TradingVolume"] = pd.to_numeric(chunk["TradingVolume"])
                diff = chunk.groupby("Code")["Price"].diff()
                first = ~chunk["Code"].duplicated()
                prior_price = chunk.loc[first, "Code"].map(last_price)
                diff.loc[first] = chunk.loc[first, "Price"].sub(prior_price).to_numpy()
                sign = np.sign(diff).replace(0, np.nan)
                sign.loc[first] = sign.loc[first].fillna(chunk.loc[first, "Code"].map(last_sign))
                sign = sign.groupby(chunk["Code"]).ffill().fillna(0)
                chunk["signed_volume"] = sign * chunk["TradingVolume"]
                timestamp = pd.to_datetime(
                    chunk["Date"].astype(str) + " " + chunk["Time"].astype(str)
                ).dt.floor("5min")
                chunk["timestamp"] = timestamp
                grouped = chunk.groupby(["timestamp", "Code"], as_index=False).agg(
                    signed_volume=("signed_volume", "sum"),
                    traded_volume=("TradingVolume", "sum"),
                    trade_count=("TransactionId", "count"),
                )
                partial.append(grouped)
                tails = chunk.groupby("Code", sort=False).tail(1)
                last_price.update(dict(zip(tails["Code"], tails["Price"])))
                last_sign.update(dict(zip(tails["Code"], sign.loc[tails.index])))
            aggregate = pd.concat(partial, ignore_index=True).groupby(
                ["timestamp", "Code"], as_index=False
            ).sum()
            aggregate["order_flow_imbalance"] = aggregate["signed_volume"].div(
                aggregate["traded_volume"].replace(0, np.nan)
            )
            aggregate.to_parquet(aggregate_path, index=False)
        pieces.append(pd.read_parquet(aggregate_path))
        print(f"tick aggregation: {file_index}/{len(listing)} files", flush=True)
    result = pd.concat(pieces, ignore_index=True)
    result["symbol"] = _strip_jquants_suffix(result["Code"])
    result["timestamp"] = pd.to_datetime(result["timestamp"]).dt.tz_localize("Asia/Tokyo")
    result = result.drop(columns="Code").sort_values(["timestamp", "symbol"])
    result.to_parquet(output / f"tick_imbalance_5m_{start}_{end}.parquet", index=False)
    return result
