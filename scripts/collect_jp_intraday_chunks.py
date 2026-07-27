"""Parallel, resumable collection for the prepared TOPIX 100bn universe."""
from multiprocessing import Process
from pathlib import Path

import pandas as pd

from trading.jp_intraday.collector import collect_jquants_minutes


START = "2026-06-01"
END = "2026-07-24"
WORKERS = 3


def collect_chunk(index: int, symbols: list[str]) -> None:
    # Chunk zero reuses files collected by the initial serial run.
    output = Path("data/jp_minutes_100bn" if index == 0 else f"data/jp_minutes_100bn_chunk{index}")
    collect_jquants_minutes(symbols, START, END, output, pause_seconds=0.5)


def main() -> None:
    universe = pd.read_parquet(
        "data/jp_intraday_reference/universe_100bn_20260529_20260724.parquet"
    )
    symbols = sorted(universe["symbol"].unique())
    chunks = [symbols[index::WORKERS] for index in range(WORKERS)]
    processes = [Process(target=collect_chunk, args=(index, chunk))
                 for index, chunk in enumerate(chunks)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode]
    if failed:
        raise SystemExit(f"collection workers failed: {failed}")


if __name__ == "__main__":
    main()
