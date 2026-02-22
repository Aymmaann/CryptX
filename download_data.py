"""
download_binance_data.py

Downloads BTCUSDT 1-minute kline CSV files from Binance public data repository.
Binance hosts monthly CSVs at:
  https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/

Usage:
    python download_binance_data.py

Downloads Feb 2024 → Dec 2025 by default (configurable below).
Saves to:  CryptX/data/raw/BTCUSDT/1m/
"""

import requests
import os
import zipfile
import io
from pathlib import Path
from datetime import datetime, date
import time

# ============================================================
# CONFIGURATION — edit these
# ============================================================
SYMBOL       = "BTCUSDT"
INTERVAL     = "1m"
START_YEAR   = 2024
START_MONTH  = 2      # February 2024
END_YEAR     = 2025
END_MONTH    = 12     # December 2025  (your existing Jul-Dec 2025 will be re-downloaded/skipped)

BASE_URL     = "https://data.binance.vision/data/spot/monthly/klines"
OUTPUT_DIR   = Path("/Users/ayman/Documents/CryptX/data")/ "raw" / "binance" / SYMBOL / INTERVAL
# ============================================================


def month_range(start_year, start_month, end_year, end_month):
    """Yield (year, month) tuples inclusive."""
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def download_month(year: int, month: int, output_dir: Path, skip_existing=True) -> bool:
    """
    Download one month of BTCUSDT 1m data from Binance.
    Returns True on success, False on failure.
    """
    filename  = f"{SYMBOL}-{INTERVAL}-{year}-{month:02d}.csv"
    zip_name  = filename.replace(".csv", ".zip")
    out_path  = output_dir / filename

    if skip_existing and out_path.exists():
        print(f"  [SKIP] {filename} already exists")
        return True

    url = f"{BASE_URL}/{SYMBOL}/{INTERVAL}/{zip_name}"
    print(f"  [DOWNLOAD] {zip_name} ...")

    try:
        resp = requests.get(url, timeout=60)

        if resp.status_code == 404:
            print(f"  [MISSING] {zip_name} not available on Binance (future month?)")
            return False

        resp.raise_for_status()

        # Unzip in memory and save just the CSV
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                print(f"  [ERROR] No CSV found inside {zip_name}")
                return False
            with zf.open(csv_names[0]) as f:
                data = f.read()

        out_path.write_bytes(data)
        size_mb = len(data) / 1_048_576
        print(f"  [OK] Saved {filename} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to download {zip_name}: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"Binance Data Downloader")
    print(f"Symbol:   {SYMBOL}  |  Interval: {INTERVAL}")
    print(f"Range:    {START_YEAR}-{START_MONTH:02d} → {END_YEAR}-{END_MONTH:02d}")
    print(f"Output:   {OUTPUT_DIR}")
    print("=" * 60)

    months = list(month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH))
    print(f"Total months to download: {len(months)}\n")

    success, failed = [], []

    for year, month in months:
        ok = download_month(year, month, OUTPUT_DIR)
        if ok:
            success.append((year, month))
        else:
            failed.append((year, month))
        time.sleep(0.3)   # be polite to Binance servers

    print("\n" + "=" * 60)
    print(f"✓ Downloaded:  {len(success)} months")
    print(f"✗ Failed/Missing: {len(failed)} months")
    if failed:
        print("  Missing months:", [f"{y}-{m:02d}" for y, m in failed])
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("\nNext step: run your data ingestion pipeline to process these CSVs into parquet.")


if __name__ == "__main__":
    main()