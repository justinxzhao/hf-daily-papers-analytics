"""Script to fetch daily papers from Hugging Face API and update the dataset on Hugging Face Hub.

Performs a full scrape of all papers, merges with the existing dataset (preserving
previously extracted author_info), fills author_info for recent papers via thumbnail
extraction, and optionally uploads to HF Hub.

Usage:
    # Full scrape + author info + upload (what the daily action runs):
    python scripts/update_hf_datasets.py --upload

    # Local test (no upload, no author info):
    python scripts/update_hf_datasets.py --skip_author_info

    # Custom lookback for author info:
    python scripts/update_hf_datasets.py --author_info_days 14 --upload
"""

import argparse
import asyncio
import os
from datetime import datetime, timedelta

import aiohttp
import dotenv
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.asyncio import tqdm

from hf_daily_papers_analytics.hf_papers_scraper import (
    extract_author_info_from_thumbnail,
    run_scraper,
)
from hf_daily_papers_analytics.utils import merge_datasets

dotenv.load_dotenv()

DATASET_NAME = "justinxzhao/hf_daily_papers"
FIRST_DATE = "2023-05-04"

# Author info extraction settings
THUMBNAIL_CONCURRENCY = 20
BATCH_SIZE = 10
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0
MAX_BACKOFF = 120.0
BACKOFF_FACTOR = 2.0


def _is_retryable_error(e: Exception) -> bool:
    """Check if an error is retryable (rate limit, server error, timeout)."""
    msg = str(e).lower()
    return any(s in msg for s in ["rate", "429", "500", "502", "503", "504", "timeout", "gateway"])


def download_hf_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset["train"])
    return df


def upload_to_hf(dataset, dataset_name, token):
    dataset_dict = DatasetDict({"train": Dataset.from_pandas(dataset)})
    dataset_dict.push_to_hub(dataset_name, token=token)


async def fetch_author_info_thumbnail(paper_id, thumbnail_url, session, semaphore):
    """Fetches author information using the HF thumbnail image with exponential backoff."""
    async with semaphore:
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(thumbnail_url) as resp:
                    if resp.status != 200:
                        raise ValueError(
                            f"Failed to fetch thumbnail, status {resp.status}"
                        )
                    image_bytes = await resp.read()

                author_info = await extract_author_info_from_thumbnail(image_bytes)
                return paper_id, [
                    {
                        "name": a.name,
                        "affiliation": a.affiliation,
                        "email": a.email,
                    }
                    for a in author_info
                ]
            except Exception as e:
                retryable = _is_retryable_error(e)
                print(
                    f"  Attempt {attempt + 1}/{MAX_RETRIES} failed for {paper_id}: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    wait = min(backoff, MAX_BACKOFF)
                    if retryable:
                        print(f"    Retryable error — backing off {wait:.1f}s")
                    await asyncio.sleep(wait)
                    backoff *= BACKOFF_FACTOR
                else:
                    print(f"  All retries failed for {paper_id}.")
                    return paper_id, []


def _count_with_author_info(df):
    """Returns the number of rows that have non-empty author_info."""
    if "author_info" not in df.columns:
        return 0
    return df["author_info"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    ).sum()


async def fill_author_info(df, days):
    """Fills author_info for recent papers that are missing it. Returns (df, num_filled)."""
    cutoff = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Find recent papers with missing author_info
    recent = df[df["date"] >= cutoff].copy()
    missing_mask = recent["author_info"].isna() | recent["author_info"].apply(
        lambda x: not isinstance(x, list) or len(x) == 0
    )
    needs_info = recent[missing_mask]

    # Filter to papers with valid thumbnail URLs
    has_thumb = needs_info["thumbnail"].notna() & (needs_info["thumbnail"] != "")
    to_process = needs_info[has_thumb]

    if to_process.empty:
        print("No papers need author info extraction.")
        return df, 0

    n_skipped = len(needs_info) - len(to_process)
    if n_skipped > 0:
        print(f"  Skipping {n_skipped} papers with no thumbnail URL.")

    print(f"Extracting author info for {len(to_process)} papers (last {days} days)...")

    semaphore = asyncio.Semaphore(THUMBNAIL_CONCURRENCY)
    items = list(zip(to_process["paper_id"], to_process["thumbnail"]))
    num_filled = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_author_info_thumbnail(pid, url, session, semaphore)
            for pid, url in items
        ]
        results = await tqdm.gather(*tasks, desc="  Extracting")

        for paper_id, author_info in results:
            if author_info:
                idxs = df.index[df["paper_id"] == paper_id]
                for idx in idxs:
                    df.at[idx, "author_info"] = author_info
                num_filled += 1

    filled = df[df["date"] >= cutoff]["author_info"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    ).sum()
    total_recent = len(df[df["date"] >= cutoff])
    print(f"Author info coverage for last {days} days: {filled}/{total_recent}")

    return df, num_filled


async def main(args):
    hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = FIRST_DATE

    print("=" * 60)
    print("HF Daily Papers — Full Scrape + Author Info Pipeline")
    print("=" * 60)

    # Step 1: Full scrape from HF API
    print(f"\n[Step 1/4] Fetching all papers from {start_date} to {end_date}...")
    new_df = await run_scraper(start_date, end_date, output_file=None)
    new_dates = sorted(new_df["date"].unique()) if not new_df.empty else []
    print(f"  Scraped {len(new_df)} papers across {len(new_dates)} dates")
    if new_dates:
        print(f"  Date range: {new_dates[0]} to {new_dates[-1]}")

    # Step 2: Download existing dataset and merge (preserving author_info)
    print(f"\n[Step 2/4] Downloading existing dataset from Hugging Face...")
    try:
        existing_df = download_hf_dataset(DATASET_NAME)
        existing_size = len(existing_df)
        existing_with_info = _count_with_author_info(existing_df)
        print(f"  Existing dataset: {existing_size} papers, "
              f"{existing_with_info} with author_info")
    except Exception as e:
        print(f"  Could not fetch existing dataset ({e}), using fresh scrape only.")
        existing_df = pd.DataFrame()
        existing_size = 0
        existing_with_info = 0

    print(f"\n[Step 3/4] Merging datasets (preserving existing author_info)...")
    merged_df = merge_datasets(existing_df, new_df)
    merged_with_info = _count_with_author_info(merged_df)
    new_papers = len(merged_df) - existing_size
    print(f"  Merged dataset: {len(merged_df)} papers "
          f"({'+' if new_papers >= 0 else ''}{new_papers} net new)")
    print(f"  Author info preserved: {merged_with_info}/{len(merged_df)} papers")

    # Step 3: Fill author_info for recent papers
    num_newly_filled = 0
    if not args.skip_author_info:
        print(f"\n[Step 4/4] Filling author info for recent papers "
              f"(last {args.author_info_days} days)...")
        merged_df, num_newly_filled = await fill_author_info(
            merged_df, args.author_info_days
        )
    else:
        print(f"\n[Step 4/4] Skipping author info extraction (--skip_author_info)")

    # Summary
    final_with_info = _count_with_author_info(merged_df)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Existing dataset size:  {existing_size} papers")
    print(f"  Fresh scrape size:      {len(new_df)} papers")
    print(f"  Final dataset size:     {len(merged_df)} papers "
          f"({'+' if new_papers >= 0 else ''}{new_papers} net new)")
    print(f"  Author info before:     {existing_with_info}")
    print(f"  Author info after:      {final_with_info} "
          f"(+{num_newly_filled} newly extracted)")
    print(f"  Missing author info:    {len(merged_df) - final_with_info}")
    print("=" * 60)

    # Save / Upload
    if args.output:
        print(f"\nSaving to {args.output}...")
        merged_df.to_json(args.output, orient="records", lines=True)
        print(f"Saved {len(merged_df)} papers to {args.output}")
    if args.upload:
        print("\nUploading to Hugging Face Hub...")
        upload_to_hf(merged_df, DATASET_NAME, hf_token)
        print("Dataset successfully updated!")
    if not args.output and not args.upload:
        print("\nDry run — not saving. Use --output or --upload.")

    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full scrape of HF daily papers + author info enrichment."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload merged dataset to HF Hub.",
    )
    parser.add_argument(
        "--author_info_days",
        type=int,
        default=7,
        help="Lookback window (days) for filling missing author_info (default: 7).",
    )
    parser.add_argument(
        "--skip_author_info",
        action="store_true",
        help="Skip the author info extraction step.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save merged dataset to a local JSONL file.",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
