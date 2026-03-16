"""Fills in detailed author info (name, affiliation, email) by sending each paper's
thumbnail (default) or PDF first page to GPT-5.4 for extraction.

Usage:
    # Thumbnail mode (default — faster, no arxiv rate limits):
    python scripts/use_gpt_to_fill_detailed_author_info.py --input data/hf_daily_papers.jsonl

    # PDF mode (higher quality, slower due to arxiv rate limits):
    python scripts/use_gpt_to_fill_detailed_author_info.py --input data/hf_daily_papers.jsonl --source pdf

    # From the HuggingFace dataset (pushes to hub after each batch):
    python scripts/use_gpt_to_fill_detailed_author_info.py --hf_dataset justinxzhao/hf_daily_papers
"""

import argparse
import asyncio
import os

import aiohttp
import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from hf_daily_papers_analytics.hf_papers_scraper import (
    extract_author_info_from_pdf,
    extract_author_info_from_thumbnail,
    get_pdf_bytes,
)

load_dotenv()

# arXiv's documented rate limit is 1 request per 3 seconds.
# See: https://info.arxiv.org/help/api/tou.html
ARXIV_DELAY_SECONDS = 3

BATCH_SIZE = 10

# Concurrency limits by source type
PDF_CONCURRENCY = 3       # Limited by arxiv rate limits
THUMBNAIL_CONCURRENCY = 20  # Only limited by OpenAI rate limits

# Exponential backoff settings for OpenAI API rate limits
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 60.0     # seconds
BACKOFF_FACTOR = 2.0


async def fetch_author_info_thumbnail(paper_id, thumbnail_url, session, semaphore):
    """Fetches author information using the HF thumbnail image."""
    async with semaphore:
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(thumbnail_url) as resp:
                    if resp.status != 200:
                        raise ValueError(
                            f"Failed to fetch thumbnail from {thumbnail_url}, "
                            f"status code: {resp.status}"
                        )
                    image_bytes = await resp.read()

                author_info = await extract_author_info_from_thumbnail(image_bytes)
                return paper_id, [
                    {
                        "name": author.name,
                        "affiliation": author.affiliation,
                        "email": author.email,
                    }
                    for author in author_info
                ]
            except Exception as e:
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                print(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {paper_id} "
                    f"(thumbnail): {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    wait = min(backoff, MAX_BACKOFF)
                    if is_rate_limit:
                        print(f"  Rate limited — backing off {wait:.1f}s")
                    await asyncio.sleep(wait)
                    backoff *= BACKOFF_FACTOR
                else:
                    print(f"All retries failed for {paper_id} (thumbnail).")
                    return paper_id, []


async def fetch_author_info_pdf(paper_id, pdf_link, session, semaphore):
    """Fetches author information using the arxiv PDF first page."""
    async with semaphore:
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(ARXIV_DELAY_SECONDS)

                pdf_bytes = await get_pdf_bytes(pdf_link, session)
                author_info = await extract_author_info_from_pdf(pdf_bytes)

                return paper_id, [
                    {
                        "name": author.name,
                        "affiliation": author.affiliation,
                        "email": author.email,
                    }
                    for author in author_info
                ]
            except Exception as e:
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                print(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {paper_id} "
                    f"(pdf): {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    wait = min(backoff, MAX_BACKOFF)
                    if is_rate_limit:
                        print(f"  Rate limited — backing off {wait:.1f}s")
                    await asyncio.sleep(wait)
                    backoff *= BACKOFF_FACTOR
                else:
                    print(f"All retries failed for {paper_id} (pdf).")
                    return paper_id, []


def update_df_with_author_info(df, author_info_map, cumulative_map):
    """Merges new results into cumulative_map, then rebuilds the author_info column.

    Uses .map() on paper_id to assign lists cleanly, avoiding pandas cell-mutation
    issues with list values. Returns number of new papers updated in this batch.
    """
    new_updates = 0
    for paper_id, author_info in author_info_map.items():
        if not author_info or paper_id in cumulative_map:
            continue
        cumulative_map[paper_id] = author_info
        new_updates += 1

    # Rebuild the column: use the cumulative map, falling back to existing values
    df["author_info"] = df["paper_id"].map(cumulative_map)
    return new_updates


def save_checkpoint(df, output_path=None, hf_dataset_name=None):
    """Saves progress either to a local file or pushes to HuggingFace Hub."""
    if output_path:
        df.to_json(output_path, orient="records", lines=True)
        print(f"Saved to {output_path}")
    elif hf_dataset_name:
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset.push_to_hub(
            hf_dataset_name,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )
        print(f"Pushed to {hf_dataset_name}")


async def process_batch(batch_items, session, source, semaphore):
    """Processes a batch of papers and returns the results."""
    if source == "thumbnail":
        tasks = [
            fetch_author_info_thumbnail(paper_id, url, session, semaphore)
            for paper_id, url in batch_items
        ]
    else:
        tasks = [
            fetch_author_info_pdf(paper_id, url, session, semaphore)
            for paper_id, url in batch_items
        ]
    results = await tqdm.gather(*tasks, desc="Processing batch")
    return {paper_id: author_info for paper_id, author_info in results}


async def run(paper_url_map, df, source, output_path=None, hf_dataset_name=None):
    """Processes papers in batches, saving after each batch."""
    items = list(paper_url_map.items())
    total_updated = 0
    cumulative_map = {}  # paper_id -> author_info, accumulated across batches

    concurrency = THUMBNAIL_CONCURRENCY if source == "thumbnail" else PDF_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; JustinsArxivBot/1.0)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} papers) ---")

            author_info_map = await process_batch(batch, session, source, semaphore)
            updated = update_df_with_author_info(df, author_info_map, cumulative_map)
            total_updated += updated
            print(f"Updated {updated} papers in this batch ({total_updated} total).")

            save_checkpoint(
                df, output_path=output_path, hf_dataset_name=hf_dataset_name
            )

    print(f"\nDone. Updated {total_updated} papers total.")


def get_papers_needing_author_info(df, source):
    """Returns a dict of paper_id -> url for papers missing author_info.

    When source='thumbnail', returns thumbnail URLs from the HF CDN.
    When source='pdf', returns PDF URLs from export.arxiv.org.
    """
    if "author_info" not in df.columns:
        mask = pd.Series(True, index=df.index)
    else:
        mask = df["author_info"].isna() | df["author_info"].apply(
            lambda x: isinstance(x, list) and len(x) == 0
        )

    paper_ids = df.loc[mask, "paper_id"]

    if source == "thumbnail":
        urls = df.loc[mask, "thumbnail"]
        # Filter out papers with no thumbnail URL
        valid = urls.notna() & (urls != "")
        if (~valid).any():
            n_missing = (~valid).sum()
            print(
                f"Warning: {n_missing} papers have no thumbnail URL and will be skipped."
            )
            paper_ids = paper_ids[valid]
            urls = urls[valid]
        return dict(zip(paper_ids, urls))
    else:
        # Use export.arxiv.org for programmatic access
        pdf_links = df.loc[mask, "pdf_link"].str.replace(
            "://arxiv.org/", "://export.arxiv.org/", regex=False
        )
        return dict(zip(paper_ids, pdf_links))


def main():
    parser = argparse.ArgumentParser(
        description="Fill detailed author info using GPT-5.4 extraction."
    )
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument(
        "--input",
        type=str,
        help="Path to a local JSONL file (reads from and saves back to this file).",
    )
    data_source.add_argument(
        "--hf_dataset",
        type=str,
        help="HuggingFace dataset name (e.g. justinxzhao/hf_daily_papers).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["thumbnail", "pdf"],
        default="thumbnail",
        help="Extraction source: 'thumbnail' (default, faster) or 'pdf' (higher quality).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    # Load data
    if args.input:
        df = pd.read_json(args.input, lines=True)
        output_path = args.input
        hf_dataset_name = None
    else:
        dataset = load_dataset(args.hf_dataset)
        df = pd.DataFrame(dataset["train"])
        output_path = None
        hf_dataset_name = args.hf_dataset

    paper_url_map = get_papers_needing_author_info(df, args.source)

    num_papers = len(paper_url_map)
    print(f"Source: {args.source}")
    print(f"Number of papers to process: {num_papers}")
    print(f"Will save every {BATCH_SIZE} papers.")
    if args.source == "thumbnail":
        print(f"Concurrency: {THUMBNAIL_CONCURRENCY} (thumbnail mode)")
    else:
        print(f"Concurrency: {PDF_CONCURRENCY} (PDF mode, arxiv rate limited)")

    if num_papers == 0:
        print("No papers left to process. Exiting.")
        return

    if not args.yes:
        confirm = (
            input("Do you want to proceed? (yes/no) [default: yes]: ").strip().lower()
        )
        if confirm not in ("", "yes"):
            print("Operation cancelled by the user.")
            return

    asyncio.run(
        run(
            paper_url_map,
            df,
            args.source,
            output_path=output_path,
            hf_dataset_name=hf_dataset_name,
        )
    )


if __name__ == "__main__":
    main()
