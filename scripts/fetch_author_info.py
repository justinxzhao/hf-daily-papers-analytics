import asyncio
import os
import random

import aiohttp
import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()

from hf_daily_papers_analytics.hf_papers_scraper import (
    extract_author_info_from_pdf,
    get_pdf_bytes,
)

# Limit concurrent requests to 2
semaphore = asyncio.Semaphore(2)

BATCH_SIZE = 50


async def fetch_author_info_for_paper(
    paper_id, pdf_link, session, retries=3, cooldown=2
):
    """Fetches author information for a single paper."""
    async with semaphore:
        for attempt in range(retries):
            try:
                # Add a randomized delay to respect rate limits
                await asyncio.sleep(random.uniform(5, 10))

                # Download the PDF and parse authors
                pdf_bytes = await get_pdf_bytes(pdf_link, session)
                author_info = await extract_author_info_from_pdf(pdf_bytes)

                # Convert to dictionaries
                return paper_id, [
                    {
                        "name": author.name,
                        "affiliation": author.affiliation,
                        "email": author.email,
                    }
                    for author in author_info
                ]
            except Exception as e:
                print(
                    f"Attempt {attempt + 1} failed for {pdf_link} (paper ID {paper_id}): {e}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(cooldown)
                else:
                    print(f"All retries failed for {pdf_link} (paper ID {paper_id}).")
                    return paper_id, []


def update_df_with_author_info(df, author_info_map):
    """Updates the DataFrame with fetched author info. Returns number of papers updated."""
    updated = 0
    for paper_id, author_info in author_info_map.items():
        if not author_info:
            continue
        existing_info = df.loc[df["paper_id"] == paper_id, "author_info"].values[0]
        if existing_info:
            continue
        df.at[df.index[df["paper_id"] == paper_id][0], "author_info"] = author_info
        updated += 1
    return updated


def push_to_hub(df):
    """Pushes the DataFrame to HuggingFace Hub."""
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(
        "justinxzhao/hf_daily_papers",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
    print("Dataset pushed to hub.")


async def process_batch(batch_items, session):
    """Processes a batch of papers and returns the results."""
    tasks = [
        fetch_author_info_for_paper(paper_id, pdf_link, session)
        for paper_id, pdf_link in batch_items
    ]
    results = await tqdm.gather(*tasks, desc="Processing batch")
    return {paper_id: author_info for paper_id, author_info in results}


async def run(paper_pdf_map, df):
    """Processes papers in batches, uploading after each batch."""
    items = list(paper_pdf_map.items())
    total_updated = 0

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; JustinsArxivBot/1.0; +https://yourdomain.com/contact)"
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} papers) ---")

            author_info_map = await process_batch(batch, session)
            updated = update_df_with_author_info(df, author_info_map)
            total_updated += updated
            print(f"Updated {updated} papers in this batch ({total_updated} total).")

            push_to_hub(df)

    print(f"\nDone. Updated {total_updated} papers total.")


def main():
    # Load the dataset
    dataset = load_dataset("justinxzhao/hf_daily_papers")

    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(dataset["train"])

    # Create a map of paper ID to PDF links
    if "author_info" in df.columns:
        paper_pdf_map = dict(
            zip(
                df.loc[
                    df["author_info"].isna()
                    | (df["author_info"].apply(lambda x: len(x) if x else 0) == 0),
                    "paper_id",
                ],
                df.loc[
                    df["author_info"].isna()
                    | (df["author_info"].apply(lambda x: len(x) if x else 0) == 0),
                    "pdf_link",
                ],
            )
        )
    else:
        paper_pdf_map = dict(zip(df["paper_id"], df["pdf_link"]))

    # Print the number of papers to process and ask for confirmation
    num_papers = len(paper_pdf_map)
    print(f"Number of papers to process: {num_papers}")
    print(f"Will upload every {BATCH_SIZE} papers.")

    if num_papers == 0:
        print("No papers left to process. Exiting.")
        return

    confirm = input("Do you want to proceed? (yes/no) [default: yes]: ").strip().lower()
    if confirm not in ("", "yes"):
        print("Operation cancelled by the user.")
        return

    asyncio.run(run(paper_pdf_map, df))


if __name__ == "__main__":
    main()
