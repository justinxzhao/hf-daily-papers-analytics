"""Script to scrape daily papers from Hugging Face and update the dataset on Hugging Face Hub."""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta

import aiohttp
import dotenv
import pandas as pd
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.asyncio import tqdm_asyncio

from hf_daily_papers_analytics.hf_papers_scraper import run_scraper


dotenv.load_dotenv()


# Function to fetch dataset from Hugging Face
def download_hf_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset["train"])
    return df


# Function to merge datasets, keeping the latest scraper data
def merge_datasets(existing_df, new_df):
    if existing_df is not None and not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = (
            combined_df.sort_values(by="date", ascending=False)
            .drop_duplicates(subset=["date", "paper_id"], keep="first")
            .reset_index(drop=True)
        )
    else:
        combined_df = new_df
    return combined_df


# Function to upload dataset back to Hugging Face
def upload_to_hf(dataset, dataset_name, token):
    dataset_dict = DatasetDict({"train": Dataset.from_pandas(dataset)})
    dataset_dict.push_to_hub(dataset_name, token=token)


async def main(args):
    dataset_name = "justinxzhao/hf_daily_papers"
    hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
    end_date = datetime.today().strftime("%Y-%m-%d")

    start_date = (datetime.today() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    output_file = f"extractions/hf_papers_{start_date}_to_{end_date}.jsonl"

    print(f"Scraping papers from {start_date} to {end_date}...")
    new_df = await run_scraper(start_date, end_date, output_file, retries=3, cooldown=2)

    print("Downloading existing dataset from Hugging Face...")
    try:
        existing_df = download_hf_dataset(dataset_name)
    except Exception as e:
        print(f"Could not fetch existing dataset: {e}")
        existing_df = None

    print("Merging datasets...")
    merged_df = merge_datasets(existing_df, new_df)

    breakpoint()

    print("Uploading merged dataset to Hugging Face...")
    upload_to_hf(merged_df, dataset_name, hf_token)

    print("Dataset successfully updated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Hugging Face daily papers.")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days from today to scrape papers for.",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
