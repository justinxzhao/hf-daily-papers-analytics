"""Script to scrape daily papers from Hugging Face and update the dataset on Hugging Face Hub.

For a full scrape:
python scripts/update_hf_datasets.py --full_scrape --upload

Test scrape:
python scripts/update_hf_datasets.py --days 7
"""

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
from hf_daily_papers_analytics.utils import merge_datasets


dotenv.load_dotenv()


# Function to fetch dataset from Hugging Face
def download_hf_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset["train"])
    return df


# Function to upload dataset back to Hugging Face
def upload_to_hf(dataset, dataset_name, token):
    dataset_dict = DatasetDict({"train": Dataset.from_pandas(dataset)})
    dataset_dict.push_to_hub(dataset_name, token=token)


async def main(args):
    dataset_name = "justinxzhao/hf_daily_papers"
    hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]

    if not args.full_scrape:
        end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = "2023-05-04"

    print(f"Scraping papers from {start_date} to {end_date}...")
    new_df = await run_scraper(
        start_date, end_date, output_file=None, retries=3, cooldown=2
    )

    if not args.full_scrape:
        print("Downloading existing dataset from Hugging Face...")
        try:
            existing_df = download_hf_dataset(dataset_name)
            print("Merging datasets...")
            merged_df = merge_datasets(existing_df, new_df)
        except Exception as e:
            raise ValueError(f"Could not fetch existing dataset: {e}")
    else:
        merged_df = new_df

    if args.upload:
        print("Uploading merged dataset to Hugging Face...")
        upload_to_hf(merged_df, dataset_name, hf_token)
        print("Dataset successfully updated!")

    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Hugging Face daily papers.")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days from today to scrape papers for.",
    )
    parser.add_argument(
        "--full_scrape",
        action="store_true",
        help="Scrape papers from the beginning of time.",
        default=False,
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HF.",
        default=False,
    )

    args = parser.parse_args()

    asyncio.run(main(args))
