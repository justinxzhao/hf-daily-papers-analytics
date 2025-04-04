from hf_daily_papers_analytics.hf_papers_scraper import (
    get_pdf_first_page_image,
    extract_author_info_from_image,
)
import aiohttp
import asyncio
from datasets import Dataset, load_dataset
import pandas as pd
from tqdm.asyncio import tqdm
import random


async def fetch_author_info(paper_pdf_map: dict) -> dict:
    """Fetches author information asynchronously for a map of paper ID to PDF link."""

    async def process_pdf_link(paper_id, pdf_link, session, retries=3, cooldown=2):
        for attempt in range(retries):
            try:
                # Download the first page of the PDF and parse authors
                first_pdf_page = await get_pdf_first_page_image(pdf_link, session)
                author_info = await extract_author_info_from_image(first_pdf_page)

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

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_pdf_link(paper_id, pdf_link, session)
            for paper_id, pdf_link in paper_pdf_map.items()
        ]
        results = await tqdm.gather(
            *tasks, desc="Fetching author info", total=len(tasks)
        )
        return {paper_id: author_info for paper_id, author_info in results}


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

    # TEMP: Select a random set of 100 keys from the paper_pdf_map
    paper_pdf_map = dict(
        random.sample(list(paper_pdf_map.items()), min(10, len(paper_pdf_map)))
    )

    # Print the number of papers to process and ask for confirmation
    num_papers = len(paper_pdf_map)
    print(f"Number of papers to process: {num_papers}")

    if num_papers == 0:
        print("No papers left to process. Exiting.")
        return

    confirm = input("Do you want to proceed? (yes/no) [default: yes]: ").strip().lower()
    if confirm not in ("", "yes"):
        print("Operation cancelled by the user.")
        return

    # Fetch author information
    author_info_map = asyncio.run(fetch_author_info(paper_pdf_map))

    # Add the author information to the DataFrame additively
    for paper_id, author_info in author_info_map.items():
        if not author_info:  # Skip if no author info was fetched
            continue
        existing_info = df.loc[df["paper_id"] == paper_id, "author_info"].values[0]
        if existing_info:
            continue
        df.at[df.index[df["paper_id"] == paper_id][0], "author_info"] = author_info

    # Convert the DataFrame back to a Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub("justinxzhao/hf_daily_papers")


if __name__ == "__main__":
    main()
