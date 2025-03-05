import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta

import aiohttp
import pandas as pd
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio


# Function to generate URLs for the specified date range
def generate_date_urls(start_date: str, end_date: str) -> list:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [
        (start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end - start).days + 1)
    ]
    return [f"https://huggingface.co/papers?date={date}" for date in date_list]


# Function to extract paper links from a given daily papers page
async def extract_paper_links(date_url: str, date: str, session: ClientSession) -> list:
    async with session.get(date_url) as response:
        if response.status != 200:
            return []
        soup = BeautifulSoup(await response.text(), "html.parser")
        paper_links = list(
            set(
                [
                    f"https://huggingface.co{a['href']}".replace("#community", "")
                    for a in soup.find_all("a", href=True)
                    if a["href"].startswith("/papers/")
                ]
            )
        )
        return [{"date": date, "url": link} for link in paper_links]


def get_paper_data(paper, soup):
    paper_id = paper["url"].split("/")[-1]
    title = soup.find("h1").text.strip() if soup.find("h1") else "N/A"
    authors = (
        [span.text.strip("\n\t,") for span in soup.find_all("span", class_="author")]
        if soup.find_all("span", class_="author")
        else []
    )
    abstract = soup.find("p", class_="text-gray-700 dark:text-gray-400").text.strip()

    upvotes = soup.find("div", class_="font-semibold text-orange-500").text
    if upvotes == "-":
        upvotes = 0
    else:
        # Replace 1,056 with 1056
        upvotes = int(upvotes.replace(",", ""))
    citing_counts = soup.find_all("span", class_="ml-3 font-normal text-gray-400")
    models_citing = int(citing_counts[0].text.replace(",", ""))
    datasets_citing = int(citing_counts[1].text.replace(",", ""))
    spaces_citing = int(citing_counts[2].text.replace(",", ""))
    collections_including = int(citing_counts[3].text.replace(",", ""))
    return {
        "date": paper["date"],
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "upvotes": upvotes,
        "models_citing": models_citing,
        "datasets_citing": datasets_citing,
        "spaces_citing": spaces_citing,
        "collections_including": collections_including,
        "url": paper["url"],
    }


# Function to extract detailed information from each paper page
async def extract_paper_details_with_metadata(
    paper: dict, session: ClientSession, retries, cooldown
) -> dict:
    attempt = 0
    while attempt < retries:
        async with session.get(paper["url"]) as response:
            if response.status != 200:
                attempt += 1
                await asyncio.sleep(cooldown)
                continue
            soup = BeautifulSoup(await response.text(), "html.parser")

            try:
                paper_data = get_paper_data(paper, soup)
                return paper_data
            except Exception as e:
                print(f"Error processing {paper['url']}: {e}")
                breakpoint()
        attempt += 1
        await asyncio.sleep(cooldown)


# Main function to run the scraper with asyncio and a progress bar
async def run_scraper(
    start_date: str,
    end_date: str,
    output_file: str | None,
    retries: int = 3,
    cooldown: int = 5,
) -> pd.DataFrame:
    urls = generate_date_urls(start_date, end_date)
    async with aiohttp.ClientSession() as session:
        tasks = [
            extract_paper_links(url, date, session)
            for date, url in zip([u.split("=")[-1] for u in urls], urls)
        ]
        all_paper_links = await tqdm_asyncio.gather(
            *tasks, desc="Extracting Paper Links"
        )
        all_paper_links = [link for sublist in all_paper_links for link in sublist]

        tasks = [
            extract_paper_details_with_metadata(
                paper, session, retries=retries, cooldown=cooldown
            )
            for paper in all_paper_links
        ]
        all_paper_details = await tqdm_asyncio.gather(*tasks, desc="Scraping Papers")
        all_paper_details = [detail for detail in all_paper_details if detail]

        # Create a pandas dataframe from the extracted data
        df = pd.DataFrame(all_paper_details)
        if output_file:
            if output_file.endswith(".json"):
                df.to_json(output_file, orient="records")
            elif output_file.endswith(".jsonl"):
                df.to_json(output_file, orient="records", lines=True)
            else:
                print("Output file must be a .json or .jsonl file. Data not saved.")
            print(f"Data saved to {output_file}")
        return df
