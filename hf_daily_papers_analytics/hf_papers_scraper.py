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
                    # Skip the links that have /date/ in the URL.
                    if a["href"].startswith("/papers/") and "/date/" not in a["href"]
                ]
            )
        )
        daily_paper_links = list(
            set(
                [
                    f"https://huggingface.co{a['href']}".replace("#community", "")
                    for a in soup.find_all("a", href=True)
                    # Skip the links that have /date/ in the URL.
                    if "/date/" in a["href"]
                ]
            )
        )
        return {
            "papers": [{"date": date, "url": link} for link in paper_links],
            "daily_paper_links": daily_paper_links,
        }


def convert_published_on_to_date(published_on: str) -> str:
    try:
        # Try to parse the date with the current year
        date_obj = datetime.strptime(
            f"{published_on} {datetime.now().year}", "%b %d %Y"
        )
    except ValueError:
        # If parsing fails, assume the year is included in the string
        date_obj = datetime.strptime(published_on, "%b %d, %Y")
    return date_obj.strftime("%Y-%m-%d")


def get_paper_data(paper, soup):
    paper_id = paper["url"].split("/")[-1]
    title = soup.find("h1").text.strip() if soup.find("h1") else "N/A"
    authors = (
        [span.text.strip("\n\t,") for span in soup.find_all("span", class_="author")]
        if soup.find_all("span", class_="author")
        else []
    )
    abstract = soup.find("div", class_="flex flex-col gap-y-2.5").text.strip()

    submitted_by = soup.find("span", class_="contents").text.strip()

    published_on = (
        soup.find("div", string=lambda text: text and "Published on" in text)
        .text.strip()
        .replace("Published on ", "")
    )

    published_on = convert_published_on_to_date(published_on)

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
        "submitted_by": submitted_by,
        "published_on": published_on,
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
        attempt += 1
        await asyncio.sleep(cooldown)


def get_valid_daily_paper_dates(daily_paper_links):
    """Returns a set of valid daily paper dates from the list of daily paper links."""
    return set(
        [
            link.split("/")[-1]
            for link in daily_paper_links
            if link.split("/")[-1] != "date"
        ]
    )


async def run_scraper(
    start_date: str,
    end_date: str,
    output_file: str | None,
    retries: int = 3,
    cooldown: int = 5,
    max_requests_per_second: int = 100,
) -> pd.DataFrame:
    urls = generate_date_urls(start_date, end_date)
    semaphore = asyncio.Semaphore(max_requests_per_second)

    async with aiohttp.ClientSession() as session:

        async def limited_extract_paper_links(url, date):
            async with semaphore:
                return await extract_paper_links(url, date, session)

        tasks = [
            limited_extract_paper_links(url, date)
            for date, url in zip([u.split("=")[-1] for u in urls], urls)
        ]
        all_relevant_page_links = await tqdm_asyncio.gather(
            *tasks, desc="Extracting Paper Links"
        )
        all_paper_links = [
            link
            for subresult in all_relevant_page_links
            for link in subresult["papers"]
        ]

        all_daily_paper_links = [
            link
            for subresult in all_relevant_page_links
            for link in subresult["daily_paper_links"]
        ]

        all_valid_daily_paper_dates = get_valid_daily_paper_dates(all_daily_paper_links)

        # Filter out all papers that are not in the valid daily paper dates.
        # This can happen for dates where there is no published daily paper list (e.g. weekends).
        # These are still crawled because of redirects.
        len_before_filter = len(all_paper_links)

        all_paper_links = [
            paper
            for paper in all_paper_links
            if paper["date"] in all_valid_daily_paper_dates
        ]
        len_after_filter = len(all_paper_links)
        print(
            f"Filtered out {len_before_filter - len_after_filter} papers not in valid daily paper dates (due to weekends, etc.)"
        )

        async def limited_extract_paper_details(paper):
            async with semaphore:
                return await extract_paper_details_with_metadata(
                    paper, session, retries=retries, cooldown=cooldown
                )

        tasks = [limited_extract_paper_details(paper) for paper in all_paper_links]
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
