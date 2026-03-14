import argparse
import asyncio
import base64
import json
import time
from datetime import datetime, timedelta
import os
import traceback

import aiohttp
import pandas as pd
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


def generate_date_urls(start_date: str, end_date: str) -> list:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [
        (start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end - start).days + 1)
    ]
    return [f"https://huggingface.co/papers?date={date}" for date in date_list]


async def extract_paper_links(date_url: str, date: str, session: ClientSession) -> dict:
    valid_dates = []
    async with session.get(date_url, allow_redirects=False) as response:
        if response.status != 200 and response.status != 302:
            return {"papers": [], "valid_dates": []}

        if response.status != 302:
            valid_dates.append(date)

        soup_text = await response.text()
        soup = BeautifulSoup(soup_text, "html.parser")
        paper_links = list(
            set(
                [
                    f"https://huggingface.co{a['href']}".replace("#community", "")
                    for a in soup.find_all("a", href=True)
                    # Skip the links that have /date/ in the URL.
                    if a["href"].startswith("/papers/") and "/date/" not in a["href"] and a["href"] != "/papers/trending"
                ]
            )
        )

        # Find all image tags
        img_tags = soup.find_all("img")

        # Extract image URLs
        img_urls = [img.get("src") for img in img_tags if img.get("src")]
        paper_img_urls = [
            img_url for img_url in img_urls if "social-thumbnails/papers" in img_url
        ]

        return {
            "papers": [{"date": date, "url": link} for link in paper_links],
            "valid_dates": valid_dates,
            "paper_img_urls": paper_img_urls,
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


class AuthorInfo(BaseModel):
    name: str
    affiliation: str
    email: str


async def get_pdf_bytes(pdf_url: str, session: ClientSession) -> bytes:
    """Fetches a PDF from a given URL and returns the raw bytes."""
    async with session.get(pdf_url) as response:
        if response.status != 200:
            raise ValueError(
                f"Failed to fetch PDF from {pdf_url}, status code: {response.status}"
            )
        return await response.read()


async def extract_author_info_from_pdf(pdf_bytes: bytes) -> list[AuthorInfo]:
    """
    Sends PDF bytes to OpenAI GPT-5.4 to extract author information.
    Uses run_in_executor to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    def _call_openai_api():
        client = OpenAI()
        b64_pdf = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-5.4",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract author information from the first page of this paper. "
                                "Return a JSON object with a single key 'authors' containing an array of objects, "
                                "each with keys: 'name' (string), 'affiliation' (string), and 'email' (string). "
                                "If email is not provided, use an empty string."
                            ),
                        },
                        {
                            "type": "file",
                            "file": {
                                "filename": "paper.pdf",
                                "file_data": f"data:application/pdf;base64,{b64_pdf}",
                            },
                        },
                    ],
                }
            ],
        )

        result = json.loads(response.choices[0].message.content)
        return [AuthorInfo(**author) for author in result["authors"]]

    return await loop.run_in_executor(None, _call_openai_api)


async def get_paper_data(paper, soup, session: ClientSession) -> dict:
    """Returns a dictionary containing detailed information about a paper."""
    paper_id = paper["url"].split("/")[-1]
    title = soup.find("h1").text.strip() if soup.find("h1") else "N/A"
    authors = (
        [span.text.strip("\n\t,") for span in soup.find_all("span", class_="author")]
        if soup.find_all("span", class_="author")
        else []
    )
    abstract_tag = soup.find("div", class_="flex flex-col gap-y-2.5")
    abstract = abstract_tag.text.strip() if abstract_tag else ""

    submitted_span = soup.find("span", class_="contents")
    submitted_by = submitted_span.text.strip() if submitted_span else "N/A"

    published_on_div = soup.find(
        "div", string=lambda text: text and "Published on" in text
    )
    published_on = (
        published_on_div.text.strip().replace("Published on ", "")
        if published_on_div
        else None
    )
    published_on = convert_published_on_to_date(published_on) if published_on else "N/A"

    upvotes_div = soup.find("div", class_="font-semibold text-orange-500")
    upvotes = upvotes_div.text if upvotes_div else "-"

    if upvotes == "-":
        upvotes = 0
    else:
        # Replace 1,056 with 1056
        upvotes = int(upvotes.replace(",", ""))

    citing_counts = soup.find_all("span", class_="ml-3 font-normal text-gray-400")
    if len(citing_counts) >= 4:
        models_citing = int(citing_counts[0].text.replace(",", ""))
        datasets_citing = int(citing_counts[1].text.replace(",", ""))
        spaces_citing = int(citing_counts[2].text.replace(",", ""))
        collections_including = int(citing_counts[3].text.replace(",", ""))
    else:
        models_citing, datasets_citing, spaces_citing, collections_including = (
            0,
            0,
            0,
            0,
        )

    # Get the PDF link (ensure exactly one PDF link from arxiv.org/pdf)
    pdf_links = []
    for a_tag in soup.find_all("a", href=True):
        link = a_tag["href"]
        if "arxiv.org/pdf" in a_tag["href"]:
            pdf_links.append(a_tag["href"])

    unique_pdf_links = set(pdf_links)

    if len(unique_pdf_links) != 1:
        # Check if one of the pdf links has the paper_id in it, if so use that. If not, then break.
        paper_id = paper["url"].split("/")[-1]
        pdf_links_with_id = [link for link in pdf_links if paper_id in link]
        unique_pdf_links_with_id = set(pdf_links_with_id)
        if len(unique_pdf_links_with_id) == 1:
            pdf_link = pdf_links_with_id[0]
        elif len(unique_pdf_links_with_id) > 1:
            # Multiple variants (e.g., versioned and unversioned); pick the shortest (unversioned).
            pdf_link = min(unique_pdf_links_with_id, key=len)
        else:
            print(
                f"No matching PDF links found for {paper_id}: {pdf_links}"
            )
            raise ValueError("No PDF link found containing the paper_id")
    else:
        pdf_link = pdf_links[0]

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
        "pdf_link": pdf_link,
        "author_info": None,
    }


async def extract_paper_details_with_metadata(
    paper: dict, session: ClientSession, retries: int, cooldown: int
) -> dict:
    """
    Wrapper to fetch and parse paper details, with simple retry logic.
    """
    attempt = 0
    while attempt < retries:
        try:
            async with session.get(paper["url"]) as response:
                if response.status != 200:
                    attempt += 1
                    await asyncio.sleep(cooldown)
                    continue
                soup_text = await response.text()
                soup = BeautifulSoup(soup_text, "html.parser")

                paper_data = await get_paper_data(paper, soup, session)
                return paper_data
        except Exception as e:
            print(f"Error processing {paper['url']} (attempt {attempt+1}): {e}")
            traceback.print_exc()
            attempt += 1
            await asyncio.sleep(cooldown)
    return {}


async def run_scraper(
    start_date: str,
    end_date: str,
    output_file: str | None,
    retries: int = 3,
    cooldown: int = 5,
    max_requests_per_second: int = 100,
    solicit_user_confirmation: bool = True,
) -> pd.DataFrame:
    urls = generate_date_urls(start_date, end_date)
    semaphore = asyncio.Semaphore(max_requests_per_second)

    headers = {}
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async with aiohttp.ClientSession(headers=headers) as session:

        async def limited_extract_paper_links(url, date):
            async with semaphore:
                return await extract_paper_links(url, date, session)

        link_tasks = [
            limited_extract_paper_links(url, date)
            for date, url in zip([u.split("=")[-1] for u in urls], urls)
        ]
        all_relevant_page_links = await tqdm_asyncio.gather(
            *link_tasks, desc="Extracting Paper Links"
        )

        all_paper_links = [
            link
            for subresult in all_relevant_page_links
            for link in subresult["papers"]
        ]

        async def limited_extract_paper_details(paper):
            async with semaphore:
                return await extract_paper_details_with_metadata(
                    paper, session, retries=retries, cooldown=cooldown
                )

        # Print the number of unique papers to be scraped
        unique_papers = {paper["url"] for paper in all_paper_links}

        # Ask for user confirmation before proceeding
        if solicit_user_confirmation:
            confirmation = (
                input(
                    f"Found {len(unique_papers)} unique papers to scrape. Proceed? (yes/no): "
                )
                .strip()
                .lower()
            )
            if confirmation not in ["yes", "y"]:
                print("Scraping aborted by user.")
                return pd.DataFrame()  # Return an empty DataFrame

        detail_tasks = [
            limited_extract_paper_details(paper) for paper in all_paper_links
        ]
        all_paper_details = await tqdm_asyncio.gather(
            *detail_tasks, desc="Scraping Papers"
        )
        all_paper_details = [detail for detail in all_paper_details if detail]

        print(f"Finished scraping {len(all_paper_details)} papers.")

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
