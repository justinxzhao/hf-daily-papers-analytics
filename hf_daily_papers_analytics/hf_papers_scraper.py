import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta
import os
import tempfile

import aiohttp
import pandas as pd
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio
from pdf2image import convert_from_path
from io import BytesIO
import requests
from google import genai
from pydantic import BaseModel

from dotenv import load_dotenv
import traceback

# Load environment variables from a .env file
load_dotenv()


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
async def extract_paper_links(date_url: str, date: str, session: ClientSession) -> dict:
    valid_dates = []
    async with session.get(date_url, allow_redirects=False) as response:
        if response.status != 200 and response.status != 302:
            return {"papers": [], "valid_dates": []}

        if response.status != 302:
            valid_dates.append(date)

        # Non-redirect.
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


def get_pdf_first_page_image(pdf_url: str) -> BytesIO:
    """
    Fetches the first page of a PDF from a given URL and returns the image data as a BytesIO object.
    """
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch PDF from {pdf_url}, status code: {response.status_code}"
        )

    # Save the PDF content to a temporary BytesIO object
    pdf_content = BytesIO(response.content)

    # Convert the first page of the PDF to an image
    # Save the BytesIO content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_content.getvalue())
        temp_pdf_path = temp_pdf.name

    # Convert the first page of the PDF to an image
    images = convert_from_path(temp_pdf_path, first_page=1, last_page=1)

    # Clean up the temporary file
    os.remove(temp_pdf_path)
    if not images:
        raise ValueError("No pages found in the PDF.")

    # Save the first page image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        images[0].save(temp_image, format="PNG")
        temp_image_path = temp_image.name

    return temp_image_path


class AuthorInfo(BaseModel):
    name: str
    affiliation: str
    email: str


def extract_author_info_from_image(image_data: str) -> list[AuthorInfo]:
    """
    Calls the Gemini API with the provided image data to extract author information.

    Args:
        image_data (BytesIO): The image data containing author information.

    Returns:
        list[AuthorInfo]: A list of objects containing author name, affiliation, and email.
    """
    api_key = os.getenv("GEMINI_API_KEY", "xxx")  # Replace with your API key
    client = genai.Client(api_key=api_key)

    # Upload the image data to the Gemini API
    file_upload = client.files.upload(file=image_data)

    # Define the prompt for extracting author information
    prompt = (
        "Extract a JSON payload of an array of objects where each object has three keys: "
        "the author name, the author affiliation, and the author email (if provided)."
    )

    # Call the Gemini API with structured output configuration
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, file_upload],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[AuthorInfo],
        },
    )

    # Parse and return the structured response
    return response.parsed


def get_paper_data(paper, soup):
    """Returns a dictionary containing detailed information about a paper."""
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

    # Get the link to the PDF link.
    pdf_links = []
    non_pdf_links = []
    for a_tag in soup.find_all("a", href=True):
        link = a_tag["href"]
        if "arxiv.org/pdf" in a_tag["href"]:
            pdf_links.append(a_tag["href"])

    if len(pdf_links) != 1:
        print(f"Multiple PDF links found for {paper_id}: {pdf_links}")
        raise ValueError("Multiple PDF links found")

    pdf_link = pdf_links[0]
    first_pdf_page = get_pdf_first_page_image(pdf_link)
    cv_extracted_author_info: list[dict] = [
        author.dict() for author in extract_author_info_from_image(first_pdf_page)
    ]

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
        "cv_extracted_author_info": cv_extracted_author_info,
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
                traceback.print_exc()
        attempt += 1
        await asyncio.sleep(cooldown)


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

        async def limited_extract_paper_details(paper):
            async with semaphore:
                return await extract_paper_details_with_metadata(
                    paper, session, retries=retries, cooldown=cooldown
                )

        tasks = [limited_extract_paper_details(paper) for paper in all_paper_links]
        all_paper_details = await tqdm_asyncio.gather(*tasks, desc="Scraping Papers")
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
