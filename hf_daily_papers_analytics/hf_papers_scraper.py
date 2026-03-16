import asyncio
import base64
import io
import json
import os
from datetime import datetime, timedelta

import aiohttp
import pandas as pd
from aiohttp import ClientSession
from openai import OpenAI
from pypdf import PdfReader, PdfWriter
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

HF_API_BASE = "https://huggingface.co/api"


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


_AUTHOR_EXTRACTION_PROMPT = (
    "Extract author information from this paper. "
    "Return a JSON object with a single key 'authors' containing an array of objects, "
    "each with keys: 'name' (string), 'affiliation' (string), and 'email' (string). "
    "If email is not provided, use an empty string."
)


async def extract_author_info_from_pdf(pdf_bytes: bytes) -> list[AuthorInfo]:
    """
    Sends PDF first page to OpenAI GPT-5.4 to extract author information.
    Uses run_in_executor to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    def _call_openai_api():
        client = OpenAI()

        # Extract only the first page to stay under file size limits
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        first_page_buf = io.BytesIO()
        writer.write(first_page_buf)
        first_page_bytes = first_page_buf.getvalue()

        b64_pdf = base64.standard_b64encode(first_page_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-5.4",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _AUTHOR_EXTRACTION_PROMPT},
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


async def extract_author_info_from_thumbnail(image_bytes: bytes) -> list[AuthorInfo]:
    """
    Sends a thumbnail image to OpenAI GPT-5.4 to extract author information.
    Much faster than PDF extraction since thumbnails are served from HF CDN
    with no rate limits.
    """
    loop = asyncio.get_event_loop()

    def _call_openai_api():
        client = OpenAI()
        b64_img = base64.standard_b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-5.4",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _AUTHOR_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                        },
                    ],
                }
            ],
        )

        result = json.loads(response.choices[0].message.content)
        return [AuthorInfo(**author) for author in result["authors"]]

    return await loop.run_in_executor(None, _call_openai_api)


def _parse_api_paper(entry: dict, date: str) -> dict:
    """Converts a single API response entry into our flat schema."""
    paper = entry["paper"]
    paper_id = paper["id"]
    submitter = paper.get("submittedOnDailyBy") or {}

    return {
        "date": date,
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "authors": [a["name"] for a in paper.get("authors", [])],
        "summary": paper.get("summary", ""),
        "publishedAt": paper.get("publishedAt", ""),
        "submittedOnDailyAt": paper.get("submittedOnDailyAt", ""),
        "submittedBy": submitter.get("user", ""),
        "upvotes": paper.get("upvotes", 0),
        "numComments": entry.get("numComments", 0),
        "ai_summary": paper.get("ai_summary", ""),
        "ai_keywords": paper.get("ai_keywords", []),
        "githubRepo": paper.get("githubRepo"),
        "githubStars": paper.get("githubStars"),
        "thumbnail": entry.get("thumbnail", ""),
        "url": f"https://huggingface.co/papers/{paper_id}",
        "pdf_link": f"https://arxiv.org/pdf/{paper_id}",
        "author_info": None,
    }


async def fetch_papers_for_date(
    date: str, session: ClientSession, retries: int = 3, cooldown: int = 2
) -> list[dict]:
    """Fetches all papers for a given date from the HuggingFace API."""
    url = f"{HF_API_BASE}/daily_papers?date={date}"
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Got status {response.status} for {date} (attempt {attempt + 1})")
                    await asyncio.sleep(cooldown)
                    continue
                data = await response.json()
                return [_parse_api_paper(entry, date) for entry in data]
        except Exception as e:
            print(f"Error fetching {date} (attempt {attempt + 1}): {e}")
            await asyncio.sleep(cooldown)
    print(f"All retries failed for {date}")
    return []


async def run_scraper(
    start_date: str,
    end_date: str,
    output_file: str | None,
    retries: int = 3,
    cooldown: int = 2,
    max_concurrent: int = 20,
) -> pd.DataFrame:
    """Fetches daily papers from the HuggingFace API for a date range."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = [
        (start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end - start).days + 1)
    ]

    semaphore = asyncio.Semaphore(max_concurrent)

    headers = {}
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async with aiohttp.ClientSession(headers=headers) as session:

        async def limited_fetch(date):
            async with semaphore:
                return await fetch_papers_for_date(
                    date, session, retries=retries, cooldown=cooldown
                )

        tasks = [limited_fetch(date) for date in dates]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching papers")

    all_papers = [paper for date_papers in results for paper in date_papers]
    print(f"Fetched {len(all_papers)} papers across {len(dates)} dates.")

    df = pd.DataFrame(all_papers)

    if output_file and not df.empty:
        if output_file.endswith(".json"):
            df.to_json(output_file, orient="records")
        elif output_file.endswith(".jsonl"):
            df.to_json(output_file, orient="records", lines=True)
        else:
            print("Output file must be a .json or .jsonl file. Data not saved.")
        print(f"Data saved to {output_file}")

    return df
