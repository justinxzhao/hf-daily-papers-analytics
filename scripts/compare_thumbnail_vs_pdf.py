"""Compare author info extraction: HF thumbnail vs arxiv PDF first page.

Picks 10 random papers and sends both sources to GPT-5.4, then prints
a side-by-side comparison.

Usage:
    poetry run python scripts/compare_thumbnail_vs_pdf.py
"""

import asyncio
import base64
import io
import json
import time

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader, PdfWriter

load_dotenv()

DATA_PATH = "data/hf_daily_papers.jsonl"
PROMPT = (
    "Extract author information from this paper. "
    "Return a JSON object with a single key 'authors' containing an array of objects, "
    "each with keys: 'name' (string), 'affiliation' (string), and 'email' (string). "
    "If email is not provided, use an empty string."
)


def extract_from_image(client, image_bytes: bytes) -> list[dict]:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-5.4",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("authors", [])


def extract_from_pdf(client, pdf_bytes: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    buf = io.BytesIO()
    writer.write(buf)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-5.4",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "file", "file": {
                    "filename": "paper.pdf",
                    "file_data": f"data:application/pdf;base64,{b64}",
                }},
            ],
        }],
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("authors", [])


async def download(url: str, session: aiohttp.ClientSession) -> bytes:
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.read()


async def main():
    df = pd.read_json(DATA_PATH, lines=True)
    mask = df["author_info"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    sample = df[mask].sample(10, random_state=42)

    client = OpenAI()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; JustinsArxivBot/1.0)"}

    async with aiohttp.ClientSession(headers=headers) as session:
        for idx, (_, row) in enumerate(sample.iterrows()):
            paper_id = row["paper_id"]
            thumbnail_url = row["thumbnail"]
            pdf_url = row["pdf_link"].replace("://arxiv.org/", "://export.arxiv.org/")
            existing = row["author_info"]

            print(f"\n{'='*80}")
            print(f"Paper {idx+1}/10: {paper_id} — {row['title'][:60]}...")
            print(f"  Existing author_info: {len(existing)} authors")

            # Download thumbnail (no rate limit needed)
            print(f"  Downloading thumbnail...")
            thumb_bytes = await download(thumbnail_url, session)
            print(f"    Thumbnail size: {len(thumb_bytes):,} bytes")

            # Extract from thumbnail
            print(f"  Extracting from thumbnail...")
            t0 = time.time()
            thumb_authors = extract_from_image(client, thumb_bytes)
            t_thumb = time.time() - t0
            print(f"    Got {len(thumb_authors)} authors in {t_thumb:.1f}s")

            # Download PDF (respect rate limit)
            print(f"  Downloading PDF (with 3s delay)...")
            await asyncio.sleep(3)
            pdf_bytes = await download(pdf_url, session)
            print(f"    PDF size: {len(pdf_bytes):,} bytes")

            # Extract from PDF
            print(f"  Extracting from PDF first page...")
            t0 = time.time()
            pdf_authors = extract_from_pdf(client, pdf_bytes)
            t_pdf = time.time() - t0
            print(f"    Got {len(pdf_authors)} authors in {t_pdf:.1f}s")

            # Compare
            print(f"\n  {'THUMBNAIL':<40} | {'PDF':<40} | {'EXISTING':<40}")
            print(f"  {'-'*40} | {'-'*40} | {'-'*40}")
            max_len = max(len(thumb_authors), len(pdf_authors), len(existing))
            for i in range(max_len):
                t = thumb_authors[i] if i < len(thumb_authors) else {}
                p = pdf_authors[i] if i < len(pdf_authors) else {}
                e = existing[i] if i < len(existing) else {}
                t_str = f"{t.get('name', '')[:18]:18} | {t.get('affiliation', '')[:18]:18}"
                p_str = f"{p.get('name', '')[:18]:18} | {p.get('affiliation', '')[:18]:18}"
                e_str = f"{e.get('name', '')[:18]:18} | {e.get('affiliation', '')[:18]:18}"
                print(f"  {t_str} | {p_str} | {e_str}")

    print(f"\n{'='*80}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
