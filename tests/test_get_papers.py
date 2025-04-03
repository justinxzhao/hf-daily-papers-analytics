import pytest
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from hf_daily_papers_analytics.hf_papers_scraper import (
    extract_paper_links,
    convert_published_on_to_date,
)


@pytest.mark.asyncio
async def test_extract_paper_links():
    date_url = "https://huggingface.co/papers/date/2025-03-05"
    date = "2025-03-05"

    async with ClientSession() as session:
        result = await extract_paper_links(date_url, date, session)

        paper_urls = set([paper["url"] for paper in result["papers"]])

        expected_paper_urls = {
            "https://huggingface.co/papers/2502.14856",
            "https://huggingface.co/papers/2503.00069",
            "https://huggingface.co/papers/2503.00200",
            "https://huggingface.co/papers/2503.00735",
            "https://huggingface.co/papers/2503.00876",
            "https://huggingface.co/papers/2503.00955",
            "https://huggingface.co/papers/2503.01328",
            "https://huggingface.co/papers/2503.01342",
            "https://huggingface.co/papers/2503.01842",
            "https://huggingface.co/papers/2503.01935",
            "https://huggingface.co/papers/2503.02152",
            "https://huggingface.co/papers/2503.02197",
            "https://huggingface.co/papers/2503.02268",
            "https://huggingface.co/papers/2503.02304",
            "https://huggingface.co/papers/2503.02357",
            "https://huggingface.co/papers/2503.02368",
            "https://huggingface.co/papers/2503.02537",
            "https://huggingface.co/papers/2503.02682",
            "https://huggingface.co/papers/2503.02783",
            "https://huggingface.co/papers/2503.02812",
            "https://huggingface.co/papers/2503.02823",
            "https://huggingface.co/papers/2503.02846",
            "https://huggingface.co/papers/2503.02876",
            "https://huggingface.co/papers/2503.02878",
            "https://huggingface.co/papers/2503.02879",
            "https://huggingface.co/papers/2503.03651",
        }
        assert paper_urls == expected_paper_urls

        assert result["valid_dates"] == [date]

        assert result["paper_img_urls"] == [
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.00955.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.01935.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02682.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02879.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.00735.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02846.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.03651.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.00069.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02368.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.01328.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.00200.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02537.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02878.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02812.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02197.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.01342.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02357.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.14856.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02783.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.00876.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02876.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02304.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02152.png",
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02823.png",
        ]


def test_convert_published_on_to_date():
    convert_published_on_to_date("Mar 20") == "2025-03-20"
    convert_published_on_to_date("Mar 20, 2025") == "2025-03-20"
    convert_published_on_to_date("Nov 4, 2024") == "2024-11-04"
    convert_published_on_to_date("Nov 4, 2024") == "2024-11-04"
    convert_published_on_to_date("Jul 11, 2024") == "2024-07-11"
