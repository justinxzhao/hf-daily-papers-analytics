import aiohttp
import asyncio
from hf_daily_papers_analytics.hf_papers_scraper import (
    extract_paper_details_with_metadata,
)
import pprint


def test_extract_paper_details_with_metadata():
    async def extraction_helper():
        paper = {
            "url": "https://huggingface.co/papers/2502.18934",
            "date": "2025-02-27",
        }
        async with aiohttp.ClientSession() as session:
            result = await extract_paper_details_with_metadata(
                paper,
                session,
                retries=2,
                cooldown=2,
            )
            return result

    result = asyncio.run(extraction_helper())

    assert result["date"] == "2025-02-27"
    assert result["paper_id"] == "2502.18934"
    assert result["title"] == "Kanana: Compute-efficient Bilingual Language Models"
    assert result["authors"] == [
        "Kanana LLM Team",
        "Yunju Bak",
        "Hojin Lee",
        "Minho Ryu",
        "Jiyeon Ham",
        "Seungjae Jung",
        "Daniel Wontae Nam",
        "Taegyeong Eo",
        "Donghun Lee",
        "Doohae Jung",
        "Boseop Kim",
        "Nayeon Kim",
        "Jaesun Park",
        "Hyunho Kim",
        "Hyunwoong Ko",
        "Changmin Lee",
        "Kyoung-Woon On",
        "Seulye Baeg",
        "Junrae Cho",
        "Sunghee Jung",
        "Jieun Kang",
        "EungGyun Kim",
    ]
    assert result["abstract"] == (
        "We introduce Kanana, a series of bilingual language models that demonstrate\n"
        "exceeding performance in Korean and competitive performance in English. The\n"
        "computational cost of Kanana is significantly lower than that of\n"
        "state-of-the-art models of similar size. The report details the techniques\n"
        "employed during pre-training to achieve compute-efficient yet competitive\n"
        "models, including high quality data filtering, staged pre-training, depth\n"
        "up-scaling, and pruning and distillation. Furthermore, the report outlines the\n"
        "methodologies utilized during the post-training of the Kanana models,\n"
        "encompassing supervised fine-tuning and preference optimization, aimed at\n"
        "enhancing their capability for seamless interaction with users. Lastly, the\n"
        "report elaborates on plausible approaches used for language model adaptation to\n"
        "specific scenarios, such as embedding, retrieval augmented generation, and\n"
        "function calling. The Kanana model series spans from 2.1B to 32.5B parameters\n"
        "with 2.1B models (base, instruct, embedding) publicly released to promote\n"
        "research on Korean language models."
    )
    assert result["url"] == "https://huggingface.co/papers/2502.18934"
    assert result["submitted_by"] == "bzantium"

    assert list(result.keys()) == [
        "date",
        "paper_id",
        "title",
        "submitted_by",
        "authors",
        "abstract",
        "upvotes",
        "models_citing",
        "datasets_citing",
        "spaces_citing",
        "collections_including",
        "url",
    ]
