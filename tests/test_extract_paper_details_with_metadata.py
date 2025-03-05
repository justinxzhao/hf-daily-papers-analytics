import aiohttp
import asyncio
from hf_papers_scraper import extract_paper_details_with_metadata
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

    assert result == {
        "date": "2025-02-27",
        "paper_id": "2502.18934",
        "title": "Kanana: Compute-efficient Bilingual Language Models",
        "authors": [
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
        ],
        "abstract": "We introduce Kanana, a series of bilingual language models that demonstrate\nexceeding performance in Korean and competitive performance in English. The\ncomputational cost of Kanana is significantly lower than that of\nstate-of-the-art models of similar size. The report details the techniques\nemployed during pre-training to achieve compute-efficient yet competitive\nmodels, including high quality data filtering, staged pre-training, depth\nup-scaling, and pruning and distillation. Furthermore, the report outlines the\nmethodologies utilized during the post-training of the Kanana models,\nencompassing supervised fine-tuning and preference optimization, aimed at\nenhancing their capability for seamless interaction with users. Lastly, the\nreport elaborates on plausible approaches used for language model adaptation to\nspecific scenarios, such as embedding, retrieval augmented generation, and\nfunction calling. The Kanana model series spans from 2.1B to 32.5B parameters\nwith 2.1B models (base, instruct, embedding) publicly released to promote\nresearch on Korean language models.",
        "upvotes": 49,
        "models_citing": 3,
        "datasets_citing": 0,
        "spaces_citing": 1,
        "collections_including": 2,
        "url": "https://huggingface.co/papers/2502.18934",
    }
