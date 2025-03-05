"""Sample usage:

python scripts/hf_papers_scraper.py \
    --start_date 2025-02-01 \
    --end_date 2025-02-27 \
    --output_file extractions/hf_papers_2025_02_01_to_2025_02_27.json \
    --retries 3 \
    --cooldown 2

python scripts/hf_papers_scraper.py \
    --start_date 2023-05-04 \
    --end_date 2023-12-31 \
    --output_file extractions/hf_papers_2023_05_04_to_2023_12_31.json \
    --retries 3 \
    --cooldown 2

python scripts/hf_papers_scraper.py \
    --start_date 2025-02-28 \
    --end_date 2025-03-03 \
    --output_file extractions/hf_papers_2025_02_28_to_2025_03_03.jsonl \
    --retries 3 \
    --cooldown 2
"""

import argparse
import asyncio
import os
import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.hf_papers_scraper import run_scraper


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Hugging Face papers.")
    parser.add_argument(
        "--start_date", type=str, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--output_file", type=str, help="Output file name")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument(
        "--cooldown", type=int, default=2, help="Cooldown time in seconds"
    )

    args = parser.parse_args()

    asyncio.run(
        run_scraper(
            args.start_date,
            args.end_date,
            args.output_file,
            retries=args.retries,
            cooldown=args.cooldown,
        )
    )
