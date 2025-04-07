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

python scripts/run_scraper.py \
    --start_date 2025-03-28 \
    --end_date 2025-03-28 \
    --output_file extractions/hf_papers_2025_03_28.jsonl \
    --retries 3 \
    --cooldown 2

python scripts/run_scraper.py \
    --start_date 2025-03-28 \
    --end_date 2025-03-28 \
    --output_file extractions/hf_papers_2025_03_28.jsonl


python scripts/run_scraper.py \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --output_file extractions/hf_papers_2024.jsonl
"""

import argparse
import asyncio
import os
import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hf_daily_papers_analytics.hf_papers_scraper import run_scraper


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Hugging Face papers.")
    parser.add_argument(
        "--start_date", type=str, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file name"
    )
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument(
        "--cooldown", type=int, default=2, help="Cooldown time in seconds"
    )
    parser.add_argument(
        "--solicit_user_confirmation",
        action="store_true",
        help="Solicit user confirmation before proceeding",
        default=False,
    )

    args = parser.parse_args()

    asyncio.run(
        run_scraper(
            args.start_date,
            args.end_date,
            args.output_file,
            retries=args.retries,
            cooldown=args.cooldown,
            solicit_user_confirmation=args.solicit_user_confirmation,
        )
    )
