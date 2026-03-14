"""Sample usage:

python scripts/run_scraper.py \
    --start_date 2025-01-15 \
    --end_date 2025-01-15 \
    --output_file extractions/hf_papers_2025_01_15.jsonl

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Hugging Face daily papers via API.")
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
