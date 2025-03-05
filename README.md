# HF Daily Papers Analytics

A metadata scraper for daily HF papers.

This project scrapes papers from the Hugging Face website, merges them with previously uploaded data, and publishes a consolidated dataset as an HF dataset.

A cron GHA re-scrapes and publishes new data to [`justinxzhao/hf_daily_papers`](https://huggingface.co/datasets/justinxzhao/hf_daily_papers) every day at 6:30 PST.

## Overview

- **Scraping**: Gathers papers published daily on Hugging Face.
- **Merging**: (optional) If scraping a partial set of dates, we combines new data with the existing dataset, prefering new data. Generally, the upvotes for a new paper stabilizes around 1-2 weeks after it was posted.
- **Publishing**: Uploads the merged dataset to Hugging Face Datasets under [`justinxzhao/hf_daily_papers`](https://huggingface.co/datasets/justinxzhao/hf_daily_papers).

## Repository Structure

```bash
HF_Daily_Papers_Analytics/
  ├─ .github/workflows/
  │   ├─ run_tests.yml          # CI for running tests on PRs/merges to main
  │   └─ update_hf_datasets.yml # Daily job to scrape + merge + upload data
  ├─ scripts/
  │   ├─ hf_papers_scraper.py   # Main script to scrape daily papers
  │   ├─ upload_to_hf_datasets.py
  │   └─ update_hf_datasets.py  # Script orchestrates daily update + merging
  ├─ tests/
  │   └─ test_scraper.py        # Example pytest-based tests
  ├─ pyproject.toml             # Poetry config
  └─ README.md                  # Project documentation
```

## Getting Started

1. **Clone the Repo**  

   ```bash
   git clone https://github.com/yourusername/HF_Daily_Papers_Analytics.git
   cd HF_Daily_Papers_Analytics
   ```

2. **Install Dependencies with Poetry**  

   ```bash
   poetry install
   ```

3. **Set Hugging Face Token**  

   - Create a `.env` file at the project root:

     ```bash
     HUGGINGFACE_HUB_TOKEN=<YOUR_TOKEN_HERE>
     ```

4. **Run the Scraper Locally**  

   ```bash
   poetry run python scripts/hf_papers_scraper.py --start_date 2025-02-01 --end_date 2025-02-28 --output_file test.jsonl
   ```

5. **Run the Tests**  

   ```bash
   poetry run pytest
   ```

## Automated Workflows

1. **Daily Updates**  

   - Defined in [`.github/workflows/update_hf_datasets.yml`](.github/workflows/update_hf_datasets.yml).  
   - Runs every day to scrape new papers and re-upload.

2. **Tests**

   - Defined in [`.github/workflows/run_tests.yml`](.github/workflows/run_tests.yml).  
   - Automatically runs unit tests on new pull requests and merges to `main`.

## Support / Contact

- For issues and PRs, please open a [GitHub Issue](https://github.com/justinxzhao/hf-daily-papers-analytics/issues).  
