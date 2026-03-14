# HF Daily Papers Analytics

A metadata scraper for daily HF papers.

This project scrapes papers from the Hugging Face website, merges them with previously uploaded data, and publishes a consolidated dataset as an HF dataset.

A cron GHA re-scrapes and publishes new data to [`justinxzhao/hf_daily_papers`](https://huggingface.co/datasets/justinxzhao/hf_daily_papers) every day at 6:30 PST.

## Motivation

Hugging Face daily papers has become a vibrant community hub for sharing machine learning research. HF Daily Papers Analytics was created to:

- Centralize daily papers into a searchable dataset.
- Facilitate analysis by enabling trend analysis and historical comparisons.
- Ensure fresh data every day through a streamlined CI/CD pipeline.

## Overview

- **Scraping**: Gathers papers published daily on Hugging Face.
- **Merging**: (optional) If scraping a partial set of dates, we combines new data with the existing dataset, prefering new data. Generally, the upvotes and other metadata for a new paper stabilizes around 1-2 weeks after it was posted.
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

   Create a `.env` file at the project root:

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

For issues and feature requests, please open a [GitHub Issue](https://github.com/justinxzhao/hf-daily-papers-analytics/issues).  


# Visualizations:

- average number of papers per day by (day, week, quarter, overall)
- average number of upvotes per day by (day, week, quarter, overall)
- average density of upvotes (# upvotes / # papers) per day by (day, week, quarter, overall)

- cumulative number of unique authors over time (day, week, quarter, overall)
- cumulative number of unique affiliations over time (day, week, quarter, overall)
- cumulative number of collaborations (pairs of authors) over time (day, week, quarter, overall)
- cumulative number of repeat collaborations (pairs of authors) over time (day, week, quarter, overall)

- distribution of the number of papers that an author has (how many people have published a single paper and never published again?)
- distribution of the amount of time between an author's paper to the next paper getting published
- how often does an author's affiliation change? Distribution of the bumber of unique institutions that an author is represented in?

- Who are the top authors (most # unique papers, # votes / # papers, # votes)
- Who are the top affiliations (most # unique papers, # votes / # papers, # votes)
- Who are the top first authors (most # unique papers, # votes / # papers, # votes)
- Who are the top last authors (most # unique papers, # votes / # papers, # votes)

- num of {papers, upvotes, upvote density} by Chinese authors over time {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by Chinese first authors {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by Chinese last authors {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by non-Chinese authors {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by non-Chinese first authors {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by non-Chinese last authors {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by papers with mixed Chinese and non-Chinese authors {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by Chinese affiliated institutions {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by non-Chinese affiliated institutions {day, week, quarter, year, overall}
- num of {papers, upvotes, upvote density} by papers with mixed Chinese and non-Chinese affiliated institutions {day, week, quarter, year, overall}

- Correlation between number of authors on a paper and number of upvotes?
- Correlation between number of pages and the number of upvotes?
- Correlation between number of unique institutions on a paper and number of upvotes?
- Correlation between # words in title vs. number of upvotes?
- Correlation between # words in abstract vs. number of upvotes?

- Most number of unique authors from a single institution? (which institutions have the most number of unique authors?)