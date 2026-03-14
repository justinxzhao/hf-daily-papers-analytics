# Duplicate Paper IDs in the Dataset

## Finding

Out of 13,154 total rows and 13,144 unique paper IDs, **10 papers (0.1%)** appear on two different dates (20 rows total). Every duplicate appears exactly twice.

## Examples

| paper_id | dates |
|----------|-------|
| 2503.05236 | 2025-03-09, 2025-03-10 |
| 2404.03648 | 2024-04-05, 2024-04-08 |
| 2503.05179 | 2025-03-09, 2025-03-10 |
| 2503.0213 | 2025-03-09, 2025-03-10 |
| 2603.05888 | 2026-03-08, 2026-03-09 |

## Why this happens

A paper can be submitted to HuggingFace Daily Papers on one date and appear again on a subsequent date (e.g. trending the next day). The API returns it for both dates, and our schema intentionally keeps both rows since each row represents a paper-on-a-date, not a unique paper.

## Impact on author info enrichment

Since `get_papers_needing_author_info` returns a `dict(paper_id -> pdf_link)`, duplicate paper_ids are naturally deduplicated — we only download and process each PDF once. But `update_df_with_author_info` must write the result back to **all** matching rows for that paper_id.

## Appearance distribution

| Appearances | Paper count |
|-------------|-------------|
| 1x | 13,134 |
| 2x | 10 |
