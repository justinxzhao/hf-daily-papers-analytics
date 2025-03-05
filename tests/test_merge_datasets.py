import pandas as pd
import pytest


from hf_daily_papers_analytics.utils import merge_datasets


def test_merge_datasets():
    existing_df = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02"],
            "paper_id": [1, 2],
            "title": ["Paper 1", "Paper 2"],
        }
    )

    new_df = pd.DataFrame(
        {
            "date": ["2021-01-02", "2021-01-03"],
            "paper_id": [2, 3],
            "title": ["New Paper 2", "Paper 3"],
        }
    )

    expected_df = pd.DataFrame(
        {
            "date": ["2021-01-03", "2021-01-02", "2021-01-01"],
            "paper_id": [3, 2, 1],
            "title": ["Paper 3", "New Paper 2", "Paper 1"],
        }
    ).reset_index(drop=True)

    result_df = merge_datasets(existing_df, new_df)

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_merge_datasets_no_overlap():
    existing_df = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02"],
            "paper_id": [1, 2],
            "title": ["Paper 1", "Paper 2"],
        }
    )

    new_df = pd.DataFrame(
        {
            "date": ["2021-01-03", "2021-01-04"],
            "paper_id": [3, 4],
            "title": ["Paper 3", "Paper 4"],
        }
    )

    expected_df = pd.DataFrame(
        {
            "date": ["2021-01-04", "2021-01-03", "2021-01-02", "2021-01-01"],
            "paper_id": [4, 3, 2, 1],
            "title": ["Paper 4", "Paper 3", "Paper 2", "Paper 1"],
        }
    ).reset_index(drop=True)

    result_df = merge_datasets(existing_df, new_df)

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_merge_datasets_empty_existing():
    existing_df = pd.DataFrame(columns=["date", "paper_id", "title"])

    new_df = pd.DataFrame(
        {
            "date": ["2021-01-02", "2021-01-01"],
            "paper_id": [2, 1],
            "title": ["Paper 2", "Paper 1"],
        }
    )

    expected_df = new_df.reset_index(drop=True)

    result_df = merge_datasets(existing_df, new_df)

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_merge_datasets_empty_new():
    existing_df = pd.DataFrame(
        {
            "date": ["2021-01-02", "2021-01-01"],
            "paper_id": [2, 1],
            "title": ["Paper 2", "Paper 1"],
        }
    )
    new_df = pd.DataFrame(columns=["date", "paper_id", "title"])

    expected_df = existing_df.reset_index(drop=True)
    # expected_df = existing_df

    result_df = merge_datasets(existing_df, new_df)

    pd.testing.assert_frame_equal(result_df, expected_df)
