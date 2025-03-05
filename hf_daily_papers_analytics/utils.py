import pandas as pd


def merge_datasets(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merges newest data with existing data, preferring the new data on shared keys."""
    if existing_df is None or existing_df.empty:
        return new_df
    if new_df is None or new_df.empty:
        return existing_df

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    return (
        combined_df.drop_duplicates(subset=["date", "paper_id"], keep="last")
        .sort_values(by="date", ascending=False)
        .reset_index(drop=True)
    )
