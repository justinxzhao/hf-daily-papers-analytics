import pandas as pd


def merge_datasets(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merges newest data with existing data, preferring new data but preserving author_info.

    Fresh scrapes return author_info=None for all rows. This merge keeps the new data's
    metadata (upvotes, comments, etc.) but backfills author_info from the existing dataset
    so previously extracted author information is not lost.
    """
    if existing_df is None or existing_df.empty:
        return new_df
    if new_df is None or new_df.empty:
        return existing_df

    # Build lookup of existing author_info before merging
    if "author_info" in existing_df.columns:
        has_info = existing_df.dropna(subset=["author_info"])
        has_info = has_info[has_info["author_info"].apply(
            lambda x: isinstance(x, list) and len(x) > 0
        )]
        info_lookup = dict(
            zip(has_info["paper_id"].astype(str), has_info["author_info"])
        )
    else:
        info_lookup = {}

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    # Normalize columns to avoid mixed-type comparison failures
    combined_df["date"] = combined_df["date"].astype(str).str[:10]
    combined_df["paper_id"] = combined_df["paper_id"].astype(str)
    merged = (
        combined_df.drop_duplicates(subset=["date", "paper_id"], keep="last")
        .sort_values(by="date", ascending=False)
        .reset_index(drop=True)
    )

    # Backfill author_info from existing dataset
    if info_lookup and "author_info" in merged.columns:
        null_mask = merged["author_info"].isna() | merged["author_info"].apply(
            lambda x: not isinstance(x, list) or len(x) == 0
        )
        merged.loc[null_mask, "author_info"] = merged.loc[null_mask, "paper_id"].map(
            info_lookup
        )

    return merged
