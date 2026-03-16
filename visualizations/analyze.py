"""Generate all visualizations and tables for HF Daily Papers analysis.

Usage:
    poetry run python visualizations/analyze.py
"""

import os
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = Path(__file__).parent
DATA_PATH = Path(__file__).parent.parent / "data" / "hf_daily_papers.jsonl"

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.figsize": (14, 6), "figure.dpi": 150, "savefig.bbox": "tight"})


# ── Chinese classification heuristics ──────────────────────────────────────────

CHINESE_SURNAMES = {
    "Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou",
    "Xu", "Sun", "Ma", "Zhu", "Hu", "Guo", "He", "Lin", "Luo", "Gao",
    "Liang", "Zheng", "Xie", "Han", "Tang", "Feng", "Yu", "Dong", "Xiao", "Cheng",
    "Cao", "Yuan", "Deng", "Peng", "Pan", "Cai", "Jiang", "Su", "Lu", "Wei",
    "Ding", "Ren", "Shen", "Ye", "Yao", "Pei", "Zou", "Tan", "Wan", "Qin",
    "Qiu", "Hou", "Bai", "Meng", "Xiong", "Qian", "Shi", "Shao", "Cui", "Tao",
    "Kong", "Jia", "Zeng", "Xue", "Fan", "Kang", "Dai", "Song", "Tian", "Lei",
    "Lv", "Jin", "Yan", "Duan", "Fang", "Hao", "Gu", "Long", "Wan", "Chang",
    "Zuo", "Wen", "Nie", "Mao", "Jing", "Yue", "Fu", "Ge", "Lai", "Ke",
    "Ai", "Bian", "Cha", "Chi", "Chu", "Geng", "Ji", "Kuang", "Lang", "Lian",
    "Ling", "Mi", "Min", "Mo", "Mu", "Nan", "Ou", "Pi", "Pu", "Qi",
    "Qu", "Rong", "Sang", "Shan", "Shang", "Shu", "Si", "Sui", "Tu", "Weng",
    "Xi", "Xia", "Xiang", "Xin", "Xing", "Yao", "Yin", "You", "Zhan", "Zhong",
    "Zhuang", "Zhuo",
}

CHINESE_AFFILIATION_KEYWORDS = [
    "china", "chinese", "beijing", "shanghai", "shenzhen", "hangzhou", "nanjing",
    "guangzhou", "chengdu", "wuhan", "xi'an", "xian", "tianjin", "hefei",
    "tsinghua", "peking university", "pku", "fudan", "zhejiang", "sjtu",
    "shanghai jiao tong", "nanjing university", "ustc", "harbin", "huazhong",
    "beihang", "renmin", "nankai", "tongji", "southeast university",
    "sun yat-sen", "zhongshan", "xiamen", "sichuan university",
    "baidu", "alibaba", "tencent", "bytedance", "huawei", "sensetime",
    "megvii", "didi", "jd.com", "xiaomi", "oppo", "vivo",
    "chinese academy", "cas ", "casia",
]


def is_chinese_name(name: str) -> bool:
    """Heuristic: check if surname (last token) is a common Chinese surname."""
    parts = name.strip().split()
    if not parts:
        return False
    surname = parts[-1]
    return surname in CHINESE_SURNAMES


def is_chinese_affiliation(affiliation: str) -> bool:
    """Heuristic: check if affiliation contains Chinese institution keywords."""
    if not affiliation:
        return False
    aff_lower = affiliation.lower()
    return any(kw in aff_lower for kw in CHINESE_AFFILIATION_KEYWORDS)


# ── Data loading & preprocessing ──────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    df = pd.read_json(DATA_PATH, lines=True)
    df["date"] = pd.to_datetime(df["date"])
    df["num_authors"] = df["authors"].apply(len)
    df["title_word_count"] = df["title"].str.split().apply(len)
    df["abstract_word_count"] = df["summary"].str.split().apply(len)

    # Extract author_info fields
    df["has_author_info"] = df["author_info"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )

    # Number of unique institutions per paper
    def count_institutions(info):
        if not isinstance(info, list):
            return 0
        affs = {a.get("affiliation", "").strip() for a in info if a.get("affiliation", "").strip()}
        return len(affs)

    df["num_institutions"] = df["author_info"].apply(count_institutions)

    return df


def explode_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Create a row per author per paper using author_info."""
    rows = []
    for _, paper in df.iterrows():
        info = paper["author_info"]
        if not isinstance(info, list):
            continue
        for i, author in enumerate(info):
            rows.append({
                "date": paper["date"],
                "paper_id": paper["paper_id"],
                "upvotes": paper["upvotes"],
                "author_name": author.get("name", ""),
                "affiliation": author.get("affiliation", ""),
                "email": author.get("email", ""),
                "is_first_author": i == 0,
                "is_last_author": i == len(info) - 1,
                "is_chinese_name": is_chinese_name(author.get("name", "")),
                "is_chinese_affiliation": is_chinese_affiliation(author.get("affiliation", "")),
            })
    return pd.DataFrame(rows)


# ── Plotting helpers ──────────────────────────────────────────────────────────


def plot_time_series_multi_agg(daily: pd.Series, title: str, ylabel: str, filepath: str):
    """Plot a time series at daily, weekly, quarterly granularity + overall mean."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16)

    overall_mean = daily.mean()

    # Daily
    ax = axes[0, 0]
    ax.plot(daily.index, daily.values, alpha=0.4, linewidth=0.5)
    ax.axhline(overall_mean, color="red", linestyle="--", label=f"Overall mean: {overall_mean:.1f}")
    ax.set_title("Daily")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    # Weekly
    ax = axes[0, 1]
    weekly = daily.resample("W").mean()
    ax.plot(weekly.index, weekly.values, linewidth=1)
    ax.axhline(overall_mean, color="red", linestyle="--", label=f"Overall mean: {overall_mean:.1f}")
    ax.set_title("Weekly Average")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    # Quarterly
    ax = axes[1, 0]
    quarterly = daily.resample("QE").mean()
    ax.bar(quarterly.index, quarterly.values, width=60, alpha=0.7)
    ax.axhline(overall_mean, color="red", linestyle="--", label=f"Overall mean: {overall_mean:.1f}")
    ax.set_title("Quarterly Average")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-Q%q"))
    ax.tick_params(axis="x", rotation=45)

    # Overall summary stats
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = (
        f"Mean: {daily.mean():.2f}\n"
        f"Median: {daily.median():.2f}\n"
        f"Std: {daily.std():.2f}\n"
        f"Min: {daily.min():.2f}\n"
        f"Max: {daily.max():.2f}\n"
        f"Total days: {len(daily)}"
    )
    ax.text(0.3, 0.5, stats_text, transform=ax.transAxes, fontsize=14,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
    ax.set_title("Summary Statistics")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved {filepath}")


def plot_cumulative(dates_series: pd.Series, values: list, title: str, ylabel: str, filepath: str):
    """Plot cumulative values over time at multiple aggregations."""
    cum_df = pd.DataFrame({"date": dates_series, "value": values}).set_index("date").sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)

    ax = axes[0]
    ax.plot(cum_df.index, cum_df["value"], linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title("Daily")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1]
    quarterly = cum_df.resample("QE").last()
    ax.bar(quarterly.index, quarterly["value"], width=60, alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title("Quarterly")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-Q%q"))
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved {filepath}")


def plot_comparative_hist(data_dict: dict, xlabel: str, title: str, filepath: str,
                          bins=100, normalize=False):
    """Overlay normalized histograms for multiple groups."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (label, values), color in zip(data_dict.items(), colors):
        if len(values) == 0:
            continue
        vals = np.asarray(values, dtype=float)
        med = np.median(vals)
        # Always normalize to % so groups with different sizes are comparable
        weights = np.ones_like(vals) / len(vals) * 100
        ax.hist(vals, bins=bins, alpha=0.5, weights=weights,
                label=f"{label} (n={len(vals):,}, med={med:.0f})",
                color=color, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of Group")
    ax.set_title(title)
    ax.legend()
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved {filepath}")


def compute_gaps(author_df: pd.DataFrame, subset: pd.DataFrame = None) -> list:
    """Compute days between consecutive papers for authors in subset."""
    if subset is None:
        subset = author_df
    author_dates = subset.groupby("author_name")["date"].apply(
        lambda x: sorted(x.unique())
    )
    gaps = []
    for dates_list in author_dates:
        if len(dates_list) < 2:
            continue
        for i in range(1, len(dates_list)):
            gap = (dates_list[i] - dates_list[i - 1]).days
            if gap > 0:
                gaps.append(gap)
    return gaps


def compute_affs_per_author(subset: pd.DataFrame) -> pd.Series:
    """Count unique affiliations per author in subset."""
    return subset.groupby("author_name")["affiliation"].apply(
        lambda x: len(set(a.strip().lower() for a in x if a.strip()))
    )


# ── Group A: Paper & Upvote Volume ────────────────────────────────────────────


def plot_group_a(df: pd.DataFrame):
    print("Group A: Paper & Upvote Volume")
    daily = df.set_index("date")

    papers_per_day = daily.groupby(daily.index).size()
    plot_time_series_multi_agg(papers_per_day, "Papers per Day", "# Papers",
                               str(OUT_DIR / "a1_papers_per_day.png"))

    upvotes_per_day = daily.groupby(daily.index)["upvotes"].sum()
    plot_time_series_multi_agg(upvotes_per_day, "Total Upvotes per Day", "# Upvotes",
                               str(OUT_DIR / "a2_upvotes_per_day.png"))

    density = upvotes_per_day / papers_per_day
    plot_time_series_multi_agg(density, "Upvote Density (Upvotes / Paper)", "Upvotes per Paper",
                               str(OUT_DIR / "a3_upvote_density.png"))


# ── Group B: Cumulative Growth ────────────────────────────────────────────────


def plot_group_b(df: pd.DataFrame):
    print("Group B: Cumulative Growth")
    df_sorted = df.sort_values("date")

    # b1: cumulative unique authors
    seen_authors = set()
    cum_authors = []
    dates = []
    for _, row in df_sorted.iterrows():
        if isinstance(row["author_info"], list):
            for a in row["author_info"]:
                seen_authors.add(a.get("name", ""))
        cum_authors.append(len(seen_authors))
        dates.append(row["date"])
    plot_cumulative(pd.Series(dates), cum_authors, "Cumulative Unique Authors",
                    "# Unique Authors", str(OUT_DIR / "b1_cumulative_authors.png"))

    # b2: cumulative unique affiliations
    seen_affs = set()
    cum_affs = []
    for _, row in df_sorted.iterrows():
        if isinstance(row["author_info"], list):
            for a in row["author_info"]:
                aff = a.get("affiliation", "").strip()
                if aff:
                    seen_affs.add(aff.lower())
        cum_affs.append(len(seen_affs))
    plot_cumulative(pd.Series(dates), cum_affs, "Cumulative Unique Affiliations",
                    "# Unique Affiliations", str(OUT_DIR / "b2_cumulative_affiliations.png"))

    # b3: cumulative unique collaborations (author pairs)
    seen_pairs = set()
    cum_pairs = []
    for _, row in df_sorted.iterrows():
        if isinstance(row["author_info"], list) and len(row["author_info"]) >= 2:
            names = [a.get("name", "") for a in row["author_info"]]
            for pair in combinations(sorted(names), 2):
                seen_pairs.add(pair)
        cum_pairs.append(len(seen_pairs))
    plot_cumulative(pd.Series(dates), cum_pairs, "Cumulative Unique Collaborations (Author Pairs)",
                    "# Unique Pairs", str(OUT_DIR / "b3_cumulative_collaborations.png"))

    # b4: cumulative repeat collaborations
    pair_counts: Counter = Counter()
    cum_repeats = []
    for _, row in df_sorted.iterrows():
        if isinstance(row["author_info"], list) and len(row["author_info"]) >= 2:
            names = [a.get("name", "") for a in row["author_info"]]
            for pair in combinations(sorted(names), 2):
                pair_counts[pair] += 1
        repeat_count = sum(1 for c in pair_counts.values() if c > 1)
        cum_repeats.append(repeat_count)
    plot_cumulative(pd.Series(dates), cum_repeats, "Cumulative Repeat Collaborations",
                    "# Pairs with 2+ Papers", str(OUT_DIR / "b4_cumulative_repeat_collabs.png"))

    # b5: cumulative unique institution collaborations
    seen_inst_pairs = set()
    cum_inst_pairs = []
    for _, row in df_sorted.iterrows():
        if isinstance(row["author_info"], list) and len(row["author_info"]) >= 2:
            affs = sorted(set(
                a.get("affiliation", "").strip().lower()
                for a in row["author_info"]
                if a.get("affiliation", "").strip()
            ))
            if len(affs) >= 2:
                for pair in combinations(affs, 2):
                    seen_inst_pairs.add(pair)
        cum_inst_pairs.append(len(seen_inst_pairs))
    plot_cumulative(pd.Series(dates), cum_inst_pairs, "Cumulative Unique Institution Collaborations",
                    "# Unique Institution Pairs", str(OUT_DIR / "b5_cumulative_institution_collaborations.png"))

    # b6: cumulative repeat institution collaborations
    inst_pair_counts: Counter = Counter()
    cum_inst_repeats = []
    for _, row in df_sorted.iterrows():
        if isinstance(row["author_info"], list) and len(row["author_info"]) >= 2:
            affs = sorted(set(
                a.get("affiliation", "").strip().lower()
                for a in row["author_info"]
                if a.get("affiliation", "").strip()
            ))
            if len(affs) >= 2:
                for pair in combinations(affs, 2):
                    inst_pair_counts[pair] += 1
        repeat_count = sum(1 for c in inst_pair_counts.values() if c > 1)
        cum_inst_repeats.append(repeat_count)
    plot_cumulative(pd.Series(dates), cum_inst_repeats, "Cumulative Repeat Institution Collaborations",
                    "# Institution Pairs with 2+ Papers", str(OUT_DIR / "b6_cumulative_institution_repeat_collabs.png"))


# ── Group C: Author Activity Distributions ────────────────────────────────────


def plot_group_c(df: pd.DataFrame, author_df: pd.DataFrame):
    print("Group C: Author Activity Distributions")

    # c1: papers per author
    papers_per_author = author_df.groupby("author_name")["paper_id"].nunique()
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = range(1, min(papers_per_author.max() + 2, 52))
    ax.hist(papers_per_author.values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("# Papers")
    ax.set_ylabel("# Authors")
    ax.set_title(f"Distribution of Papers per Author (n={len(papers_per_author):,})")
    one_paper = (papers_per_author == 1).sum()
    ax.annotate(f"1 paper: {one_paper:,} authors ({one_paper/len(papers_per_author)*100:.1f}%)",
                xy=(1, one_paper), fontsize=11, ha="left",
                xytext=(5, one_paper * 0.8),
                arrowprops=dict(arrowstyle="->"))
    plt.savefig(OUT_DIR / "c1_papers_per_author_dist.png")
    plt.close()
    print(f"  Saved c1_papers_per_author_dist.png")

    # c2: time between papers (normalized to %)
    gaps = compute_gaps(author_df)

    fig, ax = plt.subplots(figsize=(12, 6))
    gaps_arr = np.array(gaps)
    clip = int(np.percentile(gaps_arr, 99))
    weights = np.ones_like(gaps_arr, dtype=float) / len(gaps_arr) * 100
    ax.hist(gaps_arr, bins=100, edgecolor="black", alpha=0.7, weights=weights, range=(0, clip))
    ax.set_xlabel("Days Between Papers")
    ax.set_ylabel("% of Occurrences")
    ax.set_title(f"Distribution of Time Between Consecutive Papers (n={len(gaps):,} gaps)")
    ax.axvline(np.median(gaps), color="red", linestyle="--", label=f"Median: {np.median(gaps):.0f} days")
    ax.legend()
    plt.savefig(OUT_DIR / "c2_time_between_papers_dist.png")
    plt.close()
    print(f"  Saved c2_time_between_papers_dist.png")

    # c3: affiliations per author (normalized to %)
    affs_per_author = compute_affs_per_author(author_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = range(0, min(affs_per_author.max() + 2, 20))
    weights = np.ones_like(affs_per_author.values, dtype=float) / len(affs_per_author) * 100
    ax.hist(affs_per_author.values, bins=bins, edgecolor="black", alpha=0.7, weights=weights)
    ax.set_xlabel("# Unique Affiliations")
    ax.set_ylabel("% of Authors")
    ax.set_title(f"Distribution of Unique Affiliations per Author (n={len(affs_per_author):,})")
    plt.savefig(OUT_DIR / "c3_affiliations_per_author_dist.png")
    plt.close()
    print(f"  Saved c3_affiliations_per_author_dist.png")

    # c4: distinct last authors per first author
    first_authors = author_df[author_df["is_first_author"]][["paper_id", "author_name"]].rename(
        columns={"author_name": "first_author"})
    last_authors = author_df[author_df["is_last_author"]][["paper_id", "author_name"]].rename(
        columns={"author_name": "last_author"})
    first_last = first_authors.merge(last_authors, on="paper_id")
    # Exclude single-author papers where first == last
    first_last = first_last[first_last["first_author"] != first_last["last_author"]]
    last_per_first = first_last.groupby("first_author")["last_author"].nunique()
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = range(1, min(last_per_first.max() + 2, 30))
    ax.hist(last_per_first.values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("# Distinct Last Authors")
    ax.set_ylabel("# First Authors")
    ax.set_title(f"Distinct Last Authors per First Author (n={len(last_per_first):,} first authors)")
    plt.savefig(OUT_DIR / "c4_last_authors_per_first_author_dist.png")
    plt.close()
    print(f"  Saved c4_last_authors_per_first_author_dist.png")

    # c5: affiliations of first authors vs last authors
    first_affs = compute_affs_per_author(author_df[author_df["is_first_author"]])
    last_affs = compute_affs_per_author(author_df[author_df["is_last_author"]])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Unique Affiliations: First Authors vs Last Authors", fontsize=16)
    max_bin = max(first_affs.max(), last_affs.max()) + 2
    bins = range(0, min(max_bin, 20))
    axes[0].hist(first_affs.values, bins=bins, edgecolor="black", alpha=0.7, color="#1f77b4")
    axes[0].set_xlabel("# Unique Affiliations")
    axes[0].set_ylabel("# Authors")
    axes[0].set_title(f"First Authors (n={len(first_affs):,})")
    axes[1].hist(last_affs.values, bins=bins, edgecolor="black", alpha=0.7, color="#ff7f0e")
    axes[1].set_xlabel("# Unique Affiliations")
    axes[1].set_title(f"Last Authors (n={len(last_affs):,})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "c5_affiliations_first_vs_last_author_dist.png")
    plt.close()
    print(f"  Saved c5_affiliations_first_vs_last_author_dist.png")

    # c6: first and last author overlap
    first_set = set(author_df[author_df["is_first_author"]]["author_name"])
    last_set = set(author_df[author_df["is_last_author"]]["author_name"])
    only_first = len(first_set - last_set)
    only_last = len(last_set - first_set)
    both = len(first_set & last_set)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(["First only", "Last only", "Both"], [only_first, only_last, both],
                  color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black")
    for bar, val in zip(bars, [only_first, only_last, both]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, f"{val:,}",
                ha="center", fontsize=12)
    ax.set_ylabel("# Authors")
    ax.set_title("Authors Who Are First Author, Last Author, or Both")
    plt.savefig(OUT_DIR / "c6_first_and_last_author_overlap.png")
    plt.close()
    print(f"  Saved c6_first_and_last_author_overlap.png")

    # c7: number of authors per paper
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = range(1, min(df["num_authors"].max() + 2, 52))
    ax.hist(df["num_authors"].values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("# Authors")
    ax.set_ylabel("# Papers")
    ax.set_title(f"Distribution of Authors per Paper (n={len(df):,}, median={df['num_authors'].median():.0f})")
    ax.axvline(df["num_authors"].median(), color="red", linestyle="--", label=f"Median: {df['num_authors'].median():.0f}")
    ax.legend()
    plt.savefig(OUT_DIR / "c7_num_authors_per_paper_dist.png")
    plt.close()
    print(f"  Saved c7_num_authors_per_paper_dist.png")

    # c8: number of affiliations per paper
    papers_with_info = df[df["has_author_info"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = range(0, min(papers_with_info["num_institutions"].max() + 2, 30))
    ax.hist(papers_with_info["num_institutions"].values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("# Unique Affiliations")
    ax.set_ylabel("# Papers")
    ax.set_title(f"Distribution of Affiliations per Paper (n={len(papers_with_info):,}, median={papers_with_info['num_institutions'].median():.0f})")
    ax.axvline(papers_with_info["num_institutions"].median(), color="red", linestyle="--",
               label=f"Median: {papers_with_info['num_institutions'].median():.0f}")
    ax.legend()
    plt.savefig(OUT_DIR / "c8_num_affiliations_per_paper_dist.png")
    plt.close()
    print(f"  Saved c8_num_affiliations_per_paper_dist.png")

    # c9: papers per author table (CSV)
    ppa_table = papers_per_author.value_counts().sort_index().reset_index()
    ppa_table.columns = ["num_papers", "num_authors"]
    ppa_table["pct_authors"] = (ppa_table["num_authors"] / ppa_table["num_authors"].sum() * 100).round(2)
    ppa_table.to_csv(OUT_DIR / "c9_papers_per_author_table.csv", index=False)
    print(f"  Saved c9_papers_per_author_table.csv")

    # c10: distinct first authors per last author
    first_last_rev = first_last.copy()  # reuse from c4
    first_per_last = first_last_rev.groupby("last_author")["first_author"].nunique()
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = range(1, min(first_per_last.max() + 2, 30))
    ax.hist(first_per_last.values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("# Distinct First Authors")
    ax.set_ylabel("# Last Authors")
    ax.set_title(f"Distinct First Authors per Last Author (n={len(first_per_last):,} last authors)")
    plt.savefig(OUT_DIR / "c10_first_authors_per_last_author_dist.png")
    plt.close()
    print(f"  Saved c10_first_authors_per_last_author_dist.png")

    # c10b: first authors per last author table (CSV)
    fpl_table = first_per_last.value_counts().sort_index().reset_index()
    fpl_table.columns = ["num_first_authors", "num_last_authors"]
    fpl_table["pct_last_authors"] = (fpl_table["num_last_authors"] / fpl_table["num_last_authors"].sum() * 100).round(2)
    fpl_table.to_csv(OUT_DIR / "c10_first_authors_per_last_author_table.csv", index=False)
    print(f"  Saved c10_first_authors_per_last_author_table.csv")

    # c11: authors per paper table (CSV)
    apa_table = df["num_authors"].value_counts().sort_index().reset_index()
    apa_table.columns = ["num_authors", "num_papers"]
    apa_table["pct_papers"] = (apa_table["num_papers"] / apa_table["num_papers"].sum() * 100).round(2)
    apa_table.to_csv(OUT_DIR / "c11_num_authors_per_paper_table.csv", index=False)
    print(f"  Saved c11_num_authors_per_paper_table.csv")

    # c12: first-author papers per first author (table)
    first_only = author_df[author_df["is_first_author"]]
    fa_papers = first_only.groupby("author_name")["paper_id"].nunique()
    fa_table = fa_papers.value_counts().sort_index().reset_index()
    fa_table.columns = ["num_first_author_papers", "num_authors"]
    fa_table["pct_authors"] = (fa_table["num_authors"] / fa_table["num_authors"].sum() * 100).round(2)
    fa_table.to_csv(OUT_DIR / "c12_first_author_papers_per_author_table.csv", index=False)
    print(f"  Saved c12_first_author_papers_per_author_table.csv")

    # c13: last-author papers per last author (table)
    last_only = author_df[author_df["is_last_author"]]
    la_papers = last_only.groupby("author_name")["paper_id"].nunique()
    la_table = la_papers.value_counts().sort_index().reset_index()
    la_table.columns = ["num_last_author_papers", "num_authors"]
    la_table["pct_authors"] = (la_table["num_authors"] / la_table["num_authors"].sum() * 100).round(2)
    la_table.to_csv(OUT_DIR / "c13_last_author_papers_per_author_table.csv", index=False)
    print(f"  Saved c13_last_author_papers_per_author_table.csv")

    # c14: time between consecutive first-author papers (normalized %)
    fa_gaps = compute_gaps(author_df, first_only)
    fig, ax = plt.subplots(figsize=(12, 6))
    fa_gaps_arr = np.array(fa_gaps)
    if len(fa_gaps_arr) > 0:
        clip = int(np.percentile(fa_gaps_arr, 99))
        weights = np.ones_like(fa_gaps_arr, dtype=float) / len(fa_gaps_arr) * 100
        ax.hist(fa_gaps_arr, bins=100, edgecolor="black", alpha=0.7, weights=weights, range=(0, clip))
        ax.axvline(np.median(fa_gaps_arr), color="red", linestyle="--",
                   label=f"Median: {np.median(fa_gaps_arr):.0f} days")
        ax.legend()
    ax.set_xlabel("Days Between First-Author Papers")
    ax.set_ylabel("% of Occurrences")
    ax.set_title(f"Time Between Consecutive First-Author Papers (n={len(fa_gaps):,} gaps)")
    plt.savefig(OUT_DIR / "c14_time_between_first_author_papers_dist.png")
    plt.close()
    print(f"  Saved c14_time_between_first_author_papers_dist.png")

    # c15: time between consecutive last-author papers (normalized %)
    la_gaps = compute_gaps(author_df, last_only)
    fig, ax = plt.subplots(figsize=(12, 6))
    la_gaps_arr = np.array(la_gaps)
    if len(la_gaps_arr) > 0:
        clip = int(np.percentile(la_gaps_arr, 99))
        weights = np.ones_like(la_gaps_arr, dtype=float) / len(la_gaps_arr) * 100
        ax.hist(la_gaps_arr, bins=100, edgecolor="black", alpha=0.7, weights=weights, range=(0, clip))
        ax.axvline(np.median(la_gaps_arr), color="red", linestyle="--",
                   label=f"Median: {np.median(la_gaps_arr):.0f} days")
        ax.legend()
    ax.set_xlabel("Days Between Last-Author Papers")
    ax.set_ylabel("% of Occurrences")
    ax.set_title(f"Time Between Consecutive Last-Author Papers (n={len(la_gaps):,} gaps)")
    plt.savefig(OUT_DIR / "c15_time_between_last_author_papers_dist.png")
    plt.close()
    print(f"  Saved c15_time_between_last_author_papers_dist.png")


# ── Group D: Top Authors & Affiliations ───────────────────────────────────────


def make_top_table(author_df: pd.DataFrame, group_col: str, filter_col: str = None) -> pd.DataFrame:
    """Build a top-N table by papers, upvotes, and upvote density."""
    if filter_col:
        subset = author_df[author_df[filter_col]]
    else:
        subset = author_df

    grouped = subset.groupby(group_col).agg(
        num_papers=("paper_id", "nunique"),
        total_upvotes=("upvotes", "sum"),
    ).reset_index()
    grouped["upvote_density"] = grouped["total_upvotes"] / grouped["num_papers"]
    return grouped.sort_values("num_papers", ascending=False).head(50)


def plot_group_d(author_df: pd.DataFrame):
    print("Group D: Top Authors & Affiliations")

    # d1: top authors
    top_authors = make_top_table(author_df, "author_name")
    top_authors.to_csv(OUT_DIR / "d1_top_authors.csv", index=False)
    print(f"  Saved d1_top_authors.csv")

    # d2: top affiliations
    aff_df = author_df[author_df["affiliation"].str.strip() != ""]
    top_affs = make_top_table(aff_df, "affiliation")
    top_affs.to_csv(OUT_DIR / "d2_top_affiliations.csv", index=False)
    print(f"  Saved d2_top_affiliations.csv")

    # d3: top first authors
    top_first = make_top_table(author_df, "author_name", "is_first_author")
    top_first.to_csv(OUT_DIR / "d3_top_first_authors.csv", index=False)
    print(f"  Saved d3_top_first_authors.csv")

    # d4: top last authors
    top_last = make_top_table(author_df, "author_name", "is_last_author")
    top_last.to_csv(OUT_DIR / "d4_top_last_authors.csv", index=False)
    print(f"  Saved d4_top_last_authors.csv")

    # d5: papers per affiliation (all affiliations)
    aff_papers = aff_df.groupby("affiliation")["paper_id"].nunique().reset_index()
    aff_papers.columns = ["affiliation", "num_papers"]
    aff_papers = aff_papers.sort_values("num_papers", ascending=False)
    aff_papers.to_csv(OUT_DIR / "d5_papers_per_affiliation.csv", index=False)
    print(f"  Saved d5_papers_per_affiliation.csv")

    # d6: top 100 author collaborations
    author_pair_counts: Counter = Counter()
    for _, row in author_df.groupby("paper_id"):
        names = sorted(row["author_name"].unique())
        if len(names) >= 2:
            for pair in combinations(names, 2):
                author_pair_counts[pair] += 1
    top_author_collabs = pd.DataFrame(
        [(a, b, c) for (a, b), c in author_pair_counts.most_common(100)],
        columns=["author_1", "author_2", "num_papers"],
    )
    top_author_collabs.to_csv(OUT_DIR / "d6_top_author_collaborations.csv", index=False)
    print(f"  Saved d6_top_author_collaborations.csv")

    # d7: top 100 institution collaborations
    inst_pair_counts: Counter = Counter()
    for paper_id, grp in aff_df.groupby("paper_id"):
        affs = sorted(set(grp["affiliation"].str.strip().str.lower()))
        if len(affs) >= 2:
            for pair in combinations(affs, 2):
                inst_pair_counts[pair] += 1
    top_inst_collabs = pd.DataFrame(
        [(a, b, c) for (a, b), c in inst_pair_counts.most_common(100)],
        columns=["institution_1", "institution_2", "num_papers"],
    )
    top_inst_collabs.to_csv(OUT_DIR / "d7_top_institution_collaborations.csv", index=False)
    print(f"  Saved d7_top_institution_collaborations.csv")


# ── Group E: Chinese vs Non-Chinese ──────────────────────────────────────────


def classify_paper_origin(author_info: list, by_affiliation: bool = False) -> str:
    """Classify a paper as 'chinese', 'non_chinese', or 'mixed'."""
    if not isinstance(author_info, list) or not author_info:
        return "unknown"

    if by_affiliation:
        flags = [is_chinese_affiliation(a.get("affiliation", "")) for a in author_info]
    else:
        flags = [is_chinese_name(a.get("name", "")) for a in author_info]

    if all(flags):
        return "chinese"
    elif not any(flags):
        return "non_chinese"
    else:
        return "mixed"


def classify_first_last(author_info: list, position: str, by_affiliation: bool = False) -> str:
    """Classify based on first or last author only."""
    if not isinstance(author_info, list) or not author_info:
        return "unknown"
    idx = 0 if position == "first" else -1
    author = author_info[idx]
    if by_affiliation:
        return "chinese" if is_chinese_affiliation(author.get("affiliation", "")) else "non_chinese"
    else:
        return "chinese" if is_chinese_name(author.get("name", "")) else "non_chinese"


def plot_origin_time_series(df: pd.DataFrame, origin_col: str, origin_val: str,
                            title: str, filepath: str):
    """Plot papers, upvotes, and density over time for a given origin classification."""
    subset = df[df[origin_col] == origin_val].copy()
    if subset.empty:
        print(f"  Skipping {filepath} (no data)")
        return

    daily_papers = subset.set_index("date").resample("D").size()
    daily_upvotes = subset.set_index("date").resample("D")["upvotes"].sum()

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(title, fontsize=16)

    for ax, data, ylabel in [
        (axes[0], daily_papers, "# Papers"),
        (axes[1], daily_upvotes, "# Upvotes"),
    ]:
        weekly = data.resample("W").mean()
        ax.plot(weekly.index, weekly.values, linewidth=1)
        quarterly = data.resample("QE").mean()
        ax.bar(quarterly.index, quarterly.values, width=60, alpha=0.3, color="orange")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

    # Density
    density = daily_upvotes / daily_papers.replace(0, np.nan)
    weekly_density = density.resample("W").mean()
    axes[2].plot(weekly_density.index, weekly_density.values, linewidth=1)
    quarterly_density = density.resample("QE").mean()
    axes[2].bar(quarterly_density.index, quarterly_density.values, width=60, alpha=0.3, color="orange")
    axes[2].set_ylabel("Upvotes / Paper")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved {filepath}")


def plot_group_e(df: pd.DataFrame, author_df: pd.DataFrame):
    print("Group E: Chinese vs Non-Chinese Analysis")

    # Classify papers
    df = df.copy()
    df["origin_by_name"] = df["author_info"].apply(lambda x: classify_paper_origin(x, by_affiliation=False))
    df["origin_by_aff"] = df["author_info"].apply(lambda x: classify_paper_origin(x, by_affiliation=True))
    df["first_author_origin"] = df["author_info"].apply(lambda x: classify_first_last(x, "first"))
    df["last_author_origin"] = df["author_info"].apply(lambda x: classify_first_last(x, "last"))
    df["first_author_aff_origin"] = df["author_info"].apply(lambda x: classify_first_last(x, "first", by_affiliation=True))
    df["last_author_aff_origin"] = df["author_info"].apply(lambda x: classify_first_last(x, "last", by_affiliation=True))

    configs = [
        ("origin_by_name", "chinese", "Chinese Authors (by name)", "e1_chinese_authors_over_time.png"),
        ("first_author_origin", "chinese", "Chinese First Authors (by name)", "e2_chinese_first_authors_over_time.png"),
        ("last_author_origin", "chinese", "Chinese Last Authors (by name)", "e3_chinese_last_authors_over_time.png"),
        ("origin_by_name", "non_chinese", "Non-Chinese Authors (by name)", "e4_non_chinese_authors_over_time.png"),
        ("first_author_origin", "non_chinese", "Non-Chinese First Authors (by name)", "e5_non_chinese_first_authors_over_time.png"),
        ("last_author_origin", "non_chinese", "Non-Chinese Last Authors (by name)", "e6_non_chinese_last_authors_over_time.png"),
        ("origin_by_name", "mixed", "Mixed Chinese/Non-Chinese Authors (by name)", "e7_mixed_authors_over_time.png"),
        ("origin_by_aff", "chinese", "Chinese Affiliations", "e8_chinese_affiliations_over_time.png"),
        ("origin_by_aff", "non_chinese", "Non-Chinese Affiliations", "e9_non_chinese_affiliations_over_time.png"),
        ("origin_by_aff", "mixed", "Mixed Chinese/Non-Chinese Affiliations", "e10_mixed_affiliations_over_time.png"),
    ]
    for col, val, title, fname in configs:
        plot_origin_time_series(df, col, val, title, str(OUT_DIR / fname))

    # e_summary: paper origin summary table
    total = len(df)
    summary_rows = []
    for label, col in [("By author name", "origin_by_name"), ("By affiliation", "origin_by_aff")]:
        counts = df[col].value_counts()
        for origin in ["chinese", "non_chinese", "mixed", "unknown"]:
            n = counts.get(origin, 0)
            summary_rows.append({
                "classification": label,
                "origin": origin,
                "num_papers": n,
                "pct_papers": round(n / total * 100, 2),
            })
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(OUT_DIR / "e_paper_origin_summary.csv", index=False)
    print(f"  Saved e_paper_origin_summary.csv")

    # e11-e13: time between papers, Chinese vs non-Chinese
    chinese_authors = author_df[author_df["is_chinese_name"]]
    non_chinese_authors = author_df[~author_df["is_chinese_name"]]

    for suffix, filter_col, label in [
        ("e11_time_between_papers_chinese_vs_non.png", None, "All Authors"),
        ("e12_time_between_papers_first_author_chinese_vs_non.png", "is_first_author", "First Authors"),
        ("e13_time_between_papers_last_author_chinese_vs_non.png", "is_last_author", "Last Authors"),
    ]:
        cn = chinese_authors[chinese_authors[filter_col]] if filter_col else chinese_authors
        ncn = non_chinese_authors[non_chinese_authors[filter_col]] if filter_col else non_chinese_authors
        cn_gaps = compute_gaps(author_df, cn)
        ncn_gaps = compute_gaps(author_df, ncn)
        plot_comparative_hist(
            {"Chinese": cn_gaps, "Non-Chinese": ncn_gaps},
            "Days Between Papers",
            f"Time Between Consecutive Papers: Chinese vs Non-Chinese ({label})",
            str(OUT_DIR / suffix),
        )

    # e14-e16: affiliations per author, Chinese vs non-Chinese
    for suffix, filter_col, label in [
        ("e14_affiliations_per_author_chinese_vs_non.png", None, "All Authors"),
        ("e15_affiliations_per_first_author_chinese_vs_non.png", "is_first_author", "First Authors"),
        ("e16_affiliations_per_last_author_chinese_vs_non.png", "is_last_author", "Last Authors"),
    ]:
        cn = chinese_authors[chinese_authors[filter_col]] if filter_col else chinese_authors
        ncn = non_chinese_authors[non_chinese_authors[filter_col]] if filter_col else non_chinese_authors
        cn_affs = compute_affs_per_author(cn)
        ncn_affs = compute_affs_per_author(ncn)
        plot_comparative_hist(
            {"Chinese": cn_affs.values, "Non-Chinese": ncn_affs.values},
            "# Unique Affiliations",
            f"Unique Affiliations per Author: Chinese vs Non-Chinese ({label})",
            str(OUT_DIR / suffix),
            bins=range(0, 15),
        )

    # e17-e18: affiliation breakdown for non-Chinese / Chinese authors
    def affiliation_breakdown(subset: pd.DataFrame) -> dict:
        """For each author, classify their affiliations as Chinese-only, non-Chinese-only, or both."""
        results = {"any": Counter(), "first": Counter(), "last": Counter()}
        for role, role_filter in [("any", None), ("first", "is_first_author"), ("last", "is_last_author")]:
            role_subset = subset[subset[role_filter]] if role_filter else subset
            for author_name, grp in role_subset.groupby("author_name"):
                affs = [a.strip() for a in grp["affiliation"] if a.strip()]
                if not affs:
                    results[role]["No affiliation"] += 1
                    continue
                has_cn = any(is_chinese_affiliation(a) for a in affs)
                has_ncn = any(not is_chinese_affiliation(a) for a in affs)
                if has_cn and has_ncn:
                    results[role]["Both"] += 1
                elif has_cn:
                    results[role]["Chinese affil. only"] += 1
                else:
                    results[role]["Non-Chinese affil. only"] += 1
        return results

    def plot_affiliation_breakdown(breakdown, name_label, fname):
        categories = ["Chinese affil. only", "Non-Chinese affil. only", "Both"]
        roles = ["any", "first", "last"]
        role_labels = ["Any Author", "First Author", "Last Author"]
        x = np.arange(len(roles))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cat in enumerate(categories):
            vals = [breakdown[role].get(cat, 0) for role in roles]
            ax.bar(x + i * width, vals, width, label=cat, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels(role_labels)
        ax.set_ylabel("# Authors")
        ax.set_title(f"{name_label} Authors: Affiliation Breakdown")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname)
        plt.close()
        print(f"  Saved {fname}")

    for subset, name_label, fname in [
        (non_chinese_authors, "Non-Chinese-Name", "e17_non_chinese_author_affiliation_breakdown.png"),
        (chinese_authors, "Chinese-Name", "e18_chinese_author_affiliation_breakdown.png"),
    ]:
        breakdown = affiliation_breakdown(subset)
        plot_affiliation_breakdown(breakdown, name_label, fname)

    # e19-e20: last co-author origin breakdown
    # For each author, classify whether their last co-authors were Chinese, non-Chinese, or both
    def last_coauthor_breakdown(subset: pd.DataFrame, author_df_full: pd.DataFrame) -> dict:
        """For each author, classify their last co-authors as Chinese-only, non-Chinese-only, or both."""
        last_authors_info = author_df_full[author_df_full["is_last_author"]][
            ["paper_id", "author_name", "is_chinese_name"]
        ].rename(columns={"author_name": "last_author", "is_chinese_name": "last_is_chinese"})

        results = {"any": Counter(), "first": Counter(), "last": Counter()}
        for role, role_filter in [("any", None), ("first", "is_first_author"), ("last", "is_last_author")]:
            role_subset = subset[subset[role_filter]] if role_filter else subset
            merged = role_subset[["paper_id", "author_name"]].merge(last_authors_info, on="paper_id")
            # Exclude papers where the author IS the last author (self)
            merged = merged[merged["author_name"] != merged["last_author"]]
            for author_name, grp in merged.groupby("author_name"):
                has_cn = grp["last_is_chinese"].any()
                has_ncn = (~grp["last_is_chinese"]).any()
                if has_cn and has_ncn:
                    results[role]["Both"] += 1
                elif has_cn:
                    results[role]["Chinese last coauthor only"] += 1
                else:
                    results[role]["Non-Chinese last coauthor only"] += 1
        return results

    for subset, name_label, fname in [
        (non_chinese_authors, "Non-Chinese-Name", "e19_non_chinese_author_last_coauthor_breakdown.png"),
        (chinese_authors, "Chinese-Name", "e20_chinese_author_last_coauthor_breakdown.png"),
    ]:
        breakdown = last_coauthor_breakdown(subset, author_df)
        categories = ["Chinese last coauthor only", "Non-Chinese last coauthor only", "Both"]
        roles = ["any", "first", "last"]
        role_labels = ["Any Author", "First Author", "Last Author"]
        x = np.arange(len(roles))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cat in enumerate(categories):
            vals = [breakdown[role].get(cat, 0) for role in roles]
            ax.bar(x + i * width, vals, width, label=cat, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels(role_labels)
        ax.set_ylabel("# Authors")
        ax.set_title(f"{name_label} Authors: Last Co-Author Origin Breakdown")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname)
        plt.close()
        print(f"  Saved {fname}")

    # e21-e24: multi-paper-only versions of affiliation and last-coauthor breakdowns
    multi_paper_names = author_df.groupby("author_name")["paper_id"].nunique()
    multi_paper_names = set(multi_paper_names[multi_paper_names > 1].index)
    cn_multi = chinese_authors[chinese_authors["author_name"].isin(multi_paper_names)]
    ncn_multi = non_chinese_authors[non_chinese_authors["author_name"].isin(multi_paper_names)]

    for subset, name_label, fname in [
        (ncn_multi, "Non-Chinese-Name Multi-Paper", "e21_non_chinese_multi_paper_affiliation_breakdown.png"),
        (cn_multi, "Chinese-Name Multi-Paper", "e22_chinese_multi_paper_affiliation_breakdown.png"),
    ]:
        breakdown = affiliation_breakdown(subset)
        plot_affiliation_breakdown(breakdown, name_label, fname)

    for subset, name_label, fname in [
        (ncn_multi, "Non-Chinese-Name Multi-Paper", "e23_non_chinese_multi_paper_last_coauthor_breakdown.png"),
        (cn_multi, "Chinese-Name Multi-Paper", "e24_chinese_multi_paper_last_coauthor_breakdown.png"),
    ]:
        breakdown = last_coauthor_breakdown(subset, author_df)
        categories = ["Chinese last coauthor only", "Non-Chinese last coauthor only", "Both"]
        roles = ["any", "first", "last"]
        role_labels = ["Any Author", "First Author", "Last Author"]
        x = np.arange(len(roles))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cat in enumerate(categories):
            vals = [breakdown[role].get(cat, 0) for role in roles]
            ax.bar(x + i * width, vals, width, label=cat, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels(role_labels)
        ax.set_ylabel("# Authors")
        ax.set_title(f"{name_label} Authors: Last Co-Author Origin Breakdown")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname)
        plt.close()
        print(f"  Saved {fname}")

    # e25: overall average upvote density by paper classification
    df["is_solo"] = df["num_authors"] == 1

    slices = [
        ("Chinese first author", df["first_author_origin"] == "chinese"),
        ("Non-Chinese first author", df["first_author_origin"] == "non_chinese"),
        ("Chinese last author", df["last_author_origin"] == "chinese"),
        ("Non-Chinese last author", df["last_author_origin"] == "non_chinese"),
        ("Chinese affil. only", df["origin_by_aff"] == "chinese"),
        ("Non-Chinese affil. only", df["origin_by_aff"] == "non_chinese"),
        ("Mixed affil.", df["origin_by_aff"] == "mixed"),
        ("Mixed authors (by name)", df["origin_by_name"] == "mixed"),
        ("Excl. Chinese authors", df["origin_by_name"] == "chinese"),
        ("Excl. non-Chinese authors", df["origin_by_name"] == "non_chinese"),
        ("Solo author", df["is_solo"]),
    ]

    labels = []
    densities = []
    counts = []
    for label, mask in slices:
        subset = df[mask]
        if subset.empty:
            continue
        avg = subset["upvotes"].mean()
        labels.append(label)
        densities.append(avg)
        counts.append(len(subset))

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, densities, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average Upvotes per Paper")
    ax.set_title("Overall Upvote Density by Paper Classification")
    ax.axvline(df["upvotes"].mean(), color="red", linestyle="--",
               label=f"Overall mean: {df['upvotes'].mean():.1f}")
    for bar, d, n in zip(bars, densities, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{d:.1f}  (n={n:,})", va="center", fontsize=9)
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "e25_upvote_density_comparison.png")
    plt.close()
    print(f"  Saved e25_upvote_density_comparison.png")

    # e26: cumulative average upvote density over time by paper classification
    fig, ax = plt.subplots(figsize=(20, 10))
    cmap = plt.cm.tab20
    for i, (label, mask) in enumerate(slices):
        subset = df[mask].sort_values("date")
        if subset.empty:
            continue
        cum_avg = subset.set_index("date")["upvotes"].expanding().mean()
        # Resample to weekly, forward-fill gaps where no papers appeared
        weekly = cum_avg.resample("W").last().ffill()
        ax.plot(weekly.index, weekly.values, linewidth=1.2, alpha=0.85,
                color=cmap(i / len(slices)), label=label)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Avg Upvotes per Paper")
    ax.set_title("Cumulative Average Upvote Density Over Time by Paper Classification")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "e26_upvote_density_over_time_comparison.png")
    plt.close()
    print(f"  Saved e26_upvote_density_over_time_comparison.png")


# ── Group F: Correlation Analysis ─────────────────────────────────────────────


def plot_correlation(df: pd.DataFrame, x_col: str, xlabel: str, filepath: str):
    """Scatter plot with correlation coefficient."""
    valid = df[[x_col, "upvotes"]].dropna()
    corr = valid[x_col].corr(valid["upvotes"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid[x_col], valid["upvotes"], alpha=0.15, s=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Upvotes")
    ax.set_title(f"{xlabel} vs Upvotes (r = {corr:.3f}, n = {len(valid):,})")

    # Add trend line
    z = np.polyfit(valid[x_col], valid["upvotes"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
    ax.plot(x_range, p(x_range), "r--", alpha=0.8, label=f"Trend (slope={z[0]:.2f})")
    ax.legend()

    plt.savefig(filepath)
    plt.close()
    print(f"  Saved {filepath}")


def plot_group_f(df: pd.DataFrame):
    print("Group F: Correlation Analysis")
    plot_correlation(df, "num_authors", "# Authors", str(OUT_DIR / "f1_num_authors_vs_upvotes.png"))
    plot_correlation(df, "title_word_count", "# Words in Title", str(OUT_DIR / "f2_title_length_vs_upvotes.png"))
    plot_correlation(df, "abstract_word_count", "# Words in Abstract", str(OUT_DIR / "f3_abstract_length_vs_upvotes.png"))
    plot_correlation(df, "num_institutions", "# Unique Institutions", str(OUT_DIR / "f4_num_institutions_vs_upvotes.png"))

    # f5: upvote density by team size
    density_by_size = df.groupby("num_authors")["upvotes"].mean()
    # Limit to team sizes with at least 5 papers for stability
    counts_by_size = df["num_authors"].value_counts()
    valid_sizes = counts_by_size[counts_by_size >= 5].index
    density_by_size = density_by_size[density_by_size.index.isin(valid_sizes)].sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(density_by_size.index, density_by_size.values, edgecolor="black", alpha=0.7)
    ax.set_xlabel("# Authors on Paper")
    ax.set_ylabel("Average Upvotes")
    ax.set_title(f"Average Upvotes by Team Size (sizes with 5+ papers)")
    ax.axhline(df["upvotes"].mean(), color="red", linestyle="--",
               label=f"Overall mean: {df['upvotes'].mean():.1f}")
    ax.legend()
    plt.savefig(OUT_DIR / "f5_upvote_density_by_team_size.png")
    plt.close()
    print(f"  Saved f5_upvote_density_by_team_size.png")


# ── Group G: Institution Analysis ─────────────────────────────────────────────


def plot_group_g(author_df: pd.DataFrame):
    print("Group G: Institution Analysis")
    aff_df = author_df[author_df["affiliation"].str.strip() != ""].copy()
    authors_per_inst = aff_df.groupby("affiliation")["author_name"].nunique().reset_index()
    authors_per_inst.columns = ["affiliation", "num_unique_authors"]
    authors_per_inst = authors_per_inst.sort_values("num_unique_authors", ascending=False)
    authors_per_inst.head(100).to_csv(OUT_DIR / "g1_authors_per_institution.csv", index=False)
    print(f"  Saved g1_authors_per_institution.csv")


# ── Group H: Author Summary & Exhaustive Table ───────────────────────────────


def plot_group_h(df: pd.DataFrame, author_df: pd.DataFrame):
    print("Group H: Author Summary & Exhaustive Table")

    total_authors = author_df["author_name"].nunique()

    # Build per-author aggregates
    per_author = author_df.groupby("author_name").agg(
        num_papers=("paper_id", "nunique"),
        total_upvotes=("upvotes", "sum"),
        was_first_author=("is_first_author", "any"),
        was_last_author=("is_last_author", "any"),
        is_chinese_name=("is_chinese_name", "first"),
        has_chinese_affiliation=("is_chinese_affiliation", "any"),
        has_non_chinese_affiliation=("is_chinese_affiliation", lambda x: (~x).any()),
    ).reset_index()

    per_author["upvote_density"] = per_author["total_upvotes"] / per_author["num_papers"]

    # Solo author: appeared on a paper where they were the only author
    solo_papers = df[df["num_authors"] == 1]["paper_id"].values
    solo_author_names = author_df[author_df["paper_id"].isin(solo_papers)]["author_name"].unique()
    per_author["was_solo_author"] = per_author["author_name"].isin(solo_author_names)

    # Num unique affiliations
    affs = author_df.groupby("author_name")["affiliation"].apply(
        lambda x: len(set(a.strip().lower() for a in x if a.strip()))
    ).reset_index()
    affs.columns = ["author_name", "num_unique_affiliations"]
    per_author = per_author.merge(affs, on="author_name", how="left")

    # List of unique affiliations per author
    aff_lists = author_df.groupby("author_name")["affiliation"].apply(
        lambda x: "; ".join(sorted(set(a.strip() for a in x if a.strip())))
    ).reset_index()
    aff_lists.columns = ["author_name", "affiliations"]
    per_author = per_author.merge(aff_lists, on="author_name", how="left")

    # Derived booleans
    per_author["is_single_paper"] = per_author["num_papers"] == 1
    per_author["is_multi_paper"] = per_author["num_papers"] > 1
    per_author["is_single_affiliation"] = per_author["num_unique_affiliations"] == 1
    per_author["is_multi_affiliation"] = per_author["num_unique_affiliations"] > 1

    # Print summary stats
    stats = {
        "Total unique authors": total_authors,
        "First authors": per_author["was_first_author"].sum(),
        "Last authors": per_author["was_last_author"].sum(),
        "Chinese-name authors": per_author["is_chinese_name"].sum(),
        "Non-Chinese-name authors": (~per_author["is_chinese_name"]).sum(),
        "With Chinese affiliation": per_author["has_chinese_affiliation"].sum(),
        "With non-Chinese affiliation": per_author["has_non_chinese_affiliation"].sum(),
        "Solo authors (on a single-author paper)": per_author["was_solo_author"].sum(),
        "Single-paper authors": per_author["is_single_paper"].sum(),
        "Multi-paper authors": per_author["is_multi_paper"].sum(),
        "Single-affiliation authors": per_author["is_single_affiliation"].sum(),
        "Multi-affiliation authors": per_author["is_multi_affiliation"].sum(),
    }

    print("  --- Author Summary Statistics ---")
    summary_rows = []
    for label, count in stats.items():
        pct = count / total_authors * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")
        summary_rows.append({"metric": label, "count": count, "pct_of_total": round(pct, 2)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "h1_author_summary_stats.csv", index=False)
    print(f"  Saved h1_author_summary_stats.csv")

    # Exhaustive author table
    per_author_out = per_author[[
        "author_name", "num_papers", "total_upvotes", "upvote_density",
        "num_unique_affiliations", "affiliations",
        "was_first_author", "was_last_author",
        "was_solo_author", "is_chinese_name", "has_chinese_affiliation",
        "has_non_chinese_affiliation", "is_single_paper", "is_multi_paper",
        "is_single_affiliation", "is_multi_affiliation",
    ]].sort_values("num_papers", ascending=False)
    per_author_out.to_csv(OUT_DIR / "h2_exhaustive_author_table.csv", index=False)
    print(f"  Saved h2_exhaustive_author_table.csv ({len(per_author_out):,} authors)")

    # h3: exhaustive last names table
    def extract_surname(name):
        parts = name.strip().split()
        return parts[-1] if parts else ""

    author_df_with_surname = author_df.copy()
    author_df_with_surname["surname"] = author_df_with_surname["author_name"].apply(extract_surname)
    surname_stats = author_df_with_surname.groupby("surname").agg(
        num_papers=("paper_id", "nunique"),
        num_authors=("author_name", "nunique"),
    ).reset_index()
    surname_stats["is_chinese_surname"] = surname_stats["surname"].isin(CHINESE_SURNAMES)
    surname_stats = surname_stats.sort_values("num_papers", ascending=False)
    surname_stats.to_csv(OUT_DIR / "h3_exhaustive_surnames_table.csv", index=False)
    print(f"  Saved h3_exhaustive_surnames_table.csv ({len(surname_stats):,} unique surnames)")

    # h4: exhaustive affiliations table
    aff_rows = author_df[author_df["affiliation"].str.strip() != ""].copy()
    aff_stats = aff_rows.groupby("affiliation").agg(
        num_papers=("paper_id", "nunique"),
        num_authors=("author_name", "nunique"),
    ).reset_index()
    aff_stats["is_chinese_affiliation"] = aff_stats["affiliation"].apply(is_chinese_affiliation)
    aff_stats = aff_stats.sort_values("num_papers", ascending=False)
    aff_stats.to_csv(OUT_DIR / "h4_exhaustive_affiliations_table.csv", index=False)
    print(f"  Saved h4_exhaustive_affiliations_table.csv ({len(aff_stats):,} unique affiliations)")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = load_data()
    print(f"Loaded {len(df):,} papers\n")

    print("Exploding author info...")
    author_df = explode_authors(df)
    print(f"Created {len(author_df):,} author-paper rows\n")

    plot_group_a(df)
    print()
    plot_group_b(df)
    print()
    plot_group_c(df, author_df)
    print()
    plot_group_d(author_df)
    print()
    plot_group_e(df, author_df)
    print()
    plot_group_f(df)
    print()
    plot_group_g(author_df)
    print()
    plot_group_h(df, author_df)
    print()
    print("Done! All outputs saved to", OUT_DIR)


if __name__ == "__main__":
    main()
