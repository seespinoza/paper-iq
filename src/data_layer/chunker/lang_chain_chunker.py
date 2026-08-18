from pathlib import Path

import pandas as pd
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

PAPER_PATH = Path("../papers/")


def _combine_pages(pages: pd.DataFrame):

    return pd.Series(
        {
            "paper_id": pages["paper_id"].iloc[0],
            "text": "\n".join(pages.sort_values("page_number")["text"]),
            "num_pages": len(pages),
        }
    )


def markdown_chunk(splitter: MarkdownHeaderTextSplitter, text: str) -> None:
    sub_chunks = splitter.split_text()
    return ("", {})


def fixed_size_chunk(splitter: RecursiveCharacterTextSplitter, text: str) -> list[str]:

    chunks = splitter.split_text(text)
    return chunks


def process_corpus(
    df: pd.DataFrame, strategy: str, **kwargs: dict[str, str]
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    return pd.DataFrame()


if __name__ == "__main__":
    df = pd.read_parquet(PAPER_PATH / "parsed_papers.parquet")
    document_df = (
        df.groupby("paper_path")
        .apply(_combine_pages, include_groups=False)
        .reset_index()
    )
    document_df = document_df[
        document_df["num_pages"] <= 200
    ].copy()  # Remove books and dissertations
