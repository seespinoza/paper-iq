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


def markdown_chunk(
    splitter: MarkdownHeaderTextSplitter, text: str
) -> tuple[list[str], list[dict[str, str]]]:
    chunk_list = []
    metadata_list = []

    chunks = splitter.split_text(text)

    for chunk in chunks:
        chunk_list.append(chunk.page_content)
        metadata_list.append(chunk.metadata)

    return chunk_list, metadata_list


def fixed_size_chunk(splitter: RecursiveCharacterTextSplitter, text: str) -> list[str]:

    chunks = splitter.split_text(text)
    return chunks


def process_corpus(
    corpus_df: pd.DataFrame, strategy: str, **kwargs: dict[str, str]
) -> pd.DataFrame:

    chunks_df = pd.DataFrame()
    records = []
    splitter = None

    if strategy == "md":
        splitter = MarkdownHeaderTextSplitter(**kwargs)

        for doc in corpus_df.itertuples(index=False):
            chunks = markdown_chunk(splitter, doc.text)
            for i, chunk in enumerate(chunks):
                records.append(
                    {
                        "paper_id": doc.paper_id,
                        "paper_path": doc.paper_path,
                        "num_pages": doc.num_pages,
                        "chunk_id": i,
                        "chunk_text": chunk,
                        "chunk_char_count": len(chunk),
                    }
                )
        chunks_df = pd.DataFrame.from_records(records)

    elif strategy == "fs":  # fixed-size
        splitter = RecursiveCharacterTextSplitter(**kwargs)

        for doc in corpus_df.itertuples(index=False):
            chunks, metadata = markdown_chunk(splitter, doc.text)
            chunk_id = 0
            for sub_chunk, sub_metadata in zip(chunks, metadata, strict=True):
                records.append(
                    {
                        "paper_id": doc.paper_id,
                        "paper_path": doc.paper_path,
                        "chunk_id": chunk_id,
                        "h1": sub_metadata.get("h1"),
                        "h2": sub_metadata.get("h2"),
                        "h3": sub_metadata.get("h3"),
                        "h4": sub_metadata.get("h4"),
                        "chunk_text": sub_chunk,
                        "chunk_char_count": len(sub_chunk),
                        "chunk_word_count": len(sub_chunk.split()),
                    }
                )
                chunk_id += 1
        chunks_df = pd.DataFrame.from_records(records)

    return chunks_df


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
