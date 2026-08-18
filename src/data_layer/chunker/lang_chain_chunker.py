import pandas as pd


def markdown_chunk(
    text: str, headers_to_split_on: list[tuple[str, str]], strip_headers: bool = False
) -> None:
    return ("", {})


def fixed_size_chunk(
    text: str, chunk_size: int, chunk_overlap: int, length_function: str = "len"
) -> None:
    return ("", "")


def process_corpus(
    df: pd.DataFrame, strategy: str, **kwargs: dict[str, str]
) -> pd.DataFrame:
    return pd.DataFrame()
