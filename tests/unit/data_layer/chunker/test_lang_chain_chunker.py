import pandas as pd

from data_layer.chunker.lang_chain_chunker import (
    fixed_size_chunk,
    markdown_chunk,
    process_corpus,
)

DOCUMENT_DF = pd.DataFrame(
    {
        "doc_id": [1, 2, 3],
        "doc_text": [
            "Jack and Sebastian ran to the store",
            "Scout jumped over the moon.",
            "Tables were set and moved! 345656576 \n",
        ],
    }
)

HEADERS = [("#", "h1"), ("##", "h2")]


def test_markdown_chunker():

    chunks, metadata = markdown_chunk(
        headers_to_split_on=HEADERS,
        strip_headers=False,
        text=DOCUMENT_DF.loc[0, "doc_text"],
    )

    assert chunks == []
    assert metadata == []

    chunks, metadata = markdown_chunk(
        headers_to_split_on=HEADERS,
        strip_headers=False,
        text=DOCUMENT_DF.loc[1, "doc_text"],
    )

    assert chunks == []
    assert metadata == []

    chunks, metadata = markdown_chunk(
        headers_to_split_on=HEADERS,
        strip_headers=False,
        text=DOCUMENT_DF.loc[2, "doc_text"],
    )

    assert chunks == []
    assert metadata == []


def test_fixed_size_chunker_empty_string():

    chunks, metadata = fixed_size_chunk(
        headers_to_split_on=HEADERS,
        strip_headers=False,
        text="",
    )

    assert chunks == []
    assert metadata == []

    assert chunks == []
    assert metadata == []


def test_process_corpus():

    process_df = process_corpus(DOCUMENT_DF)

    assert process_df == pd.DataFrame()
