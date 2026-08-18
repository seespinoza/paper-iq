import pandas as pd
import pytest

from data_layer.chunker.lang_chain_chunker import (
    fixed_size_chunk,
    markdown_chunk,
    process_corpus,
)

FIXED_SIZE_DOCUMENT_DF = pd.DataFrame(
    {
        "doc_id": [1, 2, 3],
        "doc_text": [
            "Jack and Sebastian ran to the store",
            "Scout jumped over the moon.",
            "Tables were set and moved! 345656576 \n",
        ],
    }
)

MARKDOWN_DOCUMENT_DF = pd.DataFrame(
    {
        "doc_id": [1, 2],
        "doc_text": [
            "# title\nthis is a test",
            "".join(
                [
                    "# 1\n",
                    "aaa\n" * 50,
                    "## 1.2\n",
                    "aaa\n" * 50,
                    "# 2\n",
                    "b" * 50,
                ]
            ),
        ],
    }
)

HEADERS = [("#", "h1"), ("##", "h2")]


@pytest.mark.parametrize(
    "text, expected_chunks, expected_metadata",
    [
        pytest.param("", [], [{}], id="no_text"),
        pytest.param(
            "this is a test",
            ["this is a test"],
            [{}],
            id="no_headers",
        ),
        pytest.param(
            "# title\nthis is a test",
            ["# title\nthis is a test"],
            [{"h1": "title"}],
            id="single_header",
        ),
        pytest.param(
            "".join(
                [
                    "# 1\n",
                    "aaa\n" * 50,
                    "## 1.2\n",
                    "aaa\n" * 50,
                    "# 2\n",
                    "b" * 50,
                ]
            ),
            [
                "".join(["# 1\n", "aaa\n" * 50]),
                "".join(["## 1.2\n", "aaa\n" * 50]),
                "".join(["# 2\n", "b" * 50]),
            ],
            [
                {"h1": "1"},
                {"h1": "1", "h2": "1.2"},
                {"h1": "2"},
            ],
            id="nested_headers",
        ),
    ],
)
def test_markdown_chunker(text, expected_chunks, expected_metadata):

    chunks, metadata = markdown_chunk(
        headers_to_split_on=HEADERS,
        strip_headers=False,
        text=text,
    )

    assert chunks == expected_chunks
    assert metadata == expected_metadata


@pytest.mark.parametrize(
    "text, chunk_size, chunk_overlap, expected_chunks",
    [
        pytest.param("", None, None, [], id="no_text"),
        pytest.param(
            "abc def ghi",
            3,
            1,
            ["abc", "de", "ef", "gh", "hi"],
            id="space_and_alphabet",
        ),
        pytest.param(
            "# title\n\nthis is a test",
            5,
            2,
            ["#", "titl", "tle", "this", "is a", "test"],
            id="space_newline_and_alphabet",
        ),
    ],
)
def test_fixed_size_chunker(text, chunk_size, chunk_overlap, expected_chunks):

    chunks = fixed_size_chunk(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, text=text
    )

    assert chunks == expected_chunks


def _expected_corpus_df(document_df, strategy, **chunk_kwargs):

    records = []
    for row in document_df.itertuples(index=False):
        if strategy == "fixed_size":
            chunks = fixed_size_chunk(text=row.doc_text, **chunk_kwargs)
            metadata = [{}] * len(chunks)
        elif strategy == "markdown":
            chunks, metadata = markdown_chunk(text=row.doc_text, **chunk_kwargs)
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        for chunk_id, (chunk, meta) in enumerate(zip(chunks, metadata, strict=True)):
            records.append(
                {
                    "doc_id": row.doc_id,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk,
                    **meta,
                }
            )

    return pd.DataFrame.from_records(records)


@pytest.mark.parametrize(
    "document_df, strategy, chunk_kwargs",
    [
        pytest.param(
            FIXED_SIZE_DOCUMENT_DF,
            "fixed_size",
            {"chunk_size": 10, "chunk_overlap": 2},
            id="fixed_size",
        ),
        pytest.param(
            MARKDOWN_DOCUMENT_DF,
            "markdown",
            {"headers_to_split_on": HEADERS, "strip_headers": False},
            id="markdown",
        ),
    ],
)
def test_process_corpus(document_df, strategy, chunk_kwargs):

    process_df = process_corpus(document_df, strategy=strategy, **chunk_kwargs)

    expected_df = _expected_corpus_df(document_df, strategy, **chunk_kwargs)

    pd.testing.assert_frame_equal(
        process_df.reset_index(drop=True).sort_index(axis=1),
        expected_df.reset_index(drop=True).sort_index(axis=1),
        check_dtype=False,
    )


def test_process_corpus_empty_dataframe():

    empty_df = pd.DataFrame({"doc_id": [], "doc_text": []})

    process_df = process_corpus(
        empty_df, strategy="fixed_size", chunk_size=10, chunk_overlap=2
    )

    assert process_df.empty
