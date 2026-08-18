import pandas as pd
import pytest
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from data_layer.chunker.lang_chain_chunker import (
    fixed_size_chunk,
    markdown_chunk,
    process_corpus,
)

HEADERS = [("#", "h1"), ("##", "h2")]

FIXED_SIZE_DOCUMENT_DF = pd.DataFrame(
    {
        "paper_id": [1, 2, 3],
        "paper_path": ["paper_1.pdf", "paper_2.pdf", "paper_3.pdf"],
        "num_pages": [5, 3, 7],
        "text": [
            "Jack and Sebastian ran to the store",
            "Scout jumped over the moon.",
            "Tables were set and moved! 345656576 \n",
        ],
    }
)

MARKDOWN_DOCUMENT_DF = pd.DataFrame(
    {
        "paper_id": [1, 2],
        "paper_path": ["paper_1.pdf", "paper_2.pdf"],
        "num_pages": [1, 2],
        "text": [
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


@pytest.mark.parametrize(
    "text, expected_chunks, expected_metadata",
    [
        pytest.param("", [], [], id="no_text"),
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
                "# 1\n" + "\n".join(["aaa"] * 50),
                "## 1.2\n" + "\n".join(["aaa"] * 50),
                "# 2\n" + "b" * 50,
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
def test_markdown_chunk(text, expected_chunks, expected_metadata):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS, strip_headers=False
    )

    chunks, metadata = markdown_chunk(splitter, text)

    assert chunks == expected_chunks
    assert metadata == expected_metadata


@pytest.mark.parametrize(
    "text, chunk_size, chunk_overlap, expected_chunks",
    [
        pytest.param("", 10, 2, [], id="no_text"),
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
def test_fixed_size_chunk(text, chunk_size, chunk_overlap, expected_chunks):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = fixed_size_chunk(splitter, text)

    assert chunks == expected_chunks


def test_process_corpus_fixed_size():

    process_df = process_corpus(
        FIXED_SIZE_DOCUMENT_DF, strategy="fs", chunk_size=10, chunk_overlap=2
    )

    expected_df = pd.DataFrame.from_records(
        [
            {"paper_id": 1, "paper_path": "paper_1.pdf", "num_pages": 5, "chunk_id": 0, "chunk_text": "Jack and", "chunk_char_count": 8},
            {"paper_id": 1, "paper_path": "paper_1.pdf", "num_pages": 5, "chunk_id": 1, "chunk_text": "Sebastian", "chunk_char_count": 9},
            {"paper_id": 1, "paper_path": "paper_1.pdf", "num_pages": 5, "chunk_id": 2, "chunk_text": "ran to", "chunk_char_count": 6},
            {"paper_id": 1, "paper_path": "paper_1.pdf", "num_pages": 5, "chunk_id": 3, "chunk_text": "the store", "chunk_char_count": 9},
            {"paper_id": 2, "paper_path": "paper_2.pdf", "num_pages": 3, "chunk_id": 4, "chunk_text": "Scout", "chunk_char_count": 5},
            {"paper_id": 2, "paper_path": "paper_2.pdf", "num_pages": 3, "chunk_id": 5, "chunk_text": "jumped", "chunk_char_count": 6},
            {"paper_id": 2, "paper_path": "paper_2.pdf", "num_pages": 3, "chunk_id": 6, "chunk_text": "over the", "chunk_char_count": 8},
            {"paper_id": 2, "paper_path": "paper_2.pdf", "num_pages": 3, "chunk_id": 7, "chunk_text": "moon.", "chunk_char_count": 5},
            {"paper_id": 3, "paper_path": "paper_3.pdf", "num_pages": 7, "chunk_id": 8, "chunk_text": "Tables", "chunk_char_count": 6},
            {"paper_id": 3, "paper_path": "paper_3.pdf", "num_pages": 7, "chunk_id": 9, "chunk_text": "were set", "chunk_char_count": 8},
            {"paper_id": 3, "paper_path": "paper_3.pdf", "num_pages": 7, "chunk_id": 10, "chunk_text": "and", "chunk_char_count": 3},
            {"paper_id": 3, "paper_path": "paper_3.pdf", "num_pages": 7, "chunk_id": 11, "chunk_text": "moved!", "chunk_char_count": 6},
            {"paper_id": 3, "paper_path": "paper_3.pdf", "num_pages": 7, "chunk_id": 12, "chunk_text": "345656576", "chunk_char_count": 9},
        ]
    )

    pd.testing.assert_frame_equal(
        process_df.reset_index(drop=True).sort_index(axis=1),
        expected_df.reset_index(drop=True).sort_index(axis=1),
        check_dtype=False,
    )


def test_process_corpus_markdown():

    process_df = process_corpus(
        MARKDOWN_DOCUMENT_DF,
        strategy="md",
        headers_to_split_on=HEADERS,
        strip_headers=False,
    )

    chunk_2a = "# 1\n" + "\n".join(["aaa"] * 50)
    chunk_2b = "## 1.2\n" + "\n".join(["aaa"] * 50)
    chunk_2c = "# 2\n" + "b" * 50

    expected_df = pd.DataFrame.from_records(
        [
            {
                "paper_id": 1,
                "paper_path": "paper_1.pdf",
                "chunk_id": 0,
                "h1": "title",
                "h2": None,
                "h3": None,
                "h4": None,
                "chunk_text": "# title\nthis is a test",
                "chunk_char_count": 22,
                "chunk_word_count": 6,
            },
            {
                "paper_id": 2,
                "paper_path": "paper_2.pdf",
                "chunk_id": 1,
                "h1": "1",
                "h2": None,
                "h3": None,
                "h4": None,
                "chunk_text": chunk_2a,
                "chunk_char_count": 203,
                "chunk_word_count": 52,
            },
            {
                "paper_id": 2,
                "paper_path": "paper_2.pdf",
                "chunk_id": 2,
                "h1": "1",
                "h2": "1.2",
                "h3": None,
                "h4": None,
                "chunk_text": chunk_2b,
                "chunk_char_count": 206,
                "chunk_word_count": 52,
            },
            {
                "paper_id": 2,
                "paper_path": "paper_2.pdf",
                "chunk_id": 3,
                "h1": "2",
                "h2": None,
                "h3": None,
                "h4": None,
                "chunk_text": chunk_2c,
                "chunk_char_count": 54,
                "chunk_word_count": 3,
            },
        ]
    )

    pd.testing.assert_frame_equal(
        process_df.reset_index(drop=True).sort_index(axis=1),
        expected_df.reset_index(drop=True).sort_index(axis=1),
        check_dtype=False,
    )


def test_process_corpus_empty_dataframe():

    empty_df = pd.DataFrame({"paper_id": [], "paper_path": [], "num_pages": [], "text": []})

    process_df = process_corpus(
        empty_df, strategy="fs", chunk_size=10, chunk_overlap=2
    )

    assert process_df.empty
