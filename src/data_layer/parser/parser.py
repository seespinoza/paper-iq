"""Extract text from downloaded arXiv paper PDFs.

This module is part of the "extract" stage of the data layer: it reads
PDFs already downloaded to disk by ``arxiv_api.download_paper_pdf`` and
converts them to per-page markdown text using ``pymupdf4llm``. A
separate loader module is responsible for ingesting the extracted text
into a database.
"""

import os
import time
from pathlib import Path

import pandas as pd
import pymupdf4llm

PAPERS_PATH = Path("papers/")
PARSED_PAPERS_PATH = Path("papers/parsed_papers.parquet")
PARQUET_BATCH_SIZE = 100
PARQUET_BATCH_SCHEMA = {
    "paper_path": "string",
    "paper_id": "string",
    "page_number": "int64",
    "text": "string",
}


def extract_paper_text(
    pdf_path: Path, n_pages: int | None = None, ocr: bool = True
) -> list[str]:
    """Extract per-page markdown text from a paper PDF.

    Args:
        pdf_path: Path to the paper's PDF file.
        n_pages: Number of leading pages to extract, starting from the
            first page. Defaults to all pages in the document.
        ocr: Boolean indicating if OCR will be used (PyMuPDF4LLM)
        uses it only when necessary (i.e., for tables and figure).

    Returns:
        List of markdown strings, one per extracted page, in page
        order.
    """
    doc = pymupdf4llm.pymupdf.open(pdf_path)
    n = doc.page_count if n_pages is None else min(n_pages, doc.page_count)
    chunks = pymupdf4llm.to_markdown(
        doc, pages=list(range(n)), page_chunks=True, use_ocr=ocr
    )
    doc.close()
    return [chunk["text"] for chunk in chunks]


def _papers_to_process(papers_dir: Path, output_path: Path) -> list[Path]:
    """Determine which PDFs in ``papers_dir`` still need parsing.

    Args:
        papers_dir: Directory containing paper PDFs to parse.
        output_path: Destination Parquet file, if one already exists.

    Returns:
        Paper PDFs in ``papers_dir`` not already present in
        ``output_path``.
    """
    papers_to_process = sorted(papers_dir.glob("*.pdf"))

    if output_path.is_file():
        processed = pd.read_parquet(output_path, columns=["paper_path"])
        processed = processed["paper_path"].to_list()
        papers_to_process = [p for p in papers_to_process if str(p) not in processed]

    return papers_to_process


def _flush_batch(
    output_path: Path,
    paper_paths: list[str],
    paper_ids: list[str],
    page_numbers: list[int],
    texts: list[str],
    batch_schema: dict[str, str],
) -> None:
    """Write one accumulated batch of rows to ``output_path``.

    Appends the batch to any rows already at ``output_path`` and
    writes the combined result to a scratch file, which is then
    atomically swapped in for ``output_path`` so it's never left
    partially written if the process is interrupted mid-write.

    Args:
        output_path: Destination Parquet file to append the batch to.
        paper_paths: Accumulated ``paper_path`` values for this batch.
        paper_ids: Accumulated ``paper_id`` values for this batch.
        page_numbers: Accumulated ``page_number`` values for this batch.
        texts: Accumulated ``text`` values for this batch.
        batch_schema: Schema the batch conforms to.
    """
    df = pd.DataFrame(
        {
            "paper_path": paper_paths,
            "paper_id": paper_ids,
            "page_number": page_numbers,
            "text": texts,
        }
    )

    df = df.astype(batch_schema)

    # Append to current file if one exists
    if output_path.is_file():
        current_df = pd.read_parquet(output_path)
        df = pd.concat([current_df, df], ignore_index=True)

    tmp_path = output_path.parent / f"{output_path.stem}.tmp.parquet"

    # Write to temp path to avoid file corruption
    df.to_parquet(tmp_path, compression="zstd")
    os.replace(tmp_path, output_path)


def _extract_and_write(
    papers_to_process: list[Path],
    output_path: Path,
    batch_schema: dict[str, str],
    n_pages: int | None,
    ocr: bool,
) -> list[str]:
    """Extract text from each PDF and write it to ``output_path``.

    Writes rows in batches of ``PARQUET_BATCH_SIZE`` papers via
    :func:`_flush_batch` so extracted text doesn't have to be held in
    memory for the whole run. A PDF that fails to parse is skipped
    rather than aborting the whole run, since a single malformed file
    among thousands shouldn't block the rest.

    Args:
        papers_to_process: Paper PDFs to extract and write.
        output_path: Destination Parquet file to write extracted rows to.
        batch_schema: Schema each written batch conforms to.
        n_pages: Number of leading pages to extract per paper.
            Defaults to all pages.
        ocr: Boolean indicating if OCR will be used during extraction.

    Returns:
        Filenames of PDFs that failed to parse.
    """
    paper_paths: list[str] = []
    paper_ids: list[str] = []
    page_numbers: list[int] = []
    texts: list[str] = []
    failed: list[str] = []

    for paper_num, pdf_path in enumerate(papers_to_process):
        try:
            pages = extract_paper_text(pdf_path, n_pages=n_pages, ocr=ocr)
        except Exception:
            failed.append(pdf_path.name)
            continue
        for page_number, text in enumerate(pages):
            paper_paths.append(str(pdf_path))
            paper_ids.append(pdf_path.stem)
            page_numbers.append(page_number)
            texts.append(text)

        # Batch write and clear lists
        if (paper_num + 1) % PARQUET_BATCH_SIZE == 0:
            _flush_batch(
                output_path, paper_paths, paper_ids, page_numbers, texts, batch_schema
            )
            paper_paths, paper_ids, page_numbers, texts = [], [], [], []

    # Remainder batch
    if paper_paths:
        _flush_batch(
            output_path, paper_paths, paper_ids, page_numbers, texts, batch_schema
        )

    return failed


def parse_papers_to_parquet(
    papers_dir: Path = PAPERS_PATH,
    output_path: Path = PARSED_PAPERS_PATH,
    batch_schema: dict[str, str] = PARQUET_BATCH_SCHEMA,
    n_pages: int | None = None,
    ocr: bool = True,
) -> Path:
    """Extract text from every PDF in a directory and write it to Parquet.

    Extracts each PDF's per-page markdown text via
    :func:`extract_paper_text` and writes the results to a single
    Parquet (zstd-compressed) file with one row per page. Already
    processed papers are skipped and prior progress is preserved
    across runs.

    Args:
        papers_dir: Directory containing paper PDFs to parse.
        output_path: Destination Parquet file.
        n_pages: Number of leading pages to extract per paper.
            Defaults to all pages.
        ocr: Boolean indicating if OCR will be used during extraction.

    Returns:
        The path the Parquet file was written to.
    """
    papers_to_process = _papers_to_process(papers_dir, output_path)

    failed = _extract_and_write(
        papers_to_process, output_path, batch_schema, n_pages, ocr
    )

    if failed:
        print(f"Failed to parse {len(failed)} paper(s): {', '.join(failed)}")

    return output_path


if __name__ == "__main__":
    start = time.perf_counter()
    out = parse_papers_to_parquet()
    print(f"Wrote parsed papers to {out}")
    print(f"Time spent: {time.perf_counter() - start}")
