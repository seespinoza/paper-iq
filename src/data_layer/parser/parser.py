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

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf4llm

PAPERS_PATH = Path("papers/")
PARSED_PAPERS_PATH = Path("papers/parsed_papers.parquet")
PARQUET_BATCH_SIZE = 100
PARQUET_BATCH_SCHEMA = pa.schema(
    [
        ("paper_path", pa.string()),
        ("paper_id", pa.string()),
        ("page_number", pa.int64()),
        ("text", pa.string()),
    ]
)


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


def _papers_to_process(
    papers_dir: Path, output_path: Path
) -> tuple[list[Path], pa.Table | None]:
    """Determine which PDFs still need parsing and load prior progress.

    Args:
        papers_dir: Directory containing paper PDFs to parse.
        output_path: Destination Parquet file, if one already exists.

    Returns:
        A tuple of ``(papers_to_process, existing_table)`` where
        ``papers_to_process`` excludes any paper already present in
        ``existing_table``, and ``existing_table`` is ``None`` if
        ``output_path`` doesn't exist yet.
    """
    papers_to_process = sorted(papers_dir.glob("*.pdf"))

    existing_table = None
    if output_path.is_file():
        existing_table = pq.read_table(output_path)
        processed = existing_table["paper_path"].to_pylist()
        papers_to_process = [p for p in papers_to_process if str(p) not in processed]

    return papers_to_process, existing_table


def _extract_and_write(
    papers_to_process: list[Path],
    new_rows_path: Path,
    batch_schema: pa.Schema,
    n_pages: int | None,
    ocr: bool,
) -> list[str]:
    """Extract text from each PDF and write it to a scratch Parquet file.

    Writes rows in batches of ``PARQUET_BATCH_SIZE`` papers via a
    ``ParquetWriter`` so extracted text doesn't have to be held in
    memory for the whole run. A PDF that fails to parse is skipped
    rather than aborting the whole run, since a single malformed file
    among thousands shouldn't block the rest.

    Args:
        papers_to_process: Paper PDFs to extract and write.
        new_rows_path: Scratch Parquet file to write the new rows to.
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

    with pq.ParquetWriter(new_rows_path, batch_schema, compression="zstd") as writer:
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

            # Batch writing
            if (paper_num + 1) % PARQUET_BATCH_SIZE == 0:
                table = pa.table(
                    {
                        "paper_path": paper_paths,
                        "paper_id": paper_ids,
                        "page_number": page_numbers,
                        "text": texts,
                    },
                    schema=batch_schema,
                )
                writer.write_table(table)
                paper_paths, paper_ids, page_numbers, texts = [], [], [], []

        # Remainder batch
        if paper_paths:
            table = pa.table(
                {
                    "paper_path": paper_paths,
                    "paper_id": paper_ids,
                    "page_number": page_numbers,
                    "text": texts,
                },
                schema=batch_schema,
            )
            writer.write_table(table)

    return failed


def _merge_and_publish(
    existing_table: pa.Table | None, new_rows_path: Path, output_path: Path
) -> None:
    """Merge newly written rows into ``output_path`` and clean up scratch files.

    Reads back the rows just written to ``new_rows_path``, concatenates
    them with ``existing_table`` (if any), and atomically replaces
    ``output_path`` with the merged result so ``output_path`` is never
    left partially written.

    Args:
        existing_table: Rows already present in ``output_path`` before
            this run, or ``None`` if it didn't exist yet.
        new_rows_path: Scratch Parquet file containing this run's new
            rows, written by :func:`_extract_and_write`.
        output_path: Destination Parquet file to publish the merged
            result to.
    """
    new_table = pq.read_table(new_rows_path)
    merged_table = (
        pa.concat_tables([existing_table, new_table])
        if existing_table is not None
        else new_table
    )

    merged_path = output_path.parent / f"{output_path.stem}.merged.tmp.parquet"
    pq.write_table(merged_table, merged_path, compression="zstd")
    os.replace(merged_path, output_path)
    new_rows_path.unlink()


def parse_papers_to_parquet(
    papers_dir: Path = PAPERS_PATH,
    output_path: Path = PARSED_PAPERS_PATH,
    batch_schema: pa.Schema = PARQUET_BATCH_SCHEMA,
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
    papers_to_process, existing_table = _papers_to_process(papers_dir, output_path)

    # New rows are written to a scratch file rather than output_path directly,
    # since opening a ParquetWriter on output_path would truncate it before
    # the existing rows are merged back in below.
    new_rows_path = output_path.parent / f"{output_path.stem}.new.tmp.parquet"

    failed = _extract_and_write(
        papers_to_process, new_rows_path, batch_schema, n_pages, ocr
    )

    _merge_and_publish(existing_table, new_rows_path, output_path)

    if failed:
        print(f"Failed to parse {len(failed)} paper(s): {', '.join(failed)}")

    return output_path


if __name__ == "__main__":
    start = time.perf_counter()
    out = parse_papers_to_parquet()
    print(f"Wrote parsed papers to {out}")
    print(f"Time spent: {time.perf_counter() - start}")
