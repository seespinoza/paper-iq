"""Extract text from downloaded arXiv paper PDFs.

This module is part of the "extract" stage of the data layer: it reads
PDFs already downloaded to disk by ``arxiv_api.download_paper_pdf`` and
converts them to per-page markdown text using ``pymupdf4llm``. A
separate loader module is responsible for ingesting the extracted text
into a database.
"""

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
    Parquet (zstd-compressed) file with one row per page. A PDF that
    fails to parse is skipped rather than aborting the whole run, since
    a single malformed file among thousands shouldn't block the rest.

    Args:
        papers_dir: Directory containing paper PDFs to parse.
        output_path: Destination Parquet file.
        n_pages: Number of leading pages to extract per paper.
            Defaults to all pages.
        ocr: Boolean indicating if OCR will be used during extraction.

    Returns:
        The path the Parquet file was written to.
    """
    paper_paths: list[str] = []
    paper_ids: list[str] = []
    page_numbers: list[int] = []
    texts: list[str] = []
    failed: list[str] = []

    papers_to_process = sorted(papers_dir.glob("*.pdf"))

    # Check which papers have already been processed
    if output_path.is_file():
        table = pq.read_table(output_path)
        processed = table["paper_path"].to_pylist()

        papers_to_process = [p for p in papers_to_process if p not in processed]

    with pq.ParquetWriter(output_path, batch_schema, compression="zstd"):
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
            if paper_num % PARQUET_BATCH_SIZE == 0:
                table = pa.table(
                    {
                        "paper_path": pdf_path,
                        "paper_id": paper_ids,
                        "page_number": page_numbers,
                        "text": texts,
                    }
                )
                pq.write_table(table, output_path, compression="zstd")
                paper_paths, paper_ids, page_numbers, texts = [], [], [], []

    if failed:
        print(f"Failed to parse {len(failed)} paper(s): {', '.join(failed)}")

    return output_path


if __name__ == "__main__":
    out = parse_papers_to_parquet()
    print(f"Wrote parsed papers to {out}")
