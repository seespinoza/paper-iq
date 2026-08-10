"""Extract text from downloaded arXiv paper PDFs.

This module is part of the "extract" stage of the data layer: it reads
PDFs already downloaded to disk by ``arxiv_api.download_paper_pdf`` and
converts them to per-page markdown text using ``pymupdf4llm``. A
separate loader module is responsible for ingesting the extracted text
into a database.
"""

from pathlib import Path

import pymupdf4llm

PAPERS_PATH = Path("papers/")


def extract_paper_text(pdf_path: Path, n_pages: int | None = None) -> list[str]:
    """Extract per-page markdown text from a paper PDF.

    Args:
        pdf_path: Path to the paper's PDF file.
        n_pages: Number of leading pages to extract, starting from the
            first page. Defaults to all pages in the document.

    Returns:
        List of markdown strings, one per extracted page, in page
        order.
    """
    doc = pymupdf4llm.pymupdf.open(pdf_path)
    n = doc.page_count if n_pages is None else min(n_pages, doc.page_count)
    chunks = pymupdf4llm.to_markdown(
        doc, pages=list(range(n)), page_chunks=True, use_ocr=True
    )
    doc.close()
    return [chunk["text"] for chunk in chunks]
