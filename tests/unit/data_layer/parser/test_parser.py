from pathlib import Path

from data_layer.parser.parser import extract_paper_text


def make_mock_doc(mocker, page_count: int):
    doc = mocker.Mock(page_count=page_count)
    return doc


def make_chunks(n: int) -> list[dict]:
    return [{"text": f"page {i} text"} for i in range(n)]


# extract paper text
def test_extract_paper_text_default_pages(mocker):
    doc = make_mock_doc(mocker, page_count=3)
    mock_open = mocker.patch("data_layer.parser.parser.pymupdf4llm.pymupdf.open")
    mock_open.return_value = doc
    mock_to_markdown = mocker.patch("data_layer.parser.parser.pymupdf4llm.to_markdown")
    mock_to_markdown.return_value = make_chunks(3)

    result = extract_paper_text(Path("papers/2607.00292v1.pdf"))

    assert result == ["page 0 text", "page 1 text", "page 2 text"]
    mock_to_markdown.assert_called_once_with(
        doc, pages=[0, 1, 2], page_chunks=True, use_ocr=True
    )
    doc.close.assert_called_once()


def test_extract_paper_text_limited_pages(mocker):
    doc = make_mock_doc(mocker, page_count=10)
    mock_open = mocker.patch("data_layer.parser.parser.pymupdf4llm.pymupdf.open")
    mock_open.return_value = doc
    mock_to_markdown = mocker.patch("data_layer.parser.parser.pymupdf4llm.to_markdown")
    mock_to_markdown.return_value = make_chunks(2)

    result = extract_paper_text(Path("papers/2607.00292v1.pdf"), n_pages=2)

    assert result == ["page 0 text", "page 1 text"]
    mock_to_markdown.assert_called_once_with(
        doc, pages=[0, 1], page_chunks=True, use_ocr=True
    )
    doc.close.assert_called_once()


def test_extract_paper_text_n_pages_exceeds_available(mocker):
    doc = make_mock_doc(mocker, page_count=3)
    mock_open = mocker.patch("data_layer.parser.parser.pymupdf4llm.pymupdf.open")
    mock_open.return_value = doc
    mock_to_markdown = mocker.patch("data_layer.parser.parser.pymupdf4llm.to_markdown")
    mock_to_markdown.return_value = make_chunks(3)

    result = extract_paper_text(Path("papers/2607.00292v1.pdf"), n_pages=9)

    assert len(result) == 3
    mock_to_markdown.assert_called_once_with(
        doc, pages=[0, 1, 2], page_chunks=True, use_ocr=True
    )


def test_extract_paper_text_empty_document(mocker):
    doc = make_mock_doc(mocker, page_count=0)
    mock_open = mocker.patch("data_layer.parser.parser.pymupdf4llm.pymupdf.open")
    mock_open.return_value = doc
    mock_to_markdown = mocker.patch("data_layer.parser.parser.pymupdf4llm.to_markdown")
    mock_to_markdown.return_value = []

    result = extract_paper_text(Path("papers/empty.pdf"))

    assert result == []
    mock_to_markdown.assert_called_once_with(
        doc, pages=[], page_chunks=True, use_ocr=True
    )
    doc.close.assert_called_once()


def test_extract_paper_text_closes_doc_path_passed_through(mocker):
    doc = make_mock_doc(mocker, page_count=1)
    mock_open = mocker.patch("data_layer.parser.parser.pymupdf4llm.pymupdf.open")
    mock_open.return_value = doc
    mocker.patch(
        "data_layer.parser.parser.pymupdf4llm.to_markdown",
        return_value=make_chunks(1),
    )

    pdf_path = Path("papers/2607.00292v1.pdf")
    extract_paper_text(pdf_path)

    mock_open.assert_called_once_with(pdf_path)
