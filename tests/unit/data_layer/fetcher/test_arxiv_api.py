from pathlib import Path

import pytest
import requests

from data_layer.fetcher.arxiv_api import (
    _parse_feed,
    count_number_avail_papers,
    download_paper_pdf,
    estimate_size_of_download_gb,
    fetch_paper_catalog,
)

FIXTURE_DIR = Path(__file__).parent


def load_fixture_xml(name: str):
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


# parse xml catalog
def test_parse_feed_extract_entries():
    xml = load_fixture_xml("successful_arxiv_response.xml")
    entries, total = _parse_feed(xml)
    assert len(entries) == 1
    assert total == 192137


def test_parse_feed_empty():
    xml = load_fixture_xml("successful_arxiv_response_invalid_cat.xml")
    entries, total = _parse_feed(xml)
    assert len(entries) == 0
    assert total == 0


# count number of papers
def test_count_number_avail_papers(mocker):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mock_get.return_value = mocker.Mock(
        status_code=200, text=load_fixture_xml("successful_arxiv_response.xml")
    )

    result = count_number_avail_papers("cs.AI", "202607010000", "202608010000")
    assert result == 192137


def test_count_number_avail_papers_empty(mocker):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mock_get.return_value = mocker.Mock(
        status_code=200,
        text=load_fixture_xml("successful_arxiv_response_invalid_cat.xml"),
    )

    result = count_number_avail_papers(
        "cs.some_invalid_category", "202607010000", "202608010000"
    )
    assert result == 0


# count estimated size of download
def test_estimate_size_of_download_gb(mocker):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mock_get.return_value = mocker.Mock(
        status_code=200,
        text=load_fixture_xml("successful_arxiv_response.xml"),
    )

    result = estimate_size_of_download_gb("cs.AI", "202607010000", "202608010000", 0.2)
    assert result == 192137 * 0.2


def test_estimate_size_of_download_gb_empty(mocker):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mock_get.return_value = mocker.Mock(
        status_code=200,
        text=load_fixture_xml("successful_arxiv_response_invalid_cat.xml"),
    )

    result = estimate_size_of_download_gb("cs.AI", "202607010000", "202608010000", 0.2)
    assert result == 0


# download catalog
def test_fetch_paper_catalog_one_page(mocker):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mocker.patch("data_layer.fetcher.arxiv_api.time.sleep")

    mock_get.side_effect = [
        mocker.Mock(
            status_code=200, text=load_fixture_xml("successful_arxiv_response.xml")
        ),
        mocker.Mock(
            status_code=200,
            text=load_fixture_xml("successful_arxiv_response_invalid_cat.xml"),
        ),
    ]

    result = fetch_paper_catalog("cs.AI", "202607010000", "202608010000")

    assert len(result) == 1
    assert mock_get.call_count == 2


def test_fetch_paper_catalog_multiple_pages(mocker):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mocker.patch("data_layer.fetcher.arxiv_api.time.sleep")

    mock_get.side_effect = [
        mocker.Mock(
            status_code=200, text=load_fixture_xml("successful_arxiv_response.xml")
        ),
        mocker.Mock(
            status_code=200, text=load_fixture_xml("successful_arxiv_response.xml")
        ),
        mocker.Mock(
            status_code=200,
            text=load_fixture_xml("successful_arxiv_response_invalid_cat.xml"),
        ),
    ]

    result = fetch_paper_catalog("cs.AI", "202607010000", "202608010000")

    assert len(result) == 2
    assert mock_get.call_count == 3


def test_fetch_paper_catalog_empty(mocker):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mocker.patch("data_layer.fetcher.arxiv_api.time.sleep")

    mock_get.side_effect = [
        mocker.Mock(
            status_code=200,
            text=load_fixture_xml("successful_arxiv_response_invalid_cat.xml"),
        ),
    ]

    result = fetch_paper_catalog("cs.AI", "202607010000", "202608010000")

    assert len(result) == 0
    assert mock_get.call_count == 1


def test_fetch_paper_catalog_http_error(mocker):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mocker.patch("data_layer.fetcher.arxiv_api.time.sleep")

    mock_response = mocker.Mock(status_code=500)
    mock_response.raise_for_status.side_effect = requests.HTTPError("server error")
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        fetch_paper_catalog("cs.AI", "202607010000", "202608010000")


# download pdf
def test_download_paper(mocker, tmp_path):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mocker.patch("data_layer.fetcher.arxiv_api.time.sleep")
    mock_get.return_value = mocker.Mock(status_code=200, content=b"fake pdf bytes")

    entries, _ = _parse_feed(load_fixture_xml("successful_arxiv_response.xml"))
    real_entry = entries[0]

    download_paper_pdf(real_entry, tmp_path)

    arxiv_id = real_entry.id.split("/abs/")[-1]
    saved_file = tmp_path / f"{arxiv_id}.pdf"
    assert saved_file.exists()
    assert saved_file.read_bytes() == b"fake pdf bytes"


def test_download_paper_invalid(mocker, tmp_path):
    mock_get = mocker.patch("data_layer.fetcher.arxiv_api.requests.get")
    mocker.patch("data_layer.fetcher.arxiv_api.time.sleep")

    mock_response = mocker.Mock(status_code=404)
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_get.return_value = mock_response

    fake_entry = mocker.Mock(
        id="http://arxiv.org/abs/9999.99999v1",
        links=[mocker.Mock(title="pdf", href="http://arxiv.org/pdf/9999.99999v1")],
    )

    with pytest.raises(requests.HTTPError):
        download_paper_pdf(fake_entry, tmp_path)

    assert list(tmp_path.iterdir()) == []
