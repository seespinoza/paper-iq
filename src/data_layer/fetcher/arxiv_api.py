import time
from pathlib import Path

import feedparser
import requests

BASE_URL = "http://export.arxiv.org/api/query"
HEADERS = {"User-Agent": "arxiv-agent research project (se.espinoza132@gmail.com)"}
PAPERS_PATH = Path("../../../papers/")


def _parse_feed(xml: str) -> tuple[list, int]:
    feed = feedparser.parse(xml)
    return feed.entries, int(feed.feed.opensearch_totalresults)


def count_number_avail_papers(cat: str, start_date: str, end_date: str) -> int:
    params = {
        "search_query": f"cat:{cat} AND submittedDate:[{start_date} TO {end_date}]",
        "start": 0,
        "max_results": 1,
    }
    resp = requests.get(BASE_URL, params=params, headers=HEADERS)
    resp.raise_for_status()
    _, total = _parse_feed(resp.text)
    return total


def estimate_size_of_download_gb(
    cat: str, start_date: str, end_date: str, avg_file_size_gb: float
) -> float:
    return -1


def fetch_paper_catalog(
    cat: str, start_date: str, end_date: str, max_results: int = 100
) -> list:
    all_entries = []
    start = 0
    while True:
        params = {
            "search_query": f"cat:{cat} AND submittedDate:[{start_date} TO {end_date}]",
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
            "start": start,
            "max_results": max_results,
        }
        resp = requests.get(BASE_URL, params=params, headers=HEADERS)
        resp.raise_for_status()
        entries, _ = _parse_feed(resp.text)
        if not entries:
            break
        all_entries.extend(entries)
        start += max_results
        time.sleep(3)
    return all_entries


def download_paper_pdf(entry: str, download_path: Path = PAPERS_PATH) -> None:
    entry_id = entry.id.split("/abs/")[-1]
    pdf_url = next(
        link.href for link in entry.links if getattr(link, "title", None) == "pdf"
    )
    resp = requests.get(pdf_url, headers=HEADERS)
    resp.raise_for_status()
    with open(download_path / f"{entry_id}.pdf", "wb") as f:
        f.write(resp.content)
    time.sleep(3)


if __name__ == "__main__":
    pass
