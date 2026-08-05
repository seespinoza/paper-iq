"""Extract raw paper data from the arXiv API.

This module is the "extract" stage of the data layer: it queries the
arXiv API for paper metadata and downloads the associated PDFs,
persisting both to local disk as raw artifacts. A separate loader
module is responsible for reading these artifacts and ingesting them
into a database.

Rate limiting: per arXiv's Terms of Use
(https://info.arxiv.org/help/api/tou.html), callers must make no more
than one request every three seconds and must not use concurrent
connections.

Page size: the arXiv API accepts up to 2000 results per request, but
very large pages are slower to retrieve and queries returning more
than ~1000 results should generally be narrowed. A default
``max_results`` of 100 is used here, matching the default used by the
community-maintained ``arxiv`` Python client.
"""

import gzip
import json
import time
from pathlib import Path

import feedparser
import requests

BASE_URL = "http://export.arxiv.org/api/query"
HEADERS = {"User-Agent": "arxiv-agent research project (se.espinoza132@gmail.com)"}
PAPERS_PATH = Path("papers/")
METADATA_PATH = Path("papers/metadata.jsonl.gz")


def _parse_feed(xml: str) -> tuple[list, int]:
    """Parse an arXiv Atom feed response.

    Args:
        xml: Raw XML/Atom response body returned by the arXiv API.

    Returns:
        A tuple of ``(entries, total_results)`` where ``entries`` is
        the list of ``feedparser`` entry objects on the current page
        and ``total_results`` is the total number of results matching
        the query across all pages, as reported by the feed's
        OpenSearch ``totalResults`` field.
    """
    feed = feedparser.parse(xml)
    return feed.entries, int(feed.feed.opensearch_totalresults)


def count_number_avail_papers(cat: str, start_date: str, end_date: str) -> int:
    """Count papers available for a category and date range.

    Issues a single, minimal (``max_results=1``) request to the arXiv
    API and reads the total match count off the response, without
    paging through the actual entries. Useful for sizing a download
    before calling :func:`fetch_paper_catalog`.

    Args:
        cat: arXiv category to search, e.g. ``"cs.AI"``.
        start_date: Inclusive start of the submission date range, in
            arXiv's ``YYYYMMDDHHMM`` format.
        end_date: Inclusive end of the submission date range, in
            arXiv's ``YYYYMMDDHHMM`` format.

    Returns:
        Total number of papers matching the category and date range.

    Raises:
        requests.HTTPError: If the arXiv API request fails.
    """
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
    """Estimate the total download size for a category and date range.

    Not yet implemented; intended to combine
    :func:`count_number_avail_papers` with ``avg_file_size_gb`` to
    estimate total disk usage before running a bulk download.

    Args:
        cat: arXiv category to search, e.g. ``"cs.AI"``.
        start_date: Inclusive start of the submission date range, in
            arXiv's ``YYYYMMDDHHMM`` format.
        end_date: Inclusive end of the submission date range, in
            arXiv's ``YYYYMMDDHHMM`` format.
        avg_file_size_gb: Assumed average PDF size, in gigabytes, used
            to project total download size.

    Returns:
        Estimated download size in gigabytes. Currently always
        returns ``-1`` as a placeholder.
    """
    return -1


def fetch_paper_catalog(
    cat: str, start_date: str, end_date: str, max_results: int = 100
) -> list:
    """Fetch the full catalog of paper metadata entries for a query.

    Pages through the arXiv API in ``max_results``-sized chunks,
    sorted by ascending submission date, until no more entries are
    returned. Sleeps 3 seconds between requests to comply with
    arXiv's Terms of Use rate limit.

    Args:
        cat: arXiv category to search, e.g. ``"cs.AI"``.
        start_date: Inclusive start of the submission date range, in
            arXiv's ``YYYYMMDDHHMM`` format.
        end_date: Inclusive end of the submission date range, in
            arXiv's ``YYYYMMDDHHMM`` format.
        max_results: Number of results to request per page. arXiv
            allows up to 2000, but smaller pages are faster per
            request; 100 mirrors the default used by the
            community-maintained ``arxiv`` Python client.

    Returns:
        List of ``feedparser`` entry objects, one per matching paper,
        across all pages.

    Raises:
        requests.HTTPError: If any page request fails.
    """
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


def download_paper_pdf(
    entry: feedparser.util.FeedParserDict,
    download_path: Path = PAPERS_PATH,
    metadata_path: Path = METADATA_PATH,
) -> None:
    """Download a single paper's PDF and append its metadata.

    Fetches the PDF for the given catalog entry (arXiv has no bulk
    PDF endpoint, so this is one request per paper), writes it to
    ``download_path`` named by the paper's arXiv ID, and appends the
    entry's metadata as a JSON line to the gzip-compressed
    ``metadata_path``. Sleeps 3 seconds afterward to comply with
    arXiv's Terms of Use rate limit.

    Args:
        entry: A single catalog entry as returned by
            :func:`fetch_paper_catalog`, containing the paper's
            arXiv ID, links, and metadata.
        download_path: Directory to write the downloaded PDF into.
        metadata_path: Gzip-compressed JSON Lines file to append the
            entry's metadata to.

    Raises:
        requests.HTTPError: If the PDF request fails.
        StopIteration: If ``entry`` has no link with a ``pdf`` title.
    """
    entry_id = entry.id.split("/abs/")[-1]
    pdf_url = next(
        link.href for link in entry.links if getattr(link, "title", None) == "pdf"
    )
    resp = requests.get(pdf_url, headers=HEADERS)
    resp.raise_for_status()
    with open(download_path / f"{entry_id}.pdf", "wb") as f:
        f.write(resp.content)

    with gzip.open(metadata_path, "at", encoding="utf-8") as f:
        f.write(json.dumps(dict(entry), default=str) + "\n")

    time.sleep(3)


if __name__ == "__main__":
    st = time.perf_counter()

    # If entries file already exists then do not rewrite
    entries_file = PAPERS_PATH / "entries.pkl"
    if entries_file.is_file():
        with open(entries_file, "rb") as f:
            entries = f.read()
    else:
        entries = fetch_paper_catalog("cs.AI", "202607010000", "202608010000", 200)
        with open(entries_file, "wb") as f:
            f.dump(entries, f)

    print("Time to fetch paper catalog: ", time.perf_counter() - st)
    print("Num entries: ", len(entries))

    st = time.perf_counter()

    for entry in entries:
        entry_id = entry.id.split("/abs/")[-1]

        # only download net new papers
        if not (PAPERS_PATH / f"{entry_id}.pdf").is_file():
            download_paper_pdf(entry)

    print("Time to fetch all papers", time.perf_counter() - st)
