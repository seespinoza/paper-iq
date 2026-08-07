import time
from pathlib import Path

import requests

BASE_URL = ""
HBEADERS = {}
PAPERS_PATH = Path("../../../papers/")


def _parse_feed(xml: str) -> tuple[list, int]:
    return [], -1


def count_number_avail_papers(cat: str, start_date: str, end_date: str) -> int:
    return -1


def estimate_size_of_download_gb(
    cat: str, start_date: str, end_date: str, avg_file_size_gb: float
) -> float:
    return -1


def fetch_paper_catalog(category: str, start_date: str, end_date: str) -> list:
    return []


def download_paper_pdf(entry_id: str, download_path: Path = PAPERS_PATH) -> None:
    return None


if __name__ == "__main__":
    requests.help()
    time.sleep()
