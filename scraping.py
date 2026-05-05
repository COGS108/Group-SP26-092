"""
scrape_olympedia_birthdays.py

Scrape birthdays for every athlete listed on an Olympedia country/edition page
(e.g. United States at the 2026 Winter Olympics).

Usage:
    python scrape_olympedia_birthdays.py
    # or pass a different roster URL:
    python scrape_olympedia_birthdays.py https://www.olympedia.org/countries/USA/editions/72
"""

import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

BASE = "https://www.olympedia.org"
DEFAULT_ROSTER = f"{BASE}/countries/USA/editions/62"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (birthday-scraper; personal use)"
}

# Match a numeric athlete id like /athletes/128669
ATHLETE_HREF_RE = re.compile(r"^/athletes/(\d+)$")


def get_athlete_links(roster_url: str) -> list[tuple[str, str]]:
    """Return a list of (athlete_id, name) tuples, deduplicated, in page order."""
    resp = requests.get(roster_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    seen: dict[str, str] = {}
    for a in soup.select('a[href^="/athletes/"]'):
        m = ATHLETE_HREF_RE.match(a.get("href", ""))
        if not m:
            continue
        athlete_id = m.group(1)
        if athlete_id not in seen:
            seen[athlete_id] = a.get_text(strip=True)
    return list(seen.items())


def parse_born_field(html: str) -> tuple[str | None, str | None]:
    """Extract (birth_date, birthplace) from an athlete page's HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Olympedia bio table: <tr><th>Born</th><td>3 October 1998 in City, State (USA)</td></tr>
    for th in soup.find_all("th"):
        if th.get_text(strip=True).lower() == "born":
            td = th.find_next_sibling("td")
            if not td:
                return None, None
            text = td.get_text(" ", strip=True)
            # Split "3 October 1998 in Burlington, Vermont (USA)"
            if " in " in text:
                date_part, place_part = text.split(" in ", 1)
                return date_part.strip(), place_part.strip()
            return text.strip(), None
    return None, None


def fetch_athlete(athlete_id: str, name: str, retries: int = 4) -> dict:
    url = f"{BASE}/athletes/{athlete_id}"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 10 * (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            born, place = parse_born_field(resp.text)
            return {
                "id": athlete_id,
                "name": name,
                "born": born,
                "birthplace": place,
                "url": url,
            }
        except requests.RequestException as e:
            if attempt == retries:
                return {
                    "id": athlete_id,
                    "name": name,
                    "born": None,
                    "birthplace": None,
                    "url": url,
                    "error": str(e),
                }
            time.sleep(2.0 * (attempt + 1))
    return {
        "id": athlete_id,
        "name": name,
        "born": None,
        "birthplace": None,
        "url": url,
        "error": "exhausted retries (likely 429)",
    }


def main() -> None:
    roster_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROSTER
    print(f"Fetching roster: {roster_url}")
    athletes = get_athlete_links(roster_url)
    print(f"Found {len(athletes)} unique athletes.")

    jsonl_path = "olympedia_birthdays.jsonl"

    # Resume: read any existing JSONL and skip athletes who already have a birthday.
    # Athletes with a previous error/null are retried.
    existing_rows: list[dict] = []
    already_done: set[str] = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_rows.append(rec)
                if rec.get("born"):
                    already_done.add(rec["id"])
        print(f"Resuming: {len(already_done)} athletes already have birthdays in {jsonl_path}")

    todo = [(aid, name) for aid, name in athletes if aid not in already_done]
    print(f"Remaining to fetch: {len(todo)}")

    max_workers = 3

    new_rows: list[dict] = []
    # Append mode preserves prior progress; concurrency 3 balances speed vs. 429 risk.
    with open(jsonl_path, "a", encoding="utf-8") as jf, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_athlete, aid, name): (aid, name) for aid, name in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            new_rows.append(row)
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            print(f"[{i:>3}/{len(todo)}] {row['name']:30s} -> {row['born']}", flush=True)

    # Combine and dedupe (latest record per athlete wins)
    by_id: dict[str, dict] = {r["id"]: r for r in existing_rows}
    for r in new_rows:
        by_id[r["id"]] = r

    # Preserve original page order for the CSV
    order = {aid: i for i, (aid, _) in enumerate(athletes)}
    results = sorted(by_id.values(), key=lambda r: order.get(r["id"], 10**9))

    csv_path = "olympedia_birthdays.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "name", "born", "birthplace", "url"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    missing = [r["name"] for r in results if not r["born"]]
    print(f"\nWrote {len(results)} rows to {csv_path} and {jsonl_path}")
    print(f"Missing birthdays: {len(missing)}")
    if missing:
        print("  " + ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""))


if __name__ == "__main__":
    main()