"""
scrape_olympedia_results.py

Scrape every athlete's name, sport, event, finishing position, and medal (if any)
from an Olympedia country/edition roster page.

Usage:
    python scrape_olympedia_results.py
    # or pass any roster URL:
    python scrape_olympedia_results.py https://www.olympedia.org/countries/USA/editions/72
"""

import csv
import json
import sys
import time

import requests
from bs4 import BeautifulSoup

DEFAULT_ROSTER = "https://www.olympedia.org/countries/USA/editions/62"
HEADERS = {"User-Agent": "Mozilla/5.0 (results-scraper; personal use)"}


def scrape_results(roster_url: str, jsonl_path: str, row_delay_s: float = 0.5) -> list[dict]:
    resp = requests.get(roster_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", class_="table")
    if table is None:
        raise RuntimeError("Could not find results table on page.")

    rows: list[dict] = []
    current_sport = None
    current_event = None

    # Stream each parsed row to JSONL so progress is visible while the script runs.
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")

            # Sport header row: <td colspan="4"><h2>Alpine Skiing</h2></td>
            if len(cells) == 1 and cells[0].find("h2"):
                current_sport = cells[0].get_text(strip=True)
                current_event = None
                continue

            # Athlete result row: 4 columns -> event | athlete | position | medal
            if len(cells) != 4:
                continue

            event_cell, athlete_cell, pos_cell, medal_cell = cells

            event_text = event_cell.get_text(strip=True)
            if event_text:
                current_event = event_text  # new event group starts

            athlete_name = athlete_cell.get_text(strip=True)
            athlete_link = athlete_cell.find("a")
            athlete_id = (
                athlete_link["href"].rsplit("/", 1)[-1]
                if athlete_link and athlete_link.get("href")
                else None
            )

            position = pos_cell.get_text(strip=True) or None

            # Medal cell contains <span class="Gold|Silver|Bronze">…</span> when applicable
            medal = None
            medal_span = medal_cell.find("span")
            if medal_span:
                medal = medal_span.get_text(strip=True)

            # Skip pure team-aggregate rows (no individual athlete linked).
            # Example: "United States" team rows for relays/bobsleigh listing the team itself.
            # These rows still describe an event result; keep them but mark accordingly.
            is_team_row = athlete_link is None or "/athletes/" not in athlete_link.get("href", "")

            row = {
                "sport": current_sport,
                "event": current_event,
                "athlete_id": athlete_id if not is_team_row else "",
                "athlete": athlete_name,
                "position": position,
                "medal": medal or "",
                "is_team_entry": "yes" if is_team_row else "no",
            }
            rows.append(row)

            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            jf.flush()
            print(
                f"[{len(rows):>4}] {row['sport'] or '':>18} | {row['event'] or '':<28} "
                f"| {row['athlete']:<24} | pos={row['position'] or '':<5} | medal={row['medal']}",
                flush=True,
            )
            time.sleep(row_delay_s)

    return rows


def main() -> None:
    roster_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROSTER
    print(f"Scraping {roster_url}")

    jsonl_path = "olympedia_results.jsonl"
    csv_path = "olympedia_results.csv"

    rows = scrape_results(roster_url, jsonl_path=jsonl_path, row_delay_s=0.2)

    fieldnames = [
        "sport",
        "event",
        "athlete_id",
        "athlete",
        "position",
        "medal",
        "is_team_entry",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    medal_count = sum(1 for r in rows if r["medal"])
    print(f"\nWrote {len(rows)} result rows to {csv_path} and {jsonl_path}")
    print(f"  - {medal_count} medal-winning entries")


if __name__ == "__main__":
    main()