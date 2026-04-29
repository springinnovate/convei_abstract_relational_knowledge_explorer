from __future__ import annotations

"""
Populate cleaned affiliation text for author affiliation rows.

The cleaner is intentionally conservative: it strips trailing comma-separated
place/address suffixes, but stops when it reaches institution-like text.
"""

import argparse
import csv
import logging
import re
from pathlib import Path
import sqlite3
import sys


DB_PATH = "2025_11_09_researchgate.sqlite"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


INSTITUTION_RE = re.compile(
    r"\b("
    r"univ|university|inst|institute|acad|academy|dept|department|"
    r"sch|school|fac|faculty|lab|laborator|ctr|center|centre|"
    r"hosp|hospital|coll|college|observ|observat|minist|ministry|"
    r"agency|admin|adm|corp|corporation|inc|ltd|co\.?\s*ltd|"
    r"gmbh|llc|company|technol|technology|sci|science|sciences|"
    r"research|res|foundation|soc|society|museum"
    r")\b",
    re.IGNORECASE,
)

ADDRESS_RE = re.compile(
    r"\b("
    r"road|rd|street|st|avenue|ave|drive|dr|lane|ln|blvd|"
    r"allee|via|jl|dong|ku|locked bag|po box|p\.o\. box|"
    r"campus|bldg|building|floor|room|suite"
    r")\b",
    re.IGNORECASE,
)

POSTAL_RE = re.compile(
    r"("
    r"\b[A-Z]{1,3}-?\d{4,8}\b|"  # D-53115, F-45071, BR-24220900
    r"\b\d{4,6}\b|"  # 94305, 430079, 55281
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b|"  # UK postcode-ish
    r"\b[A-Z]{2}\s*\d{4,6}\b|"  # CA 94305, PA 19486
    r"\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b"  # Canadian postal code
    r")",
    re.IGNORECASE,
)

STATE_RE = re.compile(
    r"^\s*("
    r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|"
    r"MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|"
    r"OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY|"
    r"AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT|"
    r"NSW|QLD|VIC|TAS|ACT"
    r")\s*(\d{3,6})?\s*$",
    re.IGNORECASE,
)

COUNTRY_ALIASES = {
    "usa",
    "u.s.a.",
    "u s a",
    "united states",
    "united states of america",
    "uk",
    "u.k.",
    "u k",
    "united kingdom",
    "england",
    "scotland",
    "wales",
    "northern ireland",
    "peoples r china",
    "people's r china",
    "people r china",
    "p r china",
    "pr china",
    "china",
    "south korea",
    "republic of korea",
    "korea",
    "russia",
    "iran",
    "taiwan",
}

COUNTRY_RE: re.Pattern[str] | None = None


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = normalized.strip(".;")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def build_country_pattern() -> re.Pattern[str]:
    try:
        import pycountry
    except ImportError as exc:
        raise SystemExit(
            "pycountry is required for this script. Install it with: pip install pycountry"
        ) from exc

    country_names = set(COUNTRY_ALIASES)
    for country in pycountry.countries:
        country_names.add(country.name.lower())
        if hasattr(country, "official_name"):
            country_names.add(country.official_name.lower())
        if hasattr(country, "common_name"):
            country_names.add(country.common_name.lower())

    escaped_names = [
        re.escape(country_name)
        for country_name in sorted(country_names, key=len, reverse=True)
        if len(country_name) >= 3
    ]
    return re.compile(
        r"(?<![a-z])(" + "|".join(escaped_names) + r")(?![a-z])",
        re.IGNORECASE,
    )


def country_pattern() -> re.Pattern[str]:
    global COUNTRY_RE
    if COUNTRY_RE is None:
        COUNTRY_RE = build_country_pattern()
    return COUNTRY_RE


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def looks_like_institution(text: str) -> bool:
    return bool(INSTITUTION_RE.search(text))


def looks_like_country(text: str) -> bool:
    return bool(country_pattern().search(normalize_text(text)))


def looks_like_location_anchor(text: str) -> bool:
    return bool(
        looks_like_country(text)
        or POSTAL_RE.search(text)
        or STATE_RE.search(text)
        or ADDRESS_RE.search(text)
    )


def split_affiliation_and_place(
    affiliation_text: str,
) -> tuple[str, str | None]:
    if not affiliation_text or not affiliation_text.strip():
        return "", None

    chunks = [
        chunk.strip() for chunk in affiliation_text.split(",") if chunk.strip()
    ]
    if len(chunks) <= 1:
        return affiliation_text.strip(), None

    place_chunks: list[str] = []
    found_anchor = False

    while chunks:
        chunk = chunks[-1]

        if looks_like_institution(chunk):
            break

        if looks_like_location_anchor(chunk):
            place_chunks.insert(0, chunks.pop())
            found_anchor = True
            continue

        # After a country/postcode/address anchor, adjacent short suffix chunks
        # are usually city/province/county names even when globally obscure.
        if found_anchor and token_count(chunk) <= 4:
            place_chunks.insert(0, chunks.pop())
            continue

        break

    cleaned_affiliation = ", ".join(chunks).strip()
    place = ", ".join(place_chunks).strip() or None

    if not cleaned_affiliation:
        return affiliation_text.strip(), place

    return cleaned_affiliation, place


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate cleaned affiliation text for publication_author_locations."
    )
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print raw/cleaned/place rows without modifying the database.",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select rows when --limit is provided.",
    )
    parser.add_argument(
        "--dry-run-output",
        type=Path,
        default=None,
        help="Optional CSV path for dry-run output instead of stdout.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero")
    if args.dry_run and args.limit is None:
        parser.error(
            "--dry-run requires --limit to avoid dumping the whole table"
        )
    if args.random and args.limit is None:
        parser.error("--random requires --limit")

    return args


def set_sqlite_pragmas(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def ensure_cleaned_affiliation_column(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(publication_author_locations)"
        )
    }
    if "cleaned_affiliation_text" in columns:
        return

    logger.info("Adding cleaned_affiliation_text column")
    connection.execute(
        "ALTER TABLE publication_author_locations "
        "ADD COLUMN cleaned_affiliation_text TEXT"
    )
    connection.commit()


def fetch_author_affiliation_rows(
    connection: sqlite3.Connection,
    after_id: int,
    batch_size: int,
    remaining: int | None,
) -> list[sqlite3.Row]:
    limit = batch_size if remaining is None else min(batch_size, remaining)
    return connection.execute(
        """
        select id, affiliation_text
        from publication_author_locations
        where id > ?
        order by id
        limit ?
        """,
        (after_id, limit),
    ).fetchall()


def fetch_random_author_affiliation_rows(
    connection: sqlite3.Connection,
    limit: int,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        select id, affiliation_text
        from publication_author_locations
        order by random()
        limit ?
        """,
        (limit,),
    ).fetchall()


def dry_run(args: argparse.Namespace) -> None:
    output_file = None
    if args.dry_run_output is not None:
        args.dry_run_output.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(
            args.dry_run_output, "w", newline="", encoding="utf-8"
        )

    output = output_file or sys.stdout
    writer = csv.writer(output)
    writer.writerow(
        [
            "publication_author_location_id",
            "affiliation_text",
            "cleaned_affiliation_text",
            "stripped_place_text",
        ]
    )

    try:
        with sqlite3.connect(args.db) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=60000")
            if args.random:
                rows = fetch_random_author_affiliation_rows(connection, args.limit)
            else:
                rows = fetch_author_affiliation_rows(
                    connection,
                    after_id=0,
                    batch_size=args.limit,
                    remaining=args.limit,
                )
            for row in rows:
                cleaned_affiliation, place = split_affiliation_and_place(
                    row["affiliation_text"]
                )
                writer.writerow(
                    [
                        row["id"],
                        row["affiliation_text"],
                        cleaned_affiliation,
                        place or "",
                    ]
                )
    finally:
        if output_file is not None:
            output_file.close()
            logger.info("Dry-run output written to %s", args.dry_run_output)


def update_database(args: argparse.Namespace) -> None:
    processed_count = 0
    changed_count = 0
    last_id = 0
    remaining = args.limit

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        set_sqlite_pragmas(connection)
        ensure_cleaned_affiliation_column(connection)

        if args.random:
            rows = fetch_random_author_affiliation_rows(connection, args.limit)
            changed_count = update_rows(connection, rows)
            logger.info(
                "Finished random cleaned affiliation update: processed=%s changed=%s",
                len(rows),
                changed_count,
            )
            return

        while remaining is None or remaining > 0:
            rows = fetch_author_affiliation_rows(
                connection,
                after_id=last_id,
                batch_size=args.batch_size,
                remaining=remaining,
            )
            if not rows:
                break

            changed_count += update_rows(connection, rows)

            processed_count += len(rows)
            last_id = rows[-1]["id"]
            if remaining is not None:
                remaining -= len(rows)
            logger.info(
                "Processed %s rows; changed=%s; last_id=%s",
                processed_count,
                changed_count,
                last_id,
            )

    logger.info(
        "Finished cleaned affiliation update: processed=%s changed=%s",
        processed_count,
        changed_count,
    )


def update_rows(
    connection: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> int:
    changed_count = 0
    cleaned_rows = []
    for row in rows:
        cleaned_affiliation, _ = split_affiliation_and_place(
            row["affiliation_text"]
        )
        cleaned_rows.append((cleaned_affiliation, row["id"]))
        if cleaned_affiliation != row["affiliation_text"]:
            changed_count += 1

    connection.executemany(
        """
        update publication_author_locations
        set cleaned_affiliation_text = ?
        where id = ?
        """,
        cleaned_rows,
    )
    connection.commit()
    return changed_count


def main() -> None:
    args = parse_args()
    logger.info("Database: %s", args.db)
    country_pattern()
    logger.info("Country matcher is ready")

    if args.dry_run:
        dry_run(args)
        return

    update_database(args)


if __name__ == "__main__":
    main()
