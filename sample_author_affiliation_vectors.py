from __future__ import annotations

"""
Sample author affiliations and export transformed affiliation-type vectors.

Each output row represents one row from publication_author_locations. The vector
columns are power-normalized weights based on
publication_author_location_to_affiliation_type_distance.
"""

import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
import sqlite3

from affiliation_vector_transform import power_normalize


DB_PATH = "2025_11_09_researchgate.sqlite"
REPORTS_DIR = Path("topic_mapping_reports")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def set_sqlite_pragmas(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a sample of author affiliations with their transformed "
            "affiliation-type weight vector."
        )
    )
    parser.add_argument(
        "count",
        type=int,
        help="Number of author-affiliation rows to export.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Select rows randomly instead of taking the first rows by id.",
    )
    parser.add_argument(
        "--affiliation-filter",
        default=None,
        help=(
            "Only sample author-affiliation rows where the affiliation text "
            "contains this substring, case-insensitive."
        ),
    )
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. Defaults to "
            "topic_mapping_reports/author_affiliation_vectors_sample_<timestamp>.csv."
        ),
    )
    return parser.parse_args()


def output_path_from_args(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    REPORTS_DIR.mkdir(exist_ok=True)
    return REPORTS_DIR / f"author_affiliation_vectors_sample_{timestamp_suffix()}.csv"


def placeholders(values: list[int]) -> str:
    return ", ".join("?" for _ in values)


def load_affiliation_types(connection: sqlite3.Connection) -> list[tuple[int, str]]:
    return connection.execute(
        """
        select id, short_name
        from affiliation_types
        order by short_name
        """
    ).fetchall()


def sample_author_location_ids(
    connection: sqlite3.Connection,
    count: int,
    random_sample: bool,
    affiliation_filter: str | None,
) -> list[int]:
    order_clause = "random()" if random_sample else "id"
    where_clause = ""
    params: list[object] = []
    if affiliation_filter:
        where_clause = "where lower(affiliation_text) like ?"
        params.append(f"%{affiliation_filter.lower()}%")
    params.append(count)

    rows = connection.execute(
        f"""
        select id
        from publication_author_locations
        {where_clause}
        order by {order_clause}
        limit ?
        """,
        params,
    ).fetchall()
    return [row["id"] for row in rows]


def load_author_locations(
    connection: sqlite3.Connection,
    author_location_ids: list[int],
) -> dict[int, sqlite3.Row]:
    if not author_location_ids:
        return {}

    rows = connection.execute(
        f"""
        select
            id,
            publication_id,
            author_name,
            affiliation_text
        from publication_author_locations
        where id in ({placeholders(author_location_ids)})
        """,
        author_location_ids,
    ).fetchall()
    return {row["id"]: row for row in rows}


def load_vectors(
    connection: sqlite3.Connection,
    author_location_ids: list[int],
) -> dict[int, dict[int, float]]:
    if not author_location_ids:
        return {}

    rows = connection.execute(
        f"""
        select
            publication_author_location_id,
            affiliation_type_id,
            semantic_similarity
        from publication_author_location_to_affiliation_type_distance
        where publication_author_location_id in ({placeholders(author_location_ids)})
        """,
        author_location_ids,
    )

    vectors: dict[int, dict[int, float]] = {}
    for row in rows:
        author_location_id = row["publication_author_location_id"]
        affiliation_type_id = row["affiliation_type_id"]
        vectors.setdefault(author_location_id, {})[affiliation_type_id] = float(
            row["semantic_similarity"]
        )
    return vectors


def write_csv(
    path: Path,
    author_location_ids: list[int],
    author_locations: dict[int, sqlite3.Row],
    vectors: dict[int, dict[int, float]],
    affiliation_types: list[tuple[int, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "author_name",
        "author_affiliation",
        "publication_id",
        "publication_author_location_id",
        *[short_name for _, short_name in affiliation_types],
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for author_location_id in author_location_ids:
            author_location = author_locations[author_location_id]
            vector = vectors.get(author_location_id, {})
            source_values = [
                vector.get(affiliation_type_id, 0.0)
                for affiliation_type_id, _ in affiliation_types
            ]
            transformed_values = power_normalize(source_values).tolist()

            writer.writerow(
                [
                    author_location["author_name"],
                    author_location["affiliation_text"],
                    author_location["publication_id"],
                    author_location["id"],
                    *transformed_values,
                ]
            )


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("count must be greater than zero")

    output_path = output_path_from_args(args)
    logger.info("Database: %s", args.db)
    logger.info("Output CSV: %s", output_path)
    logger.info(
        "Selecting %s author-affiliation rows%s%s",
        args.count,
        " randomly" if args.random else "",
        f" matching {args.affiliation_filter!r}" if args.affiliation_filter else "",
    )

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        set_sqlite_pragmas(connection)
        affiliation_types = load_affiliation_types(connection)
        author_location_ids = sample_author_location_ids(
            connection,
            args.count,
            args.random,
            args.affiliation_filter,
        )
        author_locations = load_author_locations(connection, author_location_ids)
        vectors = load_vectors(connection, author_location_ids)

    write_csv(
        output_path,
        author_location_ids,
        author_locations,
        vectors,
        affiliation_types,
    )
    print(output_path)


if __name__ == "__main__":
    main()
