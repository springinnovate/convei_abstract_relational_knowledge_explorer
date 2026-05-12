from __future__ import annotations

"""
Export top affiliation strings for each affiliation type.

The database stores raw semantic similarities between author affiliation strings
and affiliation types. This report keeps those values raw. For each affiliation
type, it ranks unique affiliation strings by that type's semantic similarity and
exports the top N rows, along with the full raw affiliation-type vector.
"""

import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
import sqlite3


DB_PATH = "2025_11_09_researchgate.sqlite"
REPORTS_DIR = Path("topic_mapping_reports")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export top unique author affiliation strings for each affiliation "
            "type, ranked by raw semantic similarity."
        )
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        required=True,
        help="Number of affiliation strings to export per affiliation type.",
    )
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. Defaults to "
            "topic_mapping_reports/top_affiliations_by_affiliation_type_<timestamp>.csv."
        ),
    )
    return parser.parse_args()


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def output_path_from_args(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    REPORTS_DIR.mkdir(exist_ok=True)
    return REPORTS_DIR / f"top_affiliations_by_affiliation_type_{timestamp_suffix()}.csv"


def set_sqlite_pragmas(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def load_affiliation_types(connection: sqlite3.Connection) -> list[tuple[int, str]]:
    return connection.execute(
        """
        select id, short_name
        from affiliation_types
        order by short_name
        """
    ).fetchall()


def load_ranked_affiliations(
    connection: sqlite3.Connection,
    count: int,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        with affiliation_vectors as (
            select
                pal.affiliation_text,
                d.affiliation_type_id,
                max(d.semantic_similarity) as semantic_similarity
            from publication_author_location_to_affiliation_type_distance d
            join publication_author_locations pal
                on pal.id = d.publication_author_location_id
            group by pal.affiliation_text, d.affiliation_type_id
        ),
        ranked as (
            select
                affiliation_text,
                affiliation_type_id,
                semantic_similarity,
                row_number() over (
                    partition by affiliation_type_id
                    order by semantic_similarity desc, affiliation_text
                ) as rank
            from affiliation_vectors
        )
        select
            affiliation_text,
            affiliation_type_id as ranked_affiliation_type_id,
            rank
        from ranked
        where rank <= ?
        order by ranked_affiliation_type_id, rank
        """,
        (count,),
    ).fetchall()


def load_vectors(
    connection: sqlite3.Connection,
    affiliation_strings: list[str],
) -> dict[str, dict[int, float]]:
    if not affiliation_strings:
        return {}

    placeholders = ", ".join("?" for _ in affiliation_strings)
    rows = connection.execute(
        f"""
        select
            pal.affiliation_text,
            d.affiliation_type_id,
            max(d.semantic_similarity) as semantic_similarity
        from publication_author_location_to_affiliation_type_distance d
        join publication_author_locations pal
            on pal.id = d.publication_author_location_id
        where pal.affiliation_text in ({placeholders})
        group by pal.affiliation_text, d.affiliation_type_id
        """,
        affiliation_strings,
    ).fetchall()

    vectors: dict[str, dict[int, float]] = {}
    for row in rows:
        vectors.setdefault(row["affiliation_text"], {})[
            row["affiliation_type_id"]
        ] = float(row["semantic_similarity"])
    return vectors


def top_affiliation_type(
    vector: dict[int, float],
    affiliation_type_name_by_id: dict[int, str],
) -> str:
    if not vector:
        return ""
    top_affiliation_type_id, _ = max(
        vector.items(),
        key=lambda item: (item[1], -item[0]),
    )
    return affiliation_type_name_by_id[top_affiliation_type_id]


def write_csv(
    path: Path,
    ranked_rows: list[sqlite3.Row],
    vectors: dict[str, dict[int, float]],
    affiliation_types: list[tuple[int, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    affiliation_type_name_by_id = dict(affiliation_types)
    affiliation_type_order_by_id = {
        affiliation_type_id: index
        for index, (affiliation_type_id, _) in enumerate(affiliation_types)
    }

    headers = [
        "affiliation_string",
        "rank",
        "top_affiliation_type",
        *[short_name for _, short_name in affiliation_types],
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for row in sorted(
            ranked_rows,
            key=lambda item: (
                affiliation_type_order_by_id[item["ranked_affiliation_type_id"]],
                item["rank"],
            ),
        ):
            affiliation_string = row["affiliation_text"]
            vector = vectors.get(affiliation_string, {})
            writer.writerow(
                [
                    affiliation_string,
                    row["rank"],
                    top_affiliation_type(vector, affiliation_type_name_by_id),
                    *[
                        vector.get(affiliation_type_id, 0.0)
                        for affiliation_type_id, _ in affiliation_types
                    ],
                ]
            )


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be greater than zero")

    output_path = output_path_from_args(args)
    logger.info("Database: %s", args.db)
    logger.info("Output CSV: %s", output_path)
    logger.info("Rows per affiliation type: %s", args.count)

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        set_sqlite_pragmas(connection)

        affiliation_types = load_affiliation_types(connection)
        ranked_rows = load_ranked_affiliations(connection, args.count)
        affiliation_strings = sorted(
            {row["affiliation_text"] for row in ranked_rows}
        )
        vectors = load_vectors(connection, affiliation_strings)

    write_csv(output_path, ranked_rows, vectors, affiliation_types)
    print(output_path)


if __name__ == "__main__":
    main()
