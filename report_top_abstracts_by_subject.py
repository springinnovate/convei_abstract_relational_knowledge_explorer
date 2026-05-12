from __future__ import annotations

"""
Export top publication abstracts for each subject.

The database stores raw semantic similarities between publication abstracts and
base topics. This report keeps those values raw. For each subject, it ranks
publication abstracts by that subject's semantic similarity and exports the top
N rows, along with the full raw subject vector for each selected abstract.
"""

import argparse
import csv
from datetime import datetime
import heapq
import logging
from pathlib import Path
import sqlite3

from tqdm.auto import tqdm


DB_PATH = "2025_11_09_researchgate.sqlite"
REPORTS_DIR = Path("topic_mapping_reports")
SQLITE_VARIABLE_BATCH_SIZE = 900

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export top publication abstracts for each subject, ranked by raw "
            "semantic similarity."
        )
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        required=True,
        help="Number of abstracts to export per subject.",
    )
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. Defaults to "
            "topic_mapping_reports/top_abstracts_by_subject_<timestamp>.csv."
        ),
    )
    return parser.parse_args()


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def output_path_from_args(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    REPORTS_DIR.mkdir(exist_ok=True)
    return REPORTS_DIR / f"top_abstracts_by_subject_{timestamp_suffix()}.csv"


def set_sqlite_pragmas(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def batched(values: list[int], batch_size: int) -> list[list[int]]:
    return [
        values[index : index + batch_size]
        for index in range(0, len(values), batch_size)
    ]


def load_subjects(connection: sqlite3.Connection) -> list[tuple[int, str]]:
    return connection.execute(
        """
        select id, short_name
        from base_topics
        order by short_name
        """
    ).fetchall()


def load_eligible_publication_ids(connection: sqlite3.Connection) -> set[int]:
    rows = connection.execute(
        """
        select id
        from publications
        where abstract is not null
            and trim(abstract) != ''
        """
    )
    return {int(publication_id) for publication_id, in rows}


def load_ranked_abstracts(
    connection: sqlite3.Connection,
    subjects: list[tuple[int, str]],
    count: int,
) -> list[dict[str, int]]:
    subject_ids = [subject_id for subject_id, _ in subjects]
    heaps: dict[int, list[tuple[float, int, int]]] = {
        subject_id: [] for subject_id in subject_ids
    }
    eligible_publication_ids = load_eligible_publication_ids(connection)

    rows = connection.execute(
        """
        select
            publication_id,
            base_topic_id,
            semantic_similarity
        from base_topic_to_pub_distance
        """
    )

    for publication_id, subject_id, semantic_similarity in tqdm(
        rows,
        desc="Scanning subject rows",
        unit="row",
    ):
        publication_id = int(publication_id)
        if publication_id not in eligible_publication_ids:
            continue

        heap = heaps.get(int(subject_id))
        if heap is None:
            continue

        score = float(semantic_similarity)
        heap_key = (score, -publication_id, publication_id)
        if len(heap) < count:
            heapq.heappush(heap, heap_key)
        elif heap_key > heap[0]:
            heapq.heapreplace(heap, heap_key)

    ranked_rows: list[dict[str, int]] = []
    for subject_id in subject_ids:
        top_rows = sorted(
            heaps[subject_id],
            key=lambda item: (-item[0], item[2]),
        )
        for rank, (_, _, publication_id) in enumerate(top_rows, start=1):
            ranked_rows.append(
                {
                    "publication_id": publication_id,
                    "ranked_subject_id": subject_id,
                    "rank": rank,
                }
            )

    return ranked_rows


def load_abstracts(
    connection: sqlite3.Connection,
    publication_ids: list[int],
) -> dict[int, str]:
    if not publication_ids:
        return {}

    abstracts: dict[int, str] = {}
    for publication_id_batch in batched(
        publication_ids,
        SQLITE_VARIABLE_BATCH_SIZE,
    ):
        placeholders = ", ".join("?" for _ in publication_id_batch)
        rows = connection.execute(
            f"""
            select id, abstract
            from publications
            where id in ({placeholders})
            """,
            publication_id_batch,
        ).fetchall()

        for publication_id, abstract in rows:
            abstracts[int(publication_id)] = abstract or ""

    return abstracts


def load_vectors(
    connection: sqlite3.Connection,
    publication_ids: list[int],
) -> dict[int, dict[int, float]]:
    if not publication_ids:
        return {}

    vectors: dict[int, dict[int, float]] = {}
    for publication_id_batch in batched(
        publication_ids,
        SQLITE_VARIABLE_BATCH_SIZE,
    ):
        placeholders = ", ".join("?" for _ in publication_id_batch)
        rows = connection.execute(
            f"""
            select
                publication_id,
                base_topic_id,
                semantic_similarity
            from base_topic_to_pub_distance
            where publication_id in ({placeholders})
            """,
            publication_id_batch,
        ).fetchall()

        for row in rows:
            vectors.setdefault(row["publication_id"], {})[
                row["base_topic_id"]
            ] = float(row["semantic_similarity"])

    return vectors


def top_subject(
    vector: dict[int, float],
    subject_name_by_id: dict[int, str],
) -> str:
    if not vector:
        return ""
    top_subject_id, _ = max(
        vector.items(),
        key=lambda item: (item[1], -item[0]),
    )
    return subject_name_by_id[top_subject_id]


def write_csv(
    path: Path,
    ranked_rows: list[dict[str, int]],
    abstracts: dict[int, str],
    vectors: dict[int, dict[int, float]],
    subjects: list[tuple[int, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subject_name_by_id = dict(subjects)
    subject_order_by_id = {
        subject_id: index for index, (subject_id, _) in enumerate(subjects)
    }

    headers = [
        "abstract",
        "rank",
        "top_subject",
        *[short_name for _, short_name in subjects],
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for row in sorted(
            ranked_rows,
            key=lambda item: (
                subject_order_by_id[item["ranked_subject_id"]],
                item["rank"],
            ),
        ):
            publication_id = row["publication_id"]
            vector = vectors.get(publication_id, {})
            writer.writerow(
                [
                    abstracts.get(publication_id, ""),
                    row["rank"],
                    top_subject(vector, subject_name_by_id),
                    *[
                        vector.get(subject_id, 0.0)
                        for subject_id, _ in subjects
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
    logger.info("Rows per subject: %s", args.count)

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        set_sqlite_pragmas(connection)

        subjects = load_subjects(connection)
        ranked_rows = load_ranked_abstracts(connection, subjects, args.count)
        publication_ids = sorted({row["publication_id"] for row in ranked_rows})
        abstracts = load_abstracts(connection, publication_ids)
        vectors = load_vectors(connection, publication_ids)

    write_csv(output_path, ranked_rows, abstracts, vectors, subjects)
    print(output_path)


if __name__ == "__main__":
    main()
