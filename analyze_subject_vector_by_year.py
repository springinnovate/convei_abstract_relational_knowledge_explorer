from __future__ import annotations

"""
Compute per-year base-topic distributions for publications stored in a SQLite database.

This script queries a ResearchGate-derived SQLite database containing publications, base topics,
and per-publication distances (semantic similarities) to base topics. For each year, it sums
the semantic similarity scores for each base topic across all publications in that year to form
a year-level topic vector. It also writes a normalized companion matrix where each year column
sums to 1.0.

The script can compute vectors across a year range and write a CSV matrix with one row per base
topic and one column per year.

Typical usage:
    python script.py --db path/to/db.sqlite --start-year 2015 --end-year 2020

Notes:
    - Uses SQLite WAL mode and a busy timeout for better concurrency behavior.
    - Assumes the schema defined by models.BaseTopics, models.Publication, and
      models.BaseTopicToPublicationDistance.
"""

import argparse
import csv
from datetime import datetime

import numpy as np
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from models import BaseTopics, BaseTopicToPublicationDistance, Publication

DB_PATH = "2025_11_09_researchgate.sqlite"


engine = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    connect_args={"timeout": 60},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _):
    """Configure SQLite connection pragmas for performance and reliability.

    Sets WAL journal mode, normal synchronous mode, and a busy timeout to reduce lock errors.

    Args:
        dbapi_connection: A DB-API connection object provided by SQLAlchemy.
        _: Unused connection record parameter provided by SQLAlchemy event hook.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def _load_base_topics(session: Session):
    """Load base topics and build lookup structures.

    Queries all base topics ordered by ID and returns:
      - a dense NumPy array of base topic IDs
      - a mapping from base topic ID to short text name
      - a mapping from base topic ID to its index in the dense array

    Args:
        session: An active SQLAlchemy session.

    Returns:
        tuple:
            base_topic_ids: NumPy array of base topic IDs (int64), ordered by ID.
            base_topic_text_by_id: Dict mapping base topic ID (int) -> short name (str).
            base_topic_index_by_id: Dict mapping base topic ID (int) -> index (int).
    """
    rows = session.execute(
        select(BaseTopics.id, BaseTopics.short_name).order_by(BaseTopics.id)
    ).all()
    base_topic_ids = np.array([row[0] for row in rows], dtype=np.int64)
    base_topic_text_by_id = {base_topic_id: text for base_topic_id, text in rows}
    base_topic_index_by_id = {
        base_topic_id: i for i, base_topic_id in enumerate(base_topic_ids)
    }
    return base_topic_ids, base_topic_text_by_id, base_topic_index_by_id


def year_vector(
    session: Session,
    year: int,
    base_topic_ids_all,
    base_topic_index_by_id: dict[int, int],
):
    """Compute a year-level base-topic vector by summing topic scores.

    For each publication in the specified year, this function adds each base topic's
    semantic similarity score to the corresponding year-level topic total.

    Args:
        session: An active SQLAlchemy session.
        year: Publication year to aggregate.
        base_topic_ids_all: NumPy array of all base topic IDs (int64), ordered by ID.
        base_topic_index_by_id: Dict mapping base topic ID (int) -> index (int).

    Returns:
        NumPy array (float64) with summed semantic similarity per base topic.
    """
    print(f"[year_vector] start year={year}")

    year_topic_vector = np.zeros(len(base_topic_ids_all), dtype=np.float64)

    query = (
        select(
            BaseTopicToPublicationDistance.base_topic_id,
            func.sum(BaseTopicToPublicationDistance.semantic_similarity),
        )
        .join(
            Publication,
            Publication.id == BaseTopicToPublicationDistance.publication_id,
        )
        .where(Publication.publication_year == year)
        .group_by(BaseTopicToPublicationDistance.base_topic_id)
        .order_by(BaseTopicToPublicationDistance.base_topic_id)
    )

    print("[year_vector] executing main query")

    row_counter = 0
    for base_topic_id, semantic_similarity_sum in session.execute(query):
        row_counter += 1
        year_topic_vector[base_topic_index_by_id[base_topic_id]] = float(
            semantic_similarity_sum or 0.0
        )

    print(f"[year_vector] done year={year} total_topics={row_counter}")

    return year_topic_vector


def topk(base_topic_ids, base_topic_text_by_id, year_topic_vector, k: int):
    """Return the top-k base topics by value in a year topic vector.

    Args:
        base_topic_ids: NumPy array of base topic IDs aligned with `year_topic_vector`.
        base_topic_text_by_id: Dict mapping base topic ID (int) -> short name (str).
        year_topic_vector: NumPy array of topic weights (float64).
        k: Number of top topics to return.

    Returns:
        list[tuple[int, str, float]]: Tuples of (base_topic_id, base_topic_text, value),
        sorted by descending value.
    """
    sorted_indices = np.argsort(year_topic_vector)[::-1]
    topk_rows = []
    for index in sorted_indices[:k]:
        base_topic_id = int(base_topic_ids[index])
        topk_rows.append(
            (
                base_topic_id,
                base_topic_text_by_id[base_topic_id],
                float(year_topic_vector[index]),
            )
        )
    return topk_rows


def write_csv(path: str, base_topic_ids, base_topic_text_by_id, year_topic_vector):
    """Write a year topic vector to a CSV file.

    The CSV contains one row per base topic with the summed weight for the year.

    Args:
        path: Output CSV file path.
        base_topic_ids: NumPy array of base topic IDs aligned with `year_topic_vector`.
        base_topic_text_by_id: Dict mapping base topic ID (int) -> short name (str).
        year_topic_vector: NumPy array of topic weights (float64).

    Returns:
        None
    """
    with open(path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ["base_topic_id", "base_topic_text", "sum_semantic_similarity"]
        )
        for base_topic_id, value in zip(base_topic_ids, year_topic_vector, strict=True):
            csv_writer.writerow(
                [
                    int(base_topic_id),
                    base_topic_text_by_id[int(base_topic_id)],
                    float(value),
                ]
            )


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def normalize_year_columns(year_topic_matrix: np.ndarray) -> np.ndarray:
    column_sums = year_topic_matrix.sum(axis=0)
    normalized = np.zeros_like(year_topic_matrix)
    nonzero_columns = column_sums != 0.0
    normalized[:, nonzero_columns] = (
        year_topic_matrix[:, nonzero_columns] / column_sums[nonzero_columns]
    )
    return normalized


def write_year_matrix_csv(
    path: str,
    base_topic_ids,
    base_topic_text_by_id,
    years: list[int],
    year_topic_matrix: np.ndarray,
) -> None:
    with open(path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["base_topic_id", "base_topic_text", *years])
        for base_topic_index, base_topic_id in enumerate(base_topic_ids):
            csv_writer.writerow(
                [
                    int(base_topic_id),
                    base_topic_text_by_id[int(base_topic_id)],
                    *year_topic_matrix[base_topic_index, :].tolist(),
                ]
            )


def main():
    """Entry point for CLI execution.

    Parses arguments, computes year topic vectors for the requested year range, writes a CSV
    matrix (base_topic_id, base_topic_text, per-year columns), and prints per-year topic rows
    to stdout.

    Returns:
        None
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--db", default=DB_PATH)
    argument_parser.add_argument("--start-year", type=int, required=True)
    argument_parser.add_argument("--end-year", type=int, required=True)
    args = argument_parser.parse_args()
    timestamp = timestamp_suffix()
    csv_path = f"subject_vector_by_year_{timestamp}.csv"
    normalized_csv_path = f"analyze_subject_vector_by_year_normalized_{timestamp}.csv"

    engine = create_engine(
        f"sqlite:///{args.db}",
        future=True,
        connect_args={"timeout": 60},
    )

    with Session(engine) as session:
        years = list(range(args.start_year, args.end_year + 1))

        base_topic_ids, base_topic_text_by_id, base_topic_index_by_id = (
            _load_base_topics(session)
        )

        year_topic_matrix = np.zeros(
            (len(base_topic_ids), len(years)), dtype=np.float64
        )

        for year_index, year_value in enumerate(years):
            year_topic_vector = year_vector(
                session,
                year=year_value,
                base_topic_ids_all=base_topic_ids,
                base_topic_index_by_id=base_topic_index_by_id,
            )
            year_topic_matrix[:, year_index] = year_topic_vector

        write_year_matrix_csv(
            csv_path,
            base_topic_ids,
            base_topic_text_by_id,
            years,
            year_topic_matrix,
        )

        write_year_matrix_csv(
            normalized_csv_path,
            base_topic_ids,
            base_topic_text_by_id,
            years,
            normalize_year_columns(year_topic_matrix),
        )

        for year_index, year_value in enumerate(years):
            year_topic_vector = year_topic_matrix[:, year_index]
            print(f"year={year_value} topics={len(year_topic_vector)}")
            for base_topic_id, value in zip(
                base_topic_ids, year_topic_vector, strict=True
            ):
                base_topic_text = base_topic_text_by_id[base_topic_id]
                print(f"{base_topic_id}\t{value:.6f}\t{base_topic_text}")


if __name__ == "__main__":
    main()
