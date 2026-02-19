from __future__ import annotations

"""
Compute per-year base-topic distributions for publications stored in a SQLite database.

This script queries a ResearchGate-derived SQLite database containing publications, base topics,
and per-publication distances (semantic similarities) to base topics. For each publication in a
given year, it computes a temperature-scaled softmax distribution over that publication's base
topics (optionally reweighted by the publication's similarity to a chosen base topic), then
sums those distributions across all publications in the year to form a year-level topic vector.

The script can compute vectors across a year range and write a CSV matrix with one row per base
topic and one column per year.

Typical usage:
    python script.py --db path/to/db.sqlite --start-year 2015 --end-year 2020 \
        --short-name-weighted-topic "machine_learning" --csv out.csv

Notes:
    - Uses SQLite WAL mode and a busy timeout for better concurrency behavior.
    - Assumes the schema defined by models.BaseTopics, models.Publication, and
      models.BaseTopicToPublicationDistance.
"""

import argparse
import csv

import numpy as np
from sqlalchemy import create_engine, event, select
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
    weighted_short_name: str,
):
    """Compute a year-level base-topic vector by aggregating publication-level softmax weights.

    For each publication in the specified year, this function:
      1) collects (base_topic_id, semantic_similarity) pairs
      2) finds the publication's similarity for `weighted_short_name` (if present) and uses it
         as a multiplicative factor on all topic scores for that publication
      3) applies temperature-scaled softmax to the per-publication scores
      4) accumulates the resulting weights into a year-level vector over base topics

    Args:
        session: An active SQLAlchemy session.
        year: Publication year to aggregate.
        weighted_short_name: Base topic short name whose semantic similarity (per publication)
            is used as a multiplicative weight for that publication's scores.

    Returns:
        tuple:
            base_topic_ids_all: NumPy array of all base topic IDs (int64), ordered by ID.
            base_topic_text_by_id: Dict mapping base topic ID (int) -> short name (str).
            year_topic_vector: NumPy array (float64) with summed softmax weights per base topic.
    """
    print(f"[year_vector] start year={year} weighted_short_name={weighted_short_name}")

    (
        base_topic_ids_all,
        base_topic_text_by_id,
        base_topic_index_by_id,
    ) = _load_base_topics(session)

    print(f"[year_vector] loaded {len(base_topic_ids_all)} base topics")

    year_topic_vector = np.zeros(len(base_topic_ids_all), dtype=np.float64)

    weighted_base_topic_id = session.execute(
        select(BaseTopics.id).where(BaseTopics.short_name == weighted_short_name)
    ).scalar_one()

    print(f"[year_vector] weighted_base_topic_id={weighted_base_topic_id}")

    query = (
        select(
            BaseTopicToPublicationDistance.publication_id,
            BaseTopicToPublicationDistance.base_topic_id,
            BaseTopicToPublicationDistance.semantic_similarity,
        )
        .join(
            Publication,
            Publication.id == BaseTopicToPublicationDistance.publication_id,
        )
        .where(Publication.publication_year == year)
    )

    query = query.order_by(
        BaseTopicToPublicationDistance.publication_id,
        BaseTopicToPublicationDistance.base_topic_id,
    )

    print("[year_vector] executing main query")

    rows = session.execute(query)

    temperature = 0.3

    current_publication_id = None
    publication_base_topic_ids: list[int] = []
    publication_scores: list[float] = []

    publication_counter = 0
    row_counter = 0

    def flush():
        """Flush accumulated rows for the current publication into the year vector.

        Computes the per-publication softmax distribution over base topics (after optional
        weighting and temperature scaling) and adds it to `year_topic_vector`.

        Returns:
            None
        """
        nonlocal publication_counter

        if not publication_scores:
            return

        publication_counter += 1

        if publication_counter % 10000 == 0:
            print(
                f"[year_vector] flushed {publication_counter} publications "
                f"(rows processed={row_counter})"
            )

        weighted_score = 1.0
        for base_topic_id, semantic_similarity in zip(
            publication_base_topic_ids, publication_scores, strict=True
        ):
            if base_topic_id == weighted_base_topic_id:
                weighted_score = float(semantic_similarity)
                break

        scores = np.array(publication_scores, dtype=np.float64)
        scores = np.maximum(scores, 0.0)
        scores = scores * weighted_score
        scores = scores / temperature

        max_score = float(scores.max())
        exponentials = np.exp(scores - max_score)
        denominator = float(exponentials.sum())
        if denominator == 0.0:
            return
        exponentials = exponentials / denominator
        for base_topic_id, weight in zip(
            publication_base_topic_ids, exponentials, strict=True
        ):
            year_topic_vector[base_topic_index_by_id[base_topic_id]] += float(weight)

    for publication_id, base_topic_id, semantic_similarity in rows:
        row_counter += 1

        if row_counter % 100000 == 0:
            print(f"[year_vector] processed {row_counter} rows")

        if current_publication_id is None:
            current_publication_id = publication_id

        if publication_id != current_publication_id:
            flush()
            current_publication_id = publication_id
            publication_base_topic_ids = []
            publication_scores = []

        publication_base_topic_ids.append(base_topic_id)
        publication_scores.append(semantic_similarity)

    flush()

    print(
        f"[year_vector] done year={year} "
        f"total_rows={row_counter} total_publications={publication_counter}"
    )

    return base_topic_ids_all, base_topic_text_by_id, year_topic_vector


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
    argument_parser.add_argument("--short-name-weighted-topic", type=str, required=True)
    argument_parser.add_argument("--csv", default=None)
    args = argument_parser.parse_args()

    engine = create_engine(
        f"sqlite:///{args.db}",
        future=True,
        connect_args={"timeout": 60},
    )

    with Session(engine) as session:
        years = list(range(args.start_year, args.end_year + 1))

        base_topic_ids, base_topic_text_by_id, _ = year_vector(
            session,
            year=years[0],
            weighted_short_name=args.short_name_weighted_topic,
        )

        year_topic_matrix = np.zeros(
            (len(base_topic_ids), len(years)), dtype=np.float64
        )

        for year_index, year_value in enumerate(years):
            (
                base_topic_ids_for_year,
                base_topic_text_by_id_for_year,
                year_topic_vector,
            ) = year_vector(
                session,
                year=year_value,
                weighted_short_name=args.short_name_weighted_topic,
            )
            year_topic_matrix[:, year_index] = year_topic_vector

        with open(args.csv, "w", newline="") as file:
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
