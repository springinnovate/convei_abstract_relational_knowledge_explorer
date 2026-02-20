from __future__ import annotations

"""
Compute per-year affiliation-type distributions for publications in a SQLite database.

This script aggregates semantic similarity scores between publications and
affiliation types into yearly distributions. For each publication in a given
year, it computes a temperature-scaled softmax over that publication's
affiliation-type similarity scores. These per-publication distributions are
summed to produce a year-level affiliation-type vector.

The result is written to a CSV file with one row per affiliation type and one
column per year.

Example:
    python affiliation_type_year_distribution.py \
        --db 2025_11_09_researchgate.sqlite \
        --start-year 2015 \
        --end-year 2020 \
        --temperature 0.3 \
        --csv out.csv
"""

import argparse
import csv

import numpy as np
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from models import (
    AffiliationType,
    AffiliationTypeToPublicationDistance,
    Publication,
)

DB_PATH = "2025_11_09_researchgate.sqlite"


def _set_sqlite_pragma(dbapi_connection, _):
    """Configure SQLite pragmas for performance and reliability.

    Enables WAL mode, sets synchronous mode to NORMAL, and configures
    a busy timeout to reduce locking issues during concurrent access.

    Args:
        dbapi_connection: A DB-API connection object provided by SQLAlchemy.
        _: Unused connection record parameter provided by SQLAlchemy.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def _load_affiliation_types(session: Session):
    """Load affiliation types and build lookup structures.

    Queries all affiliation types ordered by ID and constructs:
        - A NumPy array of affiliation type IDs.
        - A mapping from affiliation type ID to short name.
        - A mapping from affiliation type ID to its index in the dense array.

    Args:
        session: An active SQLAlchemy session.

    Returns:
        tuple:
            aff_type_ids (np.ndarray): Array of affiliation type IDs (int64).
            aff_type_text_by_id (dict[int, str]): ID -> short name mapping.
            aff_type_index_by_id (dict[int, int]): ID -> dense index mapping.
    """
    rows = session.execute(
        select(AffiliationType.id, AffiliationType.short_name).order_by(
            AffiliationType.id
        )
    ).all()
    aff_type_ids = np.array([row[0] for row in rows], dtype=np.int64)
    aff_type_text_by_id = {aff_type_id: text for aff_type_id, text in rows}
    aff_type_index_by_id = {
        aff_type_id: i for i, aff_type_id in enumerate(aff_type_ids)
    }
    return aff_type_ids, aff_type_text_by_id, aff_type_index_by_id


def year_vector(session: Session, year: int, temperature: float):
    """Compute a year-level affiliation-type distribution vector.

    For each publication in the specified year:
        1. Collects (affiliation_type_id, semantic_similarity) pairs.
        2. Applies temperature-scaled softmax to the similarity scores.
        3. Accumulates the resulting weights into a year-level vector.

    Args:
        session: An active SQLAlchemy session.
        year: Publication year to aggregate.
        temperature: Softmax temperature parameter controlling sharpness
            of the per-publication distribution.

    Returns:
        tuple:
            aff_type_ids_all (np.ndarray): Ordered affiliation type IDs.
            aff_type_text_by_id (dict[int, str]): ID -> short name mapping.
            year_aff_vector (np.ndarray): Summed softmax weights per affiliation type.
    """
    print(f"[year_vector] start year={year} temperature={temperature}")

    aff_type_ids_all, aff_type_text_by_id, aff_type_index_by_id = (
        _load_affiliation_types(session)
    )
    year_aff_vector = np.zeros(len(aff_type_ids_all), dtype=np.float64)

    query = (
        select(
            AffiliationTypeToPublicationDistance.publication_id,
            AffiliationTypeToPublicationDistance.affiliation_type_id,
            AffiliationTypeToPublicationDistance.semantic_similarity,
        )
        .join(
            Publication,
            Publication.id
            == AffiliationTypeToPublicationDistance.publication_id,
        )
        .where(Publication.publication_year == year)
        .order_by(
            AffiliationTypeToPublicationDistance.publication_id,
            AffiliationTypeToPublicationDistance.affiliation_type_id,
        )
    )

    rows = session.execute(query)

    current_publication_id = None
    publication_aff_type_ids: list[int] = []
    publication_scores: list[float] = []

    def flush():
        """Aggregate accumulated rows for a single publication."""
        if not publication_scores:
            return

        scores = np.array(publication_scores, dtype=np.float64)
        scores = np.maximum(scores, 0.0)
        scores = scores / temperature

        max_score = float(scores.max())
        exponentials = np.exp(scores - max_score)
        denom = float(exponentials.sum())
        if denom == 0.0:
            return

        weights = exponentials / denom

        for aff_type_id, weight in zip(
            publication_aff_type_ids, weights, strict=True
        ):
            year_aff_vector[aff_type_index_by_id[aff_type_id]] += float(weight)

    for publication_id, aff_type_id, semantic_similarity in rows:
        if current_publication_id is None:
            current_publication_id = publication_id

        if publication_id != current_publication_id:
            flush()
            current_publication_id = publication_id
            publication_aff_type_ids = []
            publication_scores = []

        publication_aff_type_ids.append(int(aff_type_id))
        publication_scores.append(float(semantic_similarity))

    flush()

    print(f"[year_vector] done year={year}")
    return aff_type_ids_all, aff_type_text_by_id, year_aff_vector


def main():
    """Parse CLI arguments and compute affiliation-type distributions by year.

    This function:
        1. Parses command-line arguments.
        2. Configures a SQLite engine with WAL and timeout settings.
        3. Computes year-level affiliation-type vectors for a range of years.
        4. Writes the resulting matrix to CSV.
        5. Prints per-year distributions to stdout.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--csv", default="affiliation_type_distribution_by_year.csv"
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    engine = create_engine(
        f"sqlite:///{args.db}",
        future=True,
        connect_args={"timeout": 60},
    )
    event.listen(engine, "connect", _set_sqlite_pragma)

    years = list(range(args.start_year, args.end_year + 1))

    with Session(engine) as session:
        aff_type_ids, aff_type_text_by_id, year_vec0 = year_vector(
            session,
            year=years[0],
            temperature=args.temperature,
        )

        year_matrix = np.zeros(
            (len(aff_type_ids), len(years)), dtype=np.float64
        )
        year_matrix[:, 0] = year_vec0

        for year_index, year_value in enumerate(years[1:], start=1):
            _, _, year_vec = year_vector(
                session,
                year=year_value,
                temperature=args.temperature,
            )
            year_matrix[:, year_index] = year_vec

    with open(args.csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["affiliation_type_id", "affiliation_type_short_name", *years]
        )
        for i, aff_type_id in enumerate(aff_type_ids):
            writer.writerow(
                [
                    int(aff_type_id),
                    aff_type_text_by_id[int(aff_type_id)],
                    *year_matrix[i, :].tolist(),
                ]
            )

    for year_index, year_value in enumerate(years):
        vec = year_matrix[:, year_index]
        print(f"year={year_value}")
        for aff_type_id, value in zip(aff_type_ids, vec, strict=True):
            print(
                f"{int(aff_type_id)}\t{float(value):.6f}\t{aff_type_text_by_id[int(aff_type_id)]}"
            )


if __name__ == "__main__":
    main()
