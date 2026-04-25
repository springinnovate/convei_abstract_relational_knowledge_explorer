from __future__ import annotations

"""
Compute per-year affiliation-type distributions for publications in a SQLite database.

This script aggregates semantic similarity scores between publications and
affiliation types into yearly distributions. For each year, it sums positive
semantic similarity scores for each affiliation type across publications to
produce a year-level affiliation-type vector. It also writes a normalized
companion matrix where each year column sums to 1.0.

The result is written to a CSV file with one row per affiliation type and one
column per year.

Example:
    python affiliation_type_year_distribution.py \
        --db 2025_11_09_researchgate.sqlite \
        --start-year 2015 \
        --end-year 2020
"""

import argparse
import csv
from datetime import datetime

import numpy as np
from sqlalchemy import create_engine, event, func, select
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


def year_vector(
    session: Session,
    year: int,
    aff_type_ids_all,
    aff_type_index_by_id: dict[int, int],
):
    """Compute a year-level affiliation-type distribution vector.

    For each publication in the specified year, this function adds each positive
    affiliation-type semantic similarity score to the year-level type total.

    Args:
        session: An active SQLAlchemy session.
        year: Publication year to aggregate.
        aff_type_ids_all: Ordered affiliation type IDs.
        aff_type_index_by_id: ID -> dense index mapping.

    Returns:
        np.ndarray: Summed positive semantic similarity per affiliation type.
    """
    print(f"[year_vector] start year={year}")

    year_aff_vector = np.zeros(len(aff_type_ids_all), dtype=np.float64)

    query = (
        select(
            AffiliationTypeToPublicationDistance.affiliation_type_id,
            func.sum(AffiliationTypeToPublicationDistance.semantic_similarity),
        )
        .join(
            Publication,
            Publication.id
            == AffiliationTypeToPublicationDistance.publication_id,
        )
        .where(Publication.publication_year == year)
        .where(AffiliationTypeToPublicationDistance.semantic_similarity > 0.0)
        .group_by(AffiliationTypeToPublicationDistance.affiliation_type_id)
        .order_by(AffiliationTypeToPublicationDistance.affiliation_type_id)
    )

    for aff_type_id, semantic_similarity_sum in session.execute(query):
        year_aff_vector[aff_type_index_by_id[aff_type_id]] = float(
            semantic_similarity_sum or 0.0
        )

    print(f"[year_vector] done year={year}")
    return year_aff_vector


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def normalize_year_columns(year_matrix: np.ndarray) -> np.ndarray:
    column_sums = year_matrix.sum(axis=0)
    normalized = np.zeros_like(year_matrix)
    nonzero_columns = column_sums != 0.0
    normalized[:, nonzero_columns] = (
        year_matrix[:, nonzero_columns] / column_sums[nonzero_columns]
    )
    return normalized


def write_year_matrix_csv(
    path: str,
    aff_type_ids,
    aff_type_text_by_id,
    years: list[int],
    year_matrix: np.ndarray,
) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["affiliation_type_id", "affiliation_type_short_name", *years])
        for i, aff_type_id in enumerate(aff_type_ids):
            writer.writerow(
                [
                    int(aff_type_id),
                    aff_type_text_by_id[int(aff_type_id)],
                    *year_matrix[i, :].tolist(),
                ]
            )


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
    args = parser.parse_args()
    timestamp = timestamp_suffix()
    csv_path = f"analyze_affiliation_vector_by_year_{timestamp}.csv"
    normalized_csv_path = (
        f"analyze_affiliation_vector_by_year_normalized_{timestamp}.csv"
    )

    engine = create_engine(
        f"sqlite:///{args.db}",
        future=True,
        connect_args={"timeout": 60},
    )
    event.listen(engine, "connect", _set_sqlite_pragma)

    years = list(range(args.start_year, args.end_year + 1))

    with Session(engine) as session:
        aff_type_ids, aff_type_text_by_id, aff_type_index_by_id = (
            _load_affiliation_types(session)
        )

        year_matrix = np.zeros(
            (len(aff_type_ids), len(years)), dtype=np.float64
        )

        for year_index, year_value in enumerate(years):
            year_vec = year_vector(
                session,
                year=year_value,
                aff_type_ids_all=aff_type_ids,
                aff_type_index_by_id=aff_type_index_by_id,
            )
            year_matrix[:, year_index] = year_vec

    write_year_matrix_csv(
        csv_path,
        aff_type_ids,
        aff_type_text_by_id,
        years,
        year_matrix,
    )
    write_year_matrix_csv(
        normalized_csv_path,
        aff_type_ids,
        aff_type_text_by_id,
        years,
        normalize_year_columns(year_matrix),
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
