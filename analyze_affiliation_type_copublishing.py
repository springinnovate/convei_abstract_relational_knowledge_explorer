from __future__ import annotations

"""
Build an affiliation-type co-publishing matrix across authors on the same paper.

For each publication, the script loads positive affiliation-type weights for
each author. It then adds pairwise products between every author and every other
author on that publication into a directed matrix: source affiliation type ->
target affiliation type.

Same-type to same-type products are included when they come from different
authors. Self-products from the same author are excluded.
"""

import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from models import (
    AffiliationType,
    PublicationAuthorLocation,
    PublicationAuthorLocationAffiliationTypeDistance,
)

DB_PATH = "2025_11_09_researchgate.sqlite"
REPORTS_DIR = Path("topic_mapping_reports")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _set_sqlite_pragma(dbapi_connection, _):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def elapsed_seconds(start_time: float) -> str:
    return f"{perf_counter() - start_time:.1f}s"


def load_affiliation_types(session: Session):
    rows = session.execute(
        select(AffiliationType.id, AffiliationType.short_name).order_by(
            AffiliationType.short_name
        )
    ).all()
    affiliation_type_ids = [row[0] for row in rows]
    affiliation_type_names = [row[1] for row in rows]
    affiliation_type_index_by_id = {
        affiliation_type_id: index
        for index, affiliation_type_id in enumerate(affiliation_type_ids)
    }
    return affiliation_type_names, affiliation_type_index_by_id


def iter_positive_author_affiliation_weights(session: Session):
    query = (
        select(
            PublicationAuthorLocation.publication_id,
            PublicationAuthorLocation.id,
            PublicationAuthorLocation.author_index,
            PublicationAuthorLocation.author_name,
            PublicationAuthorLocationAffiliationTypeDistance.affiliation_type_id,
            PublicationAuthorLocationAffiliationTypeDistance.semantic_similarity,
        )
        .join(
            PublicationAuthorLocation,
            PublicationAuthorLocation.id
            == PublicationAuthorLocationAffiliationTypeDistance.publication_author_location_id,
        )
        .where(
            PublicationAuthorLocationAffiliationTypeDistance.semantic_similarity > 0.0
        )
        .order_by(
            PublicationAuthorLocation.publication_id,
            PublicationAuthorLocation.author_index,
            PublicationAuthorLocation.author_name,
            PublicationAuthorLocation.id,
            PublicationAuthorLocationAffiliationTypeDistance.affiliation_type_id,
        )
    )
    return session.execute(query)


def add_publication_to_matrix(
    matrix: np.ndarray,
    author_vectors: list[np.ndarray],
) -> None:
    if len(author_vectors) < 2:
        return

    publication_total = np.sum(author_vectors, axis=0)
    matrix += np.outer(publication_total, publication_total)
    for vector in author_vectors:
        matrix -= np.outer(vector, vector)


def build_copublishing_matrix(
    session: Session,
    affiliation_type_index_by_id: dict[int, int],
    affiliation_type_count: int,
) -> tuple[np.ndarray, int, int]:
    matrix = np.zeros((affiliation_type_count, affiliation_type_count), dtype=np.float64)

    current_publication_id = None
    current_author_key = None
    current_vector = None
    author_vectors: list[np.ndarray] = []
    publication_count = 0
    author_count = 0

    rows = iter_positive_author_affiliation_weights(session)
    row_iterator = rows
    if tqdm is not None:
        row_iterator = tqdm(rows, desc="Affiliation weights", unit="row")

    def flush_author() -> None:
        nonlocal current_vector, author_count
        if current_vector is None:
            return
        author_vectors.append(current_vector)
        author_count += 1
        current_vector = None

    def flush_publication() -> None:
        nonlocal publication_count
        flush_author()
        if not author_vectors:
            return
        add_publication_to_matrix(matrix, author_vectors)
        author_vectors.clear()
        publication_count += 1
        if publication_count % 10000 == 0:
            logger.info(
                "Processed %d publications and %d authors",
                publication_count,
                author_count,
            )

    for (
        publication_id,
        author_location_id,
        author_index,
        author_name,
        affiliation_type_id,
        semantic_similarity,
    ) in row_iterator:
        if current_publication_id is None:
            current_publication_id = publication_id

        if publication_id != current_publication_id:
            flush_publication()
            current_publication_id = publication_id
            current_author_key = None

        author_key = author_index if author_index is not None else author_name
        if author_key is None:
            author_key = author_location_id

        if author_key != current_author_key:
            flush_author()
            current_author_key = author_key
            current_vector = np.zeros(affiliation_type_count, dtype=np.float64)

        index = affiliation_type_index_by_id[affiliation_type_id]
        current_vector[index] += float(semantic_similarity)

    flush_publication()
    return matrix, publication_count, author_count


def write_matrix_csv(
    path: Path,
    affiliation_type_names: list[str],
    matrix: np.ndarray,
) -> None:
    logger.info("Writing CSV to %s", path)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["affiliation_type", *affiliation_type_names])
        for index, affiliation_type_name in enumerate(affiliation_type_names):
            writer.writerow([affiliation_type_name, *matrix[index, :].tolist()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)
    output_path = (
        REPORTS_DIR
        / f"analyze_affiliation_type_copublishing_{timestamp_suffix()}.csv"
    )

    logger.info("Starting affiliation type co-publishing analysis")
    logger.info("Database: %s", args.db)
    logger.info("Output CSV: %s", output_path)

    engine = create_engine(
        f"sqlite:///{args.db}",
        future=True,
        connect_args={"timeout": 60},
    )
    event.listen(engine, "connect", _set_sqlite_pragma)

    with Session(engine) as session:
        start_time = perf_counter()
        logger.info("Loading affiliation types")
        affiliation_type_names, affiliation_type_index_by_id = load_affiliation_types(
            session
        )
        logger.info("Loaded %d affiliation types", len(affiliation_type_names))

        logger.info("Building co-publishing matrix from positive weights")
        matrix, publication_count, author_count = build_copublishing_matrix(
            session,
            affiliation_type_index_by_id,
            len(affiliation_type_names),
        )
        logger.info(
            "Built matrix from %d publications and %d authors in %s",
            publication_count,
            author_count,
            elapsed_seconds(start_time),
        )

    write_matrix_csv(output_path, affiliation_type_names, matrix)
    logger.info("Done")


if __name__ == "__main__":
    main()
