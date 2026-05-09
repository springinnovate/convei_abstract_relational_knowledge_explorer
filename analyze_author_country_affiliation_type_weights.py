from __future__ import annotations

"""
Build a country-by-affiliation-type weight matrix for all author locations.

This analysis uses all author location rows in `publication_author_locations`,
not just primary-author locations. It fourth-power normalizes each
author-location affiliation-type vector from
`publication_author_location_to_affiliation_type_distance`, then sums those
transformed weights by country and affiliation type.
"""

import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from affiliation_vector_transform import power_normalize
from models import (
    AffiliationType,
    Location,
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
    return affiliation_type_ids, affiliation_type_names, affiliation_type_index_by_id


def iter_country_affiliation_weights(session: Session):
    country_name = func.trim(Location.name).label("country")
    query = (
        select(
            country_name,
            PublicationAuthorLocation.id,
            PublicationAuthorLocationAffiliationTypeDistance.affiliation_type_id,
            PublicationAuthorLocationAffiliationTypeDistance.semantic_similarity,
        )
        .join(
            PublicationAuthorLocation,
            PublicationAuthorLocation.id
            == PublicationAuthorLocationAffiliationTypeDistance.publication_author_location_id,
        )
        .join(Location, Location.id == PublicationAuthorLocation.location_id)
        .where(func.lower(Location.type) == "country")
        .where(
            PublicationAuthorLocationAffiliationTypeDistance.semantic_similarity > 0.0
        )
        .order_by(
            PublicationAuthorLocation.id,
            country_name,
            PublicationAuthorLocationAffiliationTypeDistance.affiliation_type_id,
        )
    )

    return session.execute(query)


def write_matrix_csv(
    path: Path,
    affiliation_type_names: list[str],
    affiliation_type_index_by_id: dict[int, int],
    country_weight_rows,
) -> None:
    countries: dict[str, list[float]] = {}
    row_iterator = country_weight_rows
    if tqdm is not None:
        row_iterator = tqdm(
            country_weight_rows, desc="Loading country weights", unit="row"
        )

    current_author_location_id = None
    current_country = None
    current_vector = None

    def flush_author_location() -> None:
        nonlocal current_vector
        if current_vector is None or current_country is None:
            return
        # Stored similarities stay raw; country totals use report-time weights.
        country_vector = countries.setdefault(
            current_country,
            [0.0 for _ in affiliation_type_names],
        )
        for index, weight in enumerate(power_normalize(current_vector)):
            if weight > 0.0:
                country_vector[index] += float(weight)
        current_vector = None

    for country, author_location_id, affiliation_type_id, semantic_similarity in row_iterator:
        if author_location_id != current_author_location_id:
            flush_author_location()
            current_author_location_id = author_location_id
            current_country = country
            current_vector = np.zeros(len(affiliation_type_names), dtype=np.float64)

        current_vector[affiliation_type_index_by_id[affiliation_type_id]] = float(
            semantic_similarity
        )

    flush_author_location()

    logger.info("Writing %d country rows to %s", len(countries), path)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["country", *affiliation_type_names])
        for country in sorted(countries):
            writer.writerow([country, *countries[country]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)
    output_path = (
        REPORTS_DIR
        / f"analyze_author_country_affiliation_type_weights_{timestamp_suffix()}.csv"
    )

    logger.info("Starting author country affiliation type weight analysis")
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
        _, affiliation_type_names, affiliation_type_index_by_id = load_affiliation_types(
            session
        )
        logger.info(
            "Loaded %d affiliation types in %s",
            len(affiliation_type_names),
            elapsed_seconds(start_time),
        )

        query_start = perf_counter()
        logger.info("Running country-by-affiliation-type weight query")
        country_weight_rows = iter_country_affiliation_weights(session)
        write_matrix_csv(
            output_path,
            affiliation_type_names,
            affiliation_type_index_by_id,
            country_weight_rows,
        )
        logger.info("Query and write completed in %s", elapsed_seconds(query_start))

    logger.info("Done")


if __name__ == "__main__":
    main()
