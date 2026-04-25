from __future__ import annotations

"""
Build a country-by-affiliation-type weight matrix for all author locations.

This analysis uses all author location rows in `publication_author_locations`,
not just primary-author locations. It sums positive semantic similarity weights
from `publication_author_location_to_affiliation_type_distance` by country and
affiliation type, then writes a CSV matrix to `topic_mapping_reports`.
"""

import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
from time import perf_counter

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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


def load_country_affiliation_weights(session: Session):
    country_name = func.trim(Location.name).label("country")
    query = (
        select(
            country_name,
            PublicationAuthorLocationAffiliationTypeDistance.affiliation_type_id,
            func.sum(
                PublicationAuthorLocationAffiliationTypeDistance.semantic_similarity
            ),
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
        .group_by(
            country_name,
            PublicationAuthorLocationAffiliationTypeDistance.affiliation_type_id,
        )
        .order_by(
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
        row_iterator = tqdm(country_weight_rows, desc="Loading country weights", unit="row")

    for country, affiliation_type_id, weight_sum in row_iterator:
        country_vector = countries.setdefault(
            country,
            [0.0 for _ in affiliation_type_names],
        )
        country_vector[affiliation_type_index_by_id[affiliation_type_id]] = float(
            weight_sum or 0.0
        )

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
        logger.info("Running grouped country-by-affiliation-type weight query")
        country_weight_rows = load_country_affiliation_weights(session)
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
