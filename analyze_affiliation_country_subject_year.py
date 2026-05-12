from datetime import datetime
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from affiliation_vector_transform import power_normalize

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

db_path = "2025_11_09_researchgate.sqlite"
out_dir = Path("topic_mapping_reports")
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

out_dir.mkdir(exist_ok=True)

COUNTRY_MAP_QUERY = """
select distinct
    ppal.publication_id,
    l.name as country
from publication_primary_author_locations ppal
join locations l on l.id = ppal.location_id
where lower(l.type) = 'country'
"""

RAW_SUBJECT_QUERY = """
select
    btpd.publication_id,
    bt.short_name as label,
    btpd.semantic_similarity
from base_topic_to_pub_distance btpd
join base_topics bt on bt.id = btpd.base_topic_id
order by btpd.publication_id, btpd.base_topic_id
"""

RAW_AFFILIATION_QUERY = """
select
    atpd.publication_id,
    aft.short_name as label,
    atpd.semantic_similarity
from affiliation_type_to_pub_distance atpd
join affiliation_types aft on aft.id = atpd.affiliation_type_id
order by atpd.publication_id, atpd.affiliation_type_id
"""

COUNTRY_VS_YEAR_QUERY = f"""
with country_map as (
    {COUNTRY_MAP_QUERY}
)
select
    cm.country as row_label,
    p.publication_year as column_label,
    count(distinct p.id) as count
from publications p
join country_map cm on cm.publication_id = p.id
where p.publication_year is not null
group by cm.country, p.publication_year
"""

reports = [
    ("affiliation_type_vs_subject", "Affiliation type", "Subject", False),
    ("country_vs_year", "Country", "Year", True),
    ("country_vs_subject", "Country", "Subject", False),
    ("country_vs_affiliation_type", "Country", "Affiliation type", False),
]


def normalize_year_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    column_sums = pivot.sum(axis=0)
    normalized = pivot.astype(float).copy()
    nonzero_columns = column_sums != 0
    normalized.loc[:, nonzero_columns] = (
        normalized.loc[:, nonzero_columns] / column_sums[nonzero_columns]
    )
    normalized.loc[:, ~nonzero_columns] = 0.0
    return normalized


def elapsed_seconds(start_time: float) -> str:
    return f"{perf_counter() - start_time:.1f}s"


def empty_report_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["row_label", "column_label", "count"])


def load_power_normalized_publication_weights(
    conn,
    query: str,
    label_column: str,
    weight_column: str,
) -> pd.DataFrame:
    """Load raw DB similarities and derive report-time publication weights."""
    raw = pd.read_sql_query(text(query), conn)
    if raw.empty:
        return pd.DataFrame(
            columns=["publication_id", label_column, weight_column]
        )

    rows = []
    for publication_id, group in raw.groupby("publication_id", sort=False):
        weights = power_normalize(
            group["semantic_similarity"].to_numpy(dtype=np.float64)
        )
        for label, weight in zip(group["label"], weights):
            if weight > 0.0:
                rows.append(
                    {
                        "publication_id": publication_id,
                        label_column: label,
                        weight_column: float(weight),
                    }
                )

    return pd.DataFrame(
        rows, columns=["publication_id", label_column, weight_column]
    )


def load_report_inputs(conn):
    logger.info("Loading raw subject similarities")
    subject_weights = load_power_normalized_publication_weights(
        conn,
        RAW_SUBJECT_QUERY,
        "subject",
        "subject_weight",
    )
    logger.info("Loaded transformed subject weights: rows=%s", len(subject_weights))

    logger.info("Loading raw affiliation similarities")
    affiliation_weights = load_power_normalized_publication_weights(
        conn,
        RAW_AFFILIATION_QUERY,
        "affiliation_type",
        "affiliation_weight",
    )
    logger.info(
        "Loaded transformed affiliation weights: rows=%s",
        len(affiliation_weights),
    )

    logger.info("Loading country map")
    country_map = pd.read_sql_query(text(COUNTRY_MAP_QUERY), conn)
    logger.info("Loaded country map: rows=%s", len(country_map))

    return subject_weights, affiliation_weights, country_map


def build_affiliation_type_vs_subject(
    subject_weights: pd.DataFrame,
    affiliation_weights: pd.DataFrame,
) -> pd.DataFrame:
    if subject_weights.empty or affiliation_weights.empty:
        return empty_report_frame()

    affiliation_subject = affiliation_weights.merge(
        subject_weights, on="publication_id", how="inner"
    )
    affiliation_subject["count"] = (
        affiliation_subject["affiliation_weight"]
        * affiliation_subject["subject_weight"]
    )
    return (
        affiliation_subject.groupby(["affiliation_type", "subject"], as_index=False)[
            "count"
        ]
        .sum()
        .rename(
            columns={
                "affiliation_type": "row_label",
                "subject": "column_label",
            }
        )
    )


def build_country_vs_subject(
    country_map: pd.DataFrame,
    subject_weights: pd.DataFrame,
) -> pd.DataFrame:
    if country_map.empty or subject_weights.empty:
        return empty_report_frame()

    country_subject = country_map.merge(
        subject_weights, on="publication_id", how="inner"
    )
    return (
        country_subject.groupby(["country", "subject"], as_index=False)[
            "subject_weight"
        ]
        .sum()
        .rename(
            columns={
                "country": "row_label",
                "subject": "column_label",
                "subject_weight": "count",
            }
        )
    )


def build_country_vs_affiliation_type(
    country_map: pd.DataFrame,
    affiliation_weights: pd.DataFrame,
) -> pd.DataFrame:
    if country_map.empty or affiliation_weights.empty:
        return empty_report_frame()

    country_affiliation = country_map.merge(
        affiliation_weights, on="publication_id", how="inner"
    )
    return (
        country_affiliation.groupby(["country", "affiliation_type"], as_index=False)[
            "affiliation_weight"
        ]
        .sum()
        .rename(
            columns={
                "country": "row_label",
                "affiliation_type": "column_label",
                "affiliation_weight": "count",
            }
        )
    )


def build_report_frames(conn) -> dict[str, pd.DataFrame]:
    # The database stores raw semantic similarities. These data frames hold
    # report-time weights derived with the shared fourth-power normalizer.
    subject_weights, affiliation_weights, country_map = load_report_inputs(conn)

    return {
        "affiliation_type_vs_subject": build_affiliation_type_vs_subject(
            subject_weights,
            affiliation_weights,
        ),
        "country_vs_year": pd.read_sql_query(text(COUNTRY_VS_YEAR_QUERY), conn),
        "country_vs_subject": build_country_vs_subject(
            country_map,
            subject_weights,
        ),
        "country_vs_affiliation_type": build_country_vs_affiliation_type(
            country_map,
            affiliation_weights,
        ),
    }


engine = create_engine(f"sqlite:///{db_path}")

logger.info("Starting affiliation country/subject/year analysis")
logger.info("Database: %s", db_path)
logger.info("Output directory: %s", out_dir)
logger.info("Timestamp suffix: %s", timestamp)

report_iterator = reports
if tqdm is not None:
    report_iterator = tqdm(reports, desc="reports", unit="report")

with engine.connect() as conn:
    report_frames = build_report_frames(conn)

    for stem, row_name, col_name, integer_values in report_iterator:
        report_start = perf_counter()
        logger.info("Starting report: %s", stem)
        query_start = perf_counter()
        df = report_frames[stem]
        logger.info(
            "Data ready for %s: rows=%s elapsed=%s",
            stem,
            len(df),
            elapsed_seconds(query_start),
        )

        logger.info("Building pivot for %s", stem)
        pivot_start = perf_counter()
        pivot = (
            df.pivot(index="row_label", columns="column_label", values="count")
            .fillna(0)
        )
        if integer_values:
            pivot = pivot.astype(int)
        else:
            pivot = pivot.astype(float)
        pivot = pivot.reindex(sorted(pivot.index.tolist()), axis=0)
        pivot = pivot.reindex(sorted(pivot.columns.tolist()), axis=1)
        pivot.index.name = row_name
        pivot.columns.name = col_name
        logger.info(
            "Pivot complete for %s: rows=%s columns=%s elapsed=%s",
            stem,
            pivot.shape[0],
            pivot.shape[1],
            elapsed_seconds(pivot_start),
        )

        csv_path = out_dir / f"{stem}_{timestamp}.csv"
        logger.info("Writing CSV for %s: %s", stem, csv_path)
        pivot.to_csv(csv_path)
        if col_name == "Year":
            normalized_csv_path = out_dir / f"{stem}_normalized_{timestamp}.csv"
            logger.info(
                "Writing normalized year CSV for %s: %s",
                stem,
                normalized_csv_path,
            )
            normalize_year_columns(pivot).to_csv(normalized_csv_path)
        logger.info("Finished report: %s elapsed=%s", stem, elapsed_seconds(report_start))

logger.info("Finished affiliation country/subject/year analysis")
