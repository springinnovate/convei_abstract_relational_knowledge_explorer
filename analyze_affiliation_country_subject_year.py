from datetime import datetime
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd
from sqlalchemy import create_engine, text

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

# The database stores raw semantic similarities. These CTEs derive report-time
# weights by clipping negatives, raising positives to the fourth power, and
# normalizing each publication vector so it sums to 1.0 before aggregation.
common_ctes = """
with country_map as (
    select distinct
        ppal.publication_id,
        l.name as country
    from publication_primary_author_locations ppal
    join locations l on l.id = ppal.location_id
    where lower(l.type) = 'country'
)
"""

soft_weight_ctes = """
with subject_power as (
    select
        btpd.publication_id,
        bt.short_name as subject,
        case
            when btpd.semantic_similarity > 0
            then btpd.semantic_similarity
                * btpd.semantic_similarity
                * btpd.semantic_similarity
                * btpd.semantic_similarity
            else 0
        end as subject_power
    from base_topic_to_pub_distance btpd
    join base_topics bt on bt.id = btpd.base_topic_id
),
positive_subject as (
    select
        publication_id,
        subject,
        subject_power
            / sum(subject_power) over (partition by publication_id)
            as subject_weight
    from subject_power
    where subject_power > 0
),
affiliation_power as (
    select
        atpd.publication_id,
        aft.short_name as affiliation_type,
        case
            when atpd.semantic_similarity > 0
            then atpd.semantic_similarity
                * atpd.semantic_similarity
                * atpd.semantic_similarity
                * atpd.semantic_similarity
            else 0
        end as affiliation_power
    from affiliation_type_to_pub_distance atpd
    join affiliation_types aft on aft.id = atpd.affiliation_type_id
),
positive_affiliation as (
    select
        publication_id,
        affiliation_type,
        affiliation_power
            / sum(affiliation_power) over (partition by publication_id)
            as affiliation_weight
    from affiliation_power
    where affiliation_power > 0
)
"""

reports = [
    (
        "affiliation_type_vs_subject",
        "Affiliation type",
        "Subject",
        "Affiliation type vs subject",
        False,
        soft_weight_ctes
        + """
        select
            pa.affiliation_type as row_label,
            ps.subject as column_label,
            sum(pa.affiliation_weight * ps.subject_weight) as count
        from positive_affiliation pa
        join positive_subject ps on ps.publication_id = pa.publication_id
        group by pa.affiliation_type, ps.subject
        """,
    ),
    (
        "country_vs_year",
        "Country",
        "Year",
        "Country vs year",
        True,
        common_ctes
        + """
        select
            cm.country as row_label,
            p.publication_year as column_label,
            count(distinct p.id) as count
        from publications p
        join country_map cm on cm.publication_id = p.id
        where p.publication_year is not null
        group by cm.country, p.publication_year
        """,
    ),
    (
        "country_vs_subject",
        "Country",
        "Subject",
        "Country vs subject",
        False,
        soft_weight_ctes
        + """
        , country_map as (
            select distinct
                ppal.publication_id,
                l.name as country
            from publication_primary_author_locations ppal
            join locations l on l.id = ppal.location_id
            where lower(l.type) = 'country'
        )
        select
            cm.country as row_label,
            ps.subject as column_label,
            sum(ps.subject_weight) as count
        from country_map cm
        join positive_subject ps on ps.publication_id = cm.publication_id
        group by cm.country, ps.subject
        """,
    ),
    (
        "country_vs_affiliation_type",
        "Country",
        "Affiliation type",
        "Country vs affiliation type",
        False,
        soft_weight_ctes
        + """
        , country_map as (
            select distinct
                ppal.publication_id,
                l.name as country
            from publication_primary_author_locations ppal
            join locations l on l.id = ppal.location_id
            where lower(l.type) = 'country'
        )
        select
            cm.country as row_label,
            pa.affiliation_type as column_label,
            sum(pa.affiliation_weight) as count
        from country_map cm
        join positive_affiliation pa on pa.publication_id = cm.publication_id
        group by cm.country, pa.affiliation_type
        """,
    ),
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


engine = create_engine(f"sqlite:///{db_path}")

logger.info("Starting affiliation country/subject/year analysis")
logger.info("Database: %s", db_path)
logger.info("Output directory: %s", out_dir)
logger.info("Timestamp suffix: %s", timestamp)

report_iterator = reports
if tqdm is not None:
    report_iterator = tqdm(reports, desc="reports", unit="report")

with engine.connect() as conn:
    for stem, row_name, col_name, title, integer_values, query in report_iterator:
        report_start = perf_counter()
        logger.info("Starting report: %s", stem)
        logger.info("Running SQL query for %s", stem)
        query_start = perf_counter()
        df = pd.read_sql_query(text(query), conn)
        logger.info(
            "SQL complete for %s: rows=%s elapsed=%s",
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
