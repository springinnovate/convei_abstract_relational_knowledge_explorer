from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, text

db_path = "2025_11_09_researchgate.sqlite"
out_dir = Path("topic_mapping_reports")
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

out_dir.mkdir(exist_ok=True)

common_ctes = """
with top_subject as (
    select publication_id, short_name as subject
    from (
        select
            btpd.publication_id,
            bt.short_name,
            row_number() over (
                partition by btpd.publication_id
                order by btpd.semantic_similarity desc, bt.id
            ) as rn
        from base_topic_to_pub_distance btpd
        join base_topics bt on bt.id = btpd.base_topic_id
        where btpd.semantic_similarity > 0
    )
    where rn = 1
),
top_affiliation as (
    select publication_id, short_name as affiliation_type
    from (
        select
            atpd.publication_id,
            aft.short_name,
            row_number() over (
                partition by atpd.publication_id
                order by atpd.semantic_similarity desc, aft.id
            ) as rn
        from affiliation_type_to_pub_distance atpd
        join affiliation_types aft on aft.id = atpd.affiliation_type_id
        where atpd.semantic_similarity > 0
    )
    where rn = 1
),
country_map as (
    select distinct
        ppal.publication_id,
        l.name as country
    from publication_primary_author_locations ppal
    join locations l on l.id = ppal.location_id
    where lower(l.type) = 'country'
)
"""

reports = [
    (
        "affiliation_type_vs_subject",
        "Affiliation type",
        "Subject",
        "Affiliation type vs subject",
        common_ctes
        + """
        select
            ta.affiliation_type as row_label,
            ts.subject as column_label,
            count(distinct p.id) as count
        from publications p
        join top_subject ts on ts.publication_id = p.id
        join top_affiliation ta on ta.publication_id = p.id
        group by ta.affiliation_type, ts.subject
        """,
    ),
    (
        "country_vs_year",
        "Country",
        "Year",
        "Country vs year",
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
        common_ctes
        + """
        select
            cm.country as row_label,
            ts.subject as column_label,
            count(distinct p.id) as count
        from publications p
        join country_map cm on cm.publication_id = p.id
        join top_subject ts on ts.publication_id = p.id
        group by cm.country, ts.subject
        """,
    ),
    (
        "country_vs_affiliation_type",
        "Country",
        "Affiliation type",
        "Country vs affiliation type",
        common_ctes
        + """
        select
            cm.country as row_label,
            ta.affiliation_type as column_label,
            count(distinct p.id) as count
        from publications p
        join country_map cm on cm.publication_id = p.id
        join top_affiliation ta on ta.publication_id = p.id
        group by cm.country, ta.affiliation_type
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


engine = create_engine(f"sqlite:///{db_path}")

with engine.connect() as conn:
    for stem, row_name, col_name, title, query in reports:
        df = pd.read_sql_query(text(query), conn)
        pivot = (
            df.pivot(index="row_label", columns="column_label", values="count")
            .fillna(0)
            .astype(int)
        )
        pivot = pivot.reindex(sorted(pivot.index.tolist()), axis=0)
        pivot = pivot.reindex(sorted(pivot.columns.tolist()), axis=1)
        pivot.index.name = row_name
        pivot.columns.name = col_name

        csv_path = out_dir / f"{stem}_{timestamp}.csv"
        png_path = out_dir / f"{stem}_{timestamp}.png"

        pivot.to_csv(csv_path)
        if col_name == "Year":
            normalized_csv_path = out_dir / f"{stem}_normalized_{timestamp}.csv"
            normalize_year_columns(pivot).to_csv(normalized_csv_path)

        fig, ax = plt.subplots(
            figsize=(
                max(8, pivot.shape[1] * 0.6),
                max(6, pivot.shape[0] * 0.35),
            )
        )
        image = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(col_name)
        ax.set_ylabel(row_name)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([str(x) for x in pivot.columns], rotation=90)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels([str(x) for x in pivot.index])
        fig.colorbar(image, ax=ax, label="Publication count")
        fig.tight_layout()
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
