from __future__ import annotations

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
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.close()


def _load_base_topics(session: Session):
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
