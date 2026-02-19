import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, aliased

from models import BaseTopics, BaseTopicToPublicationDistance, Publication

DB_PATH = "sqlite:///2025_11_09_researchgate.sqlite"
WEIGHTED_TOPIC_TEXT = "Satellites that Observe and Image the Earth's Surface"
N = 10

max_workers = min(32, (os.cpu_count() or 4) * 2)

engine = create_engine(
    DB_PATH,
    connect_args={"check_same_thread": False},
    pool_size=max_workers,
    max_overflow=max_workers,
)
SessionLocal = sessionmaker(bind=engine)


def top_n_for_base_topic(
    topic_id: int,
    base_topic_text: str,
    base_topic_short_name: str,
    weight_topic_id: int,
) -> pd.DataFrame:
    D = BaseTopicToPublicationDistance
    W = aliased(BaseTopicToPublicationDistance)

    sim = D.semantic_similarity
    weight_sim = W.semantic_similarity
    sim_c = func.min(func.max(sim, 0.0), 1.0)
    weight_sim_c = func.min(func.max(weight_sim, 0.0), 1.0)
    weighted = (sim_c * weight_sim_c).label("weighted_topic")

    stmt = (
        select(
            Publication.title,
            Publication.abstract,
            sim.label("sim"),
            weight_sim.label("weight_sim"),
            weighted,
        )
        .select_from(D)
        .join(Publication, Publication.id == D.publication_id)
        .join(
            W,
            (W.publication_id == D.publication_id)
            & (W.base_topic_id == weight_topic_id),
        )
        .where(D.base_topic_id == topic_id)
        .order_by(weighted.desc())
        .limit(N)
    )

    with SessionLocal() as session:
        rows = session.execute(stmt).all()

    df = pd.DataFrame(
        rows,
        columns=["title", "abstract", "sim", "weight_sim", "weighted_topic"],
    )
    df.insert(0, "base_topic", base_topic_short_name)
    return df[
        [
            "base_topic",
            "weighted_topic",
            "title",
            "abstract",
            "sim",
            "weight_sim",
        ]
    ]


with SessionLocal() as session:
    weight_topic_id = session.execute(
        select(BaseTopics.id).where(BaseTopics.text == WEIGHTED_TOPIC_TEXT)
    ).scalar_one()

    base_topics = session.execute(
        select(BaseTopics.id, BaseTopics.text, BaseTopics.short_name).where(
            BaseTopics.id != weight_topic_id
        )
    ).all()

dfs = []
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = [
        ex.submit(
            top_n_for_base_topic,
            topic_id,
            topic_text,
            topic_short_name,
            weight_topic_id,
        )
        for topic_id, topic_text, topic_short_name in base_topics
    ]
    for fut in as_completed(futures):
        dfs.append(fut.result())

out = (
    pd.concat(dfs, ignore_index=True)
    if dfs
    else pd.DataFrame(
        columns=[
            "base_topic",
            "weighted_topic",
            "title",
            "abstract",
            "sim",
            "weight_sim",
        ]
    )
)

out.to_csv("top_weighted_topics_all_base_topics.csv", index=False)
