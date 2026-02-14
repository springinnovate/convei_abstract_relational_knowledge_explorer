"""
Compute semantic similarity scores between all BaseTopics embeddings and all
Publication abstract embeddings, storing the results in the
BaseTopicToPublicationDistance table.

This script performs a cross product between base topics and publications with
non-null embeddings. For each publication, it computes the dot product between
its abstract embedding and every base topic embedding, skipping pairs that
already exist in the distance table.

Processing is performed in publication batches to control memory usage.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from models import Base, BaseTopicToPublicationDistance, BaseTopics, Publication

DB_PATH = "2025_11_09_researchgate.sqlite"
PUB_BATCH = 256
COMMIT_EVERY = 65536

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    connect_args={"timeout": 60},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _):
    """Configure SQLite PRAGMA settings on connection.

    Enables WAL mode, reduces fsync strictness for performance, and increases
    busy timeout to reduce locking errors during bulk inserts.

    Args:
        dbapi_conn: The raw DB-API connection object.
        _: Unused SQLAlchemy connection record parameter.
    """
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=60000")
    cur.close()


def _load_base_topics(engine):
    """Load all base topic embeddings into memory.

    Args:
        engine: SQLAlchemy engine bound to the target database.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Array of base topic IDs (int64).
            - 2D float32 matrix of base topic embeddings
              with shape (num_topics, embedding_dim).
    """
    with Session(engine) as session:
        rows = session.execute(
            select(BaseTopics.id, BaseTopics.embedding).where(
                BaseTopics.embedding.is_not(None)
            )
        ).all()

    bt_ids = np.array([r[0] for r in rows], dtype=np.int64)
    bt_mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).astype(
        np.float32, copy=False
    )
    return bt_ids, bt_mat


def _count_pubs_with_embedding(session: Session) -> int:
    """Count publications that have a non-null abstract embedding.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        int: Number of publications with abstract embeddings.
    """
    return session.execute(
        select(func.count())
        .select_from(Publication)
        .where(Publication.abstract_embedding.is_not(None))
    ).scalar_one()


def _fetch_pub_batch(session: Session, after_pub: int, batch_size: int):
    """Fetch a batch of publications with embeddings.

    Publications are ordered by ID and fetched strictly after the given ID
    cursor to enable forward-only pagination.

    Args:
        session: Active SQLAlchemy session.
        after_pub: Last processed publication ID.
        batch_size: Maximum number of publications to return.

    Returns:
        list[tuple[int, bytes]]: List of (publication_id, abstract_embedding)
        tuples.
    """
    return session.execute(
        select(Publication.id, Publication.abstract_embedding)
        .where(Publication.abstract_embedding.is_not(None))
        .where(Publication.id > after_pub)
        .order_by(Publication.id)
        .limit(batch_size)
    ).all()


def _existing_bt_ids_for_pub(session: Session, pub_id: int) -> set[int]:
    """Retrieve base topic IDs already computed for a publication.

    Args:
        session: Active SQLAlchemy session.
        pub_id: Publication ID.

    Returns:
        set[int]: Set of base_topic_id values already present in the
        BaseTopicToPublicationDistance table for the given publication.
    """
    rows = session.execute(
        select(BaseTopicToPublicationDistance.base_topic_id).where(
            BaseTopicToPublicationDistance.publication_id == pub_id
        )
    ).all()
    return {r[0] for r in rows}


def main() -> None:
    """Execute batch computation of base topic to publication similarities.

    Workflow:
        1. Ensure tables exist.
        2. Load all base topic embeddings into memory.
        3. Iterate through publications in ID order.
        4. For each publication, compute dot products against all base topics.
        5. Insert missing similarity records in bulk.
        6. Periodically commit to reduce transaction size.
    """
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        future=True,
        connect_args={"timeout": 60},
    )
    Base.metadata.create_all(engine)

    bt_ids_all, bt_mat_all = _load_base_topics(engine)

    with Session(engine) as session:
        print("count pubs")
        total_pubs = _count_pubs_with_embedding(session)
        pbar = tqdm(
            total=total_pubs,
            unit="pub",
            desc="Computing base_topic to publication distances",
        )

        after_pub = 0
        inserted = 0
        print("start processing pubs")
        while True:
            pubs = _fetch_pub_batch(session, after_pub, PUB_BATCH)
            if not pubs:
                break

            for pub_id, pub_emb in pubs:
                after_pub = pub_id

                existing = _existing_bt_ids_for_pub(session, pub_id)
                if existing:
                    mask = np.array(
                        [bt_id not in existing for bt_id in bt_ids_all],
                        dtype=bool,
                    )
                    bt_ids = bt_ids_all[mask]
                    bt_mat = bt_mat_all[mask]
                    if bt_ids.size == 0:
                        continue
                else:
                    bt_ids = bt_ids_all
                    bt_mat = bt_mat_all

                pub_vec = np.frombuffer(pub_emb, dtype=np.float32).astype(
                    np.float32, copy=False
                )
                sim = bt_mat @ pub_vec

                items = [
                    {
                        "base_topic_id": int(bt_id),
                        "publication_id": int(pub_id),
                        "semantic_similarity": float(s),
                    }
                    for bt_id, s in zip(bt_ids, sim, strict=True)
                ]

                session.bulk_insert_mappings(BaseTopicToPublicationDistance, items)
                inserted += len(items)

                if inserted % COMMIT_EVERY == 0:
                    session.commit()

            pbar.update(len(pubs))

        session.commit()
        pbar.close()


if __name__ == "__main__":
    main()
