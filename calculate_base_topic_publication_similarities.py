from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from models import Base, BaseTopicToPublicationDistance, BaseTopics, Publication

DB_PATH = "2025_11_09_researchgate.sqlite"
PUB_BATCH = 256
COMMIT_EVERY = 65536
N_WORKERS = os.cpu_count() or 4

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    connect_args={"timeout": 60},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=60000")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.close()


_BT_IDS: np.ndarray | None = None
_BT_MAT: np.ndarray | None = None


def _load_base_topics(engine):
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
    return session.execute(
        select(func.count())
        .select_from(Publication)
        .where(Publication.abstract_embedding.is_not(None))
    ).scalar_one()


def _fetch_pub_batch(session: Session, after_pub: int, batch_size: int):
    return session.execute(
        select(Publication.id, Publication.abstract_embedding)
        .where(Publication.abstract_embedding.is_not(None))
        .where(Publication.id > after_pub)
        .order_by(Publication.id)
        .limit(batch_size)
    ).all()


def _compute(pub_id: int, pub_emb: bytes):
    pub_vec = np.frombuffer(pub_emb, dtype=np.float32).astype(np.float32, copy=False)
    sim = _BT_MAT @ pub_vec
    return pub_id, sim


def main() -> None:
    global _BT_IDS, _BT_MAT

    Base.metadata.create_all(engine)
    _BT_IDS, _BT_MAT = _load_base_topics(engine)

    table = BaseTopicToPublicationDistance.__tablename__
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"CREATE UNIQUE INDEX IF NOT EXISTS uq_bt_pub ON {table} (base_topic_id, publication_id)"
        )

    insert_sql = (
        f"INSERT OR IGNORE INTO {table} "
        f"(base_topic_id, publication_id, semantic_similarity) "
        f"VALUES (?, ?, ?)"
    )

    raw = engine.raw_connection()
    cur = raw.cursor()

    with Session(engine) as session:
        total_pubs = _count_pubs_with_embedding(session)
        pbar = tqdm(
            total=total_pubs,
            unit="pub",
            desc="Computing base_topic to publication distances",
        )

        after_pub = 0
        attempted = 0
        next_commit = COMMIT_EVERY

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            while True:
                pubs = _fetch_pub_batch(session, after_pub, PUB_BATCH)
                if not pubs:
                    break

                after_pub = pubs[-1][0]
                futures = [
                    ex.submit(_compute, pub_id, pub_emb) for pub_id, pub_emb in pubs
                ]

                for fut in as_completed(futures):
                    pub_id, sim = fut.result()
                    cur.executemany(
                        insert_sql,
                        (
                            (int(bt_id), int(pub_id), float(s))
                            for bt_id, s in zip(_BT_IDS, sim)
                        ),
                    )

                    attempted += _BT_IDS.size
                    if attempted >= next_commit:
                        raw.commit()
                        next_commit += COMMIT_EVERY

                    pbar.update(1)

        raw.commit()
        pbar.close()

    cur.close()
    raw.close()


if __name__ == "__main__":
    main()
