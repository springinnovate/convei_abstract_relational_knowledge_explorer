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
def _set_sqlite_pragma(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=60000")
    cur.close()


def _load_base_topics(session: Session):
    rows = session.execute(
        select(BaseTopics.id, BaseTopics.text).order_by(BaseTopics.id)
    ).all()
    bt_ids = np.array([r[0] for r in rows], dtype=np.int64)
    bt_text = {bt_id: text for bt_id, text in rows}
    # make a mapping here just in case the base topic ids aren't 0...n-1
    bt_index = {bt_id: i for i, bt_id in enumerate(bt_ids)}
    return bt_ids, bt_text, bt_index


def year_vector(
    session: Session,
    year: int,
):
    bt_ids_all, bt_text, bt_index = _load_base_topics(session)
    vec = np.zeros(len(bt_ids_all), dtype=np.float64)

    q = (
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

    q = q.order_by(
        BaseTopicToPublicationDistance.publication_id,
        BaseTopicToPublicationDistance.base_topic_id,
    )

    rows = session.execute(q)

    temperature = 0.3

    cur_pub = None
    pub_bt_ids: list[int] = []
    pub_scores: list[float] = []

    def flush():
        if not pub_scores:
            return
        s = np.array(pub_scores, dtype=np.float64)
        s = np.maximum(s, 0.0)
        s = s / temperature
        m = float(s.max())
        exps = np.exp(s - m)
        denom = float(exps.sum())
        if denom == 0.0:
            return
        exps = exps / denom
        for bt_id, w in zip(pub_bt_ids, exps, strict=True):
            vec[bt_index[bt_id]] += float(w)

    for pub_id, bt_id, s in rows:
        if cur_pub is None:
            cur_pub = pub_id
        if pub_id != cur_pub:
            flush()
            cur_pub = pub_id
            pub_bt_ids = []
            pub_scores = []
        pub_bt_ids.append(bt_id)
        pub_scores.append(s)

    flush()
    return bt_ids_all, bt_text, vec


def topk(bt_ids, bt_text, vec, k: int):
    idx = np.argsort(vec)[::-1]
    out = []
    for i in idx[:k]:
        bt_id = int(bt_ids[i])
        out.append((bt_id, bt_text[bt_id], float(vec[i])))
    return out


def write_csv(path: str, bt_ids, bt_text, vec):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base_topic_id", "base_topic_text", "sum_semantic_similarity"])
        for bt_id, v in zip(bt_ids, vec, strict=True):
            w.writerow([int(bt_id), bt_text[int(bt_id)], float(v)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--published-in-type", default=None)
    ap.add_argument("--satellite-type", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    engine = create_engine(
        f"sqlite:///{args.db}",
        future=True,
        connect_args={"timeout": 60},
    )

    with Session(engine) as session:
        years = list(range(args.start_year, args.end_year + 1))

        bt_ids, bt_text, _ = year_vector(
            session,
            year=years[0],
            published_in_type=args.published_in_type,
            satellite_type=args.satellite_type,
        )

        mat = np.zeros((len(bt_ids), len(years)), dtype=np.float64)

        for j, y in enumerate(years):
            bt_ids_y, bt_text_y, vec = year_vector(
                session,
                year=y,
                published_in_type=args.published_in_type,
                satellite_type=args.satellite_type,
            )
            mat[:, j] = vec

        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["base_topic_id", "base_topic_text", *years])
            for i, bt_id in enumerate(bt_ids):
                w.writerow([int(bt_id), bt_text[int(bt_id)], *mat[i, :].tolist()])

        for j, y in enumerate(years):
            vec = mat[:, j]
            print(f"year={y} topics={len(vec)}")
            for bt_id, v in zip(bt_ids, vec, strict=True):
                text = bt_text[bt_id]
                print(f"{bt_id}\t{v:.6f}\t{text}")


if __name__ == "__main__":
    main()
