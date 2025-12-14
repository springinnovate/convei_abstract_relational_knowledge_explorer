import argparse
import logging

import numpy as np
from sqlalchemy import event, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from models import Publication

logging.basicConfig(
    level=logging.INFO,
    format="[%(filename)s:%(lineno)d] %(levelname)s %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", required=True)
    p.add_argument("--max-train", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    log = logging.getLogger(__name__)

    args = parse_args()

    engine = create_engine(
        f"sqlite:///{args.db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    Session = sessionmaker(bind=engine, future=True, expire_on_commit=False)

    with Session() as session:
        log.info("building training set")
        train_stmt = (
            select(
                Publication.id,
                Publication.abstract_embedding,
                Publication.satellite_type,
            )
            .where(
                Publication.abstract_embedding.is_not(None),
                Publication.satellite_type.is_not(None),
            )
            .order_by(Publication.id)
            .limit(args.max_train)
        )
        rows = session.execute(train_stmt).all()

        n_train = len(rows)
        log.info("training rows: %d", n_train)

        label_set: set[str] = set()
        for _, _, sat_type in rows:
            if not sat_type:
                continue
            for lbl in sat_type.split(","):
                lbl = lbl.strip()
                if lbl:
                    label_set.add(lbl)

        labels = sorted(label_set)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        log.info("distinct labels: %d", len(labels))

        first_emb = rows[0][1]
        embedding_dims = len(first_emb) // 4
        X = np.empty((n_train, embedding_dims), dtype=np.float32)
        Y = np.zeros((n_train, len(labels)), dtype=np.int8)

        for i, (_, emb_bytes, sat_type) in enumerate(rows):
            X[i] = np.frombuffer(emb_bytes, dtype=np.float32)
            if sat_type:
                for lbl in sat_type.split(","):
                    lbl = lbl.strip()
                    if not lbl:
                        continue
                    j = label_to_idx.get(lbl)
                    if j is not None:
                        Y[i, j] = 1

        log.info("starting classifier training")
        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
            )
        )
        clf.fit(X, Y)
        log.info("finished classifier training")

        batch_size = args.batch_size
        unlabeled_filter = (
            Publication.abstract_embedding.is_not(None),
            Publication.satellite_type.is_(None),
        )

        unlabeled_ids_stmt = select(Publication.id).where(*unlabeled_filter)
        total_unlabeled = session.execute(
            select(func.count()).select_from(unlabeled_ids_stmt.subquery())
        ).scalar_one()
        log.info("unlabeled rows for inference: %d", total_unlabeled)

        unlabeled_stmt = (
            select(Publication)
            .where(*unlabeled_filter)
            .order_by(Publication.id)
        )

        offset = 0
        with tqdm(total=total_unlabeled) as pbar:
            while True:
                batch = (
                    session.execute(
                        unlabeled_stmt.limit(batch_size).offset(offset)
                    )
                    .scalars()
                    .all()
                )
                if not batch:
                    break

                B = len(batch)
                Xb = np.empty((B, embedding_dims), dtype=np.float32)
                for i, pub in enumerate(batch):
                    Xb[i] = np.frombuffer(
                        pub.abstract_embedding, dtype=np.float32
                    )

                Pb = clf.predict_proba(Xb)

                updated = 0
                for i, pub in enumerate(batch):
                    p = Pb[i]
                    max_p = float(p.max())
                    if max_p < 0.5:
                        continue
                    if max_p < 0.85:
                        continue
                    chosen = [labels[j] for j, pj in enumerate(p) if pj >= 0.5]
                    if not chosen:
                        continue
                    pub.satellite_type = ",".join(sorted(set(chosen)))
                    updated += 1

                session.commit()
                offset += batch_size
                pbar.update(B)
                log.info("processed batch size=%d updated=%d", B, updated)


if __name__ == "__main__":
    main()
