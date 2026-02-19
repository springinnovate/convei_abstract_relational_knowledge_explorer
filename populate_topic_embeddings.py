"""
Populate topic embeddings in a SQLite database.

This script reads topic definitions from a semicolon-delimited text file,
generates sentence embeddings using a SentenceTransformer model, and stores
them in a SQLite database. Existing topics without embeddings are updated,
and new topics are inserted.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import BaseTopics, Base

DB_PATH = "2025_11_09_researchgate.sqlite"
TOPICS_PATH = "topic_areas.txt"
EMBEDDING_MODEL = "msmarco-MiniLM-L6-cos-v5"
BATCH_SIZE = 64


def read_topics(path: str) -> list[str]:
    """Read raw topic lines from a text file.

    Args:
        path: Path to a UTF-8 encoded text file containing one topic per line.

    Returns:
        A list of stripped lines from the file.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines]


def main():
    """Generate and store embeddings for topic areas.

    This function:
        1. Connects to the configured SQLite database.
        2. Loads topic definitions from a semicolon-delimited text file.
        3. Determines which topics are new or missing embeddings.
        4. Generates normalized sentence embeddings in batches.
        5. Inserts new topics and updates existing ones with embeddings.
        6. Commits all changes to the database.

    The input file is expected to contain lines in the format:
        short_name; full topic text
    """
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    Base.metadata.create_all(engine)
    model = SentenceTransformer(EMBEDDING_MODEL)
    lines = Path(TOPICS_PATH).read_text(encoding="utf-8").splitlines()

    topics = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        short_name, text = [p.strip() for p in line.split(";", 1)]
        topics.append((short_name, text))

    with Session(engine) as session:
        existing_texts = set(session.scalars(select(BaseTopics.text)).all())
        missing_emb_texts = set(
            session.scalars(
                select(BaseTopics.text).where(BaseTopics.embedding.is_(None))
            ).all()
        )

        to_insert = [(sn, tx) for sn, tx in topics if tx not in existing_texts]
        to_update = [(sn, tx) for sn, tx in topics if tx in missing_emb_texts]
        to_add = to_insert + to_update

        if not to_add:
            return

        texts = [tx for _, tx in to_add]
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
        ).astype(np.float32)

        for (sn, tx), v in zip(to_add, vecs, strict=True):
            if tx in existing_texts:
                session.query(BaseTopics).filter(BaseTopics.text == tx).update(
                    {
                        BaseTopics.embedding: v.tobytes(),
                        BaseTopics.short_name: sn,
                    }
                )
            else:
                session.add(BaseTopics(short_name=sn, text=tx, embedding=v.tobytes()))

        session.commit()


if __name__ == "__main__":
    main()
