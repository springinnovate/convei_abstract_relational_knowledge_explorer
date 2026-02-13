"""
Populate the benefit_dimensions table with topic phrases and embeddings.

This script reads topic phrases from a newline-delimited text file,
inserts them into the benefit_dimensions table if they do not already
exist, and generates sentence-transformer embeddings for each topic.
If a topic already exists but has a NULL embedding, the embedding is
computed and backfilled.

Embeddings are generated using a cosine-tuned MS MARCO MiniLM model
and stored as float32 byte arrays in a SQLite database.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import Integer, Text, LargeBinary, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from models import Base, BenefitDimension

DB_PATH = "2025_11_09_researchgate.sqlite"
TOPICS_PATH = "topic_areas.txt"
EMBEDDING_MODEL = "msmarco-MiniLM-L6-cos-v5"
BATCH_SIZE = 64


def main():
    """Execute embedding generation and database population workflow.

    This function:
    1. Creates the database schema if it does not exist.
    2. Loads topic phrases from the configured file.
    3. Identifies topics that are new or missing embeddings.
    4. Generates normalized embeddings in batches.
    5. Inserts new records or updates missing embeddings.
    """
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    Base.metadata.create_all(engine)

    model = SentenceTransformer(EMBEDDING_MODEL)
    lines = Path(TOPICS_PATH).read_text(encoding="utf-8").splitlines()
    topics = [line.strip() for line in lines]

    with Session(engine) as session:
        existing_all = set(session.scalars(select(BenefitDimension.text)).all())
        missing_emb = set(
            session.scalars(
                select(BenefitDimension.text).where(
                    BenefitDimension.embedding.is_(None)
                )
            ).all()
        )

        to_insert = [t for t in topics if t not in existing_all]
        to_update = [t for t in topics if t in missing_emb]
        to_add = to_insert + to_update

        if not to_add:
            return

        vecs = model.encode(
            to_add,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
        ).astype(np.float32)

        for t, v in zip(to_add, vecs, strict=True):
            if t in existing_all:
                session.query(BenefitDimension).filter(
                    BenefitDimension.text == t
                ).update({BenefitDimension.embedding: v.tobytes()})
            else:
                session.add(BenefitDimension(text=t, embedding=v.tobytes()))

        session.commit()


if __name__ == "__main__":
    main()
