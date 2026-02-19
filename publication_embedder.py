"""
Backfill Publication.abstract_embedding in a SQLite database with sentence-transformer embeddings.

This script loads Publication rows from the configured SQLite database and generates
embeddings for publications that are missing embeddings (NULL). It embeds a combined
"title: abstract" payload into Publication.abstract_embedding.

Notes:
- Embeddings are generated with a cosine-tuned model and stored as float32 bytes.
- Rows with NULL/empty abstracts are skipped.
- The script is safe to re-run: it only updates rows missing embeddings.

Progress:
- Uses tqdm to show overall progress toward the number of rows currently missing embeddings.

Parallelism:
- Do not parallelize model.encode() across processes/threads when using a single GPU.
  A single encode loop with an appropriate batch size is typically fastest.
  If anything, increase batch size until you approach VRAM limits.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from models import Base, Publication


DB_PATH = "2025_11_09_researchgate.sqlite"
EMBEDDING_MODEL = "msmarco-MiniLM-L6-cos-v5"
BATCH_SIZE = 64
COMMIT_EVERY = 2048


def _iter_publications_to_embed(
    session: Session, batch_size: int
) -> list[tuple[int, str]]:
    """Fetch a batch of publications that need embeddings.

    A publication is selected if:
    - title is not NULL,
    - abstract is not NULL and not empty after stripping,
    - embedding is NULL.

    Args:
        session: SQLAlchemy session.
        batch_size: Maximum number of rows to return.

    Returns:
        A list of (publication_id, payload_text) pairs where payload_text is "title: abstract".
    """
    rows = session.execute(
        select(Publication.id, Publication.title, Publication.abstract)
        .where(Publication.title.is_not(None))
        .where(Publication.abstract.is_not(None))
        .where(Publication.abstract_embedding.is_(None))
        .order_by(Publication.id)
        .limit(batch_size)
    ).all()

    out = []
    for pub_id, title, abstract in rows:
        a = (abstract or "").strip()
        t = (title or "").strip()
        if not a:
            continue
        out.append((pub_id, f"{t}: {a}" if t else a))
    return out


def _encode_texts(
    model: SentenceTransformer, texts: list[str], batch_size: int
) -> np.ndarray:
    """Encode texts into normalized float32 embeddings.

    Args:
        model: SentenceTransformer model.
        texts: List of input texts.
        batch_size: Batch size for model encoding.

    Returns:
        A (len(texts), dim) float32 numpy array of L2-normalized embeddings.
    """
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
    ).astype(np.float32)


def _count_missing_embeddings(session: Session) -> int:
    """Count publications that are eligible for embedding but currently missing embeddings.

    Args:
        session: SQLAlchemy session.

    Returns:
        Number of publications with non-NULL title/abstract and NULL embedding.
    """
    return session.execute(
        select(func.count())
        .select_from(Publication)
        .where(Publication.title.is_not(None))
        .where(Publication.abstract.is_not(None))
        .where(Publication.abstract_embedding.is_(None))
    ).scalar_one()


def main() -> None:
    """Run the embedding backfill process for Publication.abstract_embedding."""
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    Base.metadata.create_all(engine)

    model = SentenceTransformer(EMBEDDING_MODEL)

    with Session(engine) as session:
        total = _count_missing_embeddings(session)
        pbar = tqdm(total=total, unit="pub", desc="Embedding publications")

        updated = 0
        while True:
            batch = _iter_publications_to_embed(session, BATCH_SIZE)
            if not batch:
                break

            ids = [pub_id for pub_id, _ in batch]
            texts = [payload for _, payload in batch]

            vecs = _encode_texts(model, texts, batch_size=BATCH_SIZE)

            session.bulk_update_mappings(
                Publication,
                [
                    {"id": pub_id, "abstract_embedding": v.tobytes()}
                    for pub_id, v in zip(ids, vecs, strict=True)
                ],
            )

            n = len(ids)
            updated += n
            pbar.update(n)

            if updated % COMMIT_EVERY == 0:
                session.commit()

        session.commit()
        pbar.close()


if __name__ == "__main__":
    main()
