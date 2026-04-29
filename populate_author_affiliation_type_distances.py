from __future__ import annotations

import argparse

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session
from tqdm import tqdm

from models import (
    AffiliationType,
    Base,
    PublicationAuthorLocation,
    PublicationAuthorLocationAffiliationTypeDistance,
)


EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_INSERT_BATCH_SIZE = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="2025_11_09_researchgate.sqlite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE
    )
    parser.add_argument(
        "--insert-batch-size", type=int, default=DEFAULT_INSERT_BATCH_SIZE
    )
    return parser.parse_args()


def decode_embedding(embedding_bytes: bytes) -> np.ndarray:
    return np.frombuffer(embedding_bytes, dtype=np.float32).astype(np.float64)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def chunked(sequence: list, size: int):
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def load_affiliation_types(session: Session) -> tuple[list[int], np.ndarray]:
    rows = session.execute(
        select(AffiliationType.id, AffiliationType.embedding)
        .where(AffiliationType.embedding.is_not(None))
        .order_by(AffiliationType.id)
    ).all()
    if not rows:
        raise RuntimeError("No affiliation type embeddings found in the database.")

    affiliation_type_ids = [row[0] for row in rows]
    embedding_matrix = np.vstack(
        [decode_embedding(row[1]) for row in rows]
    ).astype(np.float64)
    return affiliation_type_ids, normalize_rows(embedding_matrix)


def load_pending_author_locations(
    session: Session, limit: int | None
) -> list[tuple[int, str]]:
    query = (
        select(
            PublicationAuthorLocation.id,
            PublicationAuthorLocation.cleaned_affiliation_text,
        )
        .outerjoin(
            PublicationAuthorLocationAffiliationTypeDistance,
            PublicationAuthorLocationAffiliationTypeDistance.publication_author_location_id
            == PublicationAuthorLocation.id,
        )
        .where(
            PublicationAuthorLocationAffiliationTypeDistance.publication_author_location_id.is_(
                None
            )
        )
        .order_by(PublicationAuthorLocation.id)
    )
    if limit is not None:
        query = query.limit(limit)
    return session.execute(query).all()


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.vstack(
        [np.array(item.embedding, dtype=np.float64) for item in response.data]
    )
    return normalize_rows(vectors)


def build_distance_rows(
    author_location_id_groups: list[list[int]],
    embeddings: np.ndarray,
    affiliation_type_ids: list[int],
    affiliation_type_matrix: np.ndarray,
    top_k: int | None,
) -> list[dict]:
    similarity_matrix = embeddings @ affiliation_type_matrix.T

    rows: list[dict] = []
    for row_index, author_location_ids in enumerate(author_location_id_groups):
        similarity_row = similarity_matrix[row_index]
        if top_k is None or top_k >= len(affiliation_type_ids):
            selected_indexes = range(len(affiliation_type_ids))
        else:
            selected_indexes = np.argsort(similarity_row)[::-1][:top_k]

        for author_location_id in author_location_ids:
            for type_index in selected_indexes:
                rows.append(
                    {
                        "publication_author_location_id": author_location_id,
                        "affiliation_type_id": affiliation_type_ids[type_index],
                        "semantic_similarity": float(similarity_row[type_index]),
                    }
                )
    return rows


def main() -> None:
    load_dotenv()
    args = parse_args()

    engine = create_engine(f"sqlite:///{args.db}", future=True)
    Base.metadata.create_all(engine)
    client = OpenAI()

    with Session(engine) as session:
        affiliation_type_ids, affiliation_type_matrix = load_affiliation_types(
            session
        )
        pending_rows = load_pending_author_locations(session, args.limit)

    total_author_locations = len(pending_rows)
    if total_author_locations == 0:
        print("author_locations_processed=0")
        print("unique_affiliation_texts_processed=0")
        print("distance_rows_upserted=0")
        return

    total_distance_rows = 0
    total_unique_texts = 0

    with Session(engine) as session:
        for batch in tqdm(
            chunked(pending_rows, args.embed_batch_size),
            total=(total_author_locations + args.embed_batch_size - 1)
            // args.embed_batch_size,
            desc="Embedding author affiliations",
        ):
            affiliation_text_to_ids: dict[str, list[int]] = {}
            for author_location_id, affiliation_text in batch:
                affiliation_text_to_ids.setdefault(affiliation_text, []).append(
                    author_location_id
                )

            texts = list(affiliation_text_to_ids.keys())
            author_location_id_groups = [
                affiliation_text_to_ids[text] for text in texts
            ]
            total_unique_texts += len(texts)

            embeddings = embed_texts(client, texts)
            distance_rows = build_distance_rows(
                author_location_id_groups,
                embeddings,
                affiliation_type_ids,
                affiliation_type_matrix,
                args.top_k,
            )

            for insert_batch in chunked(distance_rows, args.insert_batch_size):
                insert_stmt = sqlite_insert(
                    PublicationAuthorLocationAffiliationTypeDistance
                ).values(insert_batch)
                insert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=[
                        "publication_author_location_id",
                        "affiliation_type_id",
                    ],
                    set_={
                        "semantic_similarity": insert_stmt.excluded.semantic_similarity
                    },
                )
                result = session.execute(insert_stmt)
                session.commit()
                total_distance_rows += result.rowcount or 0

    print(f"author_locations_processed={total_author_locations}")
    print(f"unique_affiliation_texts_processed={total_unique_texts}")
    print(f"affiliation_types_per_location={len(affiliation_type_ids)}")
    print(f"distance_rows_upserted={total_distance_rows}")


if __name__ == "__main__":
    main()
