from dotenv import load_dotenv

load_dotenv()

import array

from openai import OpenAI
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import AffiliationType, Base

DB_PATH = "2025_11_09_researchgate.sqlite"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 128

affiliation_types = [
    ("academic", "Academic (universities, colleges)"),
    ("government", "Government (ministries, agencies, national labs)"),
    ("private", "Private (for-profit)"),
    ("nonprofit", "Nonprofit/NGO"),
    ("intergovernmental", "Intergovernmental/Multilateral"),
]


def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main():
    client = OpenAI()
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    Base.metadata.create_all(engine)

    rows = affiliation_types

    with Session(engine) as session:
        existing_texts = set(
            session.scalars(select(AffiliationType.text)).all()
        )
        missing_emb_texts = set(
            session.scalars(
                select(AffiliationType.text).where(
                    AffiliationType.embedding.is_(None)
                )
            ).all()
        )

        to_insert = [(sn, tx) for sn, tx in rows if tx not in existing_texts]
        to_update = [(sn, tx) for sn, tx in rows if tx in missing_emb_texts]
        to_add = to_insert + to_update
        if not to_add:
            return

        texts = [tx for _, tx in to_add]
        embeddings = []
        for batch in chunked(texts, BATCH_SIZE):
            r = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend([d.embedding for d in r.data])

        for (sn, tx), v in zip(to_add, embeddings, strict=True):
            b = array.array("f", v).tobytes()

            if tx in existing_texts:
                session.query(AffiliationType).filter(
                    AffiliationType.text == tx
                ).update(
                    {
                        AffiliationType.embedding: b,
                        AffiliationType.short_name: sn,
                    }
                )
            else:
                session.add(
                    AffiliationType(short_name=sn, text=tx, embedding=b)
                )

        session.commit()


if __name__ == "__main__":
    main()
