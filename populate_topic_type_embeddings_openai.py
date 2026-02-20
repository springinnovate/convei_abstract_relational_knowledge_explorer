from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import BaseTopics, Base

DB_PATH = "2025_11_09_researchgate.sqlite"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 128

topics = [
    "Academic (universities, colleges)",
    "Government (ministries, agencies, national labs)",
    "Private (for-profit)",
    "Nonprofit/NGO",
    "Intergovernmental/Multilateral",
]


def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main():
    client = OpenAI()
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    Base.metadata.create_all(engine)

    rows = [(t.split(" ", 1)[0].rstrip(")").lstrip("("), t) for t in topics]

    with Session(engine) as session:
        existing_texts = set(session.scalars(select(BaseTopics.text)).all())
        missing_emb_texts = set(
            session.scalars(
                select(BaseTopics.text).where(BaseTopics.embedding.is_(None))
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
            b = memoryview(bytearray()).tobytes()

            import array

            a = array.array("f", v)
            b = a.tobytes()

            if tx in existing_texts:
                session.query(BaseTopics).filter(BaseTopics.text == tx).update(
                    {BaseTopics.embedding: b, BaseTopics.short_name: sn}
                )
            else:
                session.add(BaseTopics(short_name=sn, text=tx, embedding=b))

        session.commit()


if __name__ == "__main__":
    main()
