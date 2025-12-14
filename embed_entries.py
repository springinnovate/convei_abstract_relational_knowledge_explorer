from array import array
import time
import argparse
import logging
import queue
import concurrent.futures
import threading
import random

from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
from sqlalchemy import create_engine, select, event, bindparam, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from tqdm import tqdm

from models import Publication

load_dotenv()
CLIENT = OpenAI(timeout=600.0)
MAX_EMBED_TOKENS = 8000
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 1000


def embed_text(text, client, max_retries):
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    tokens = enc.encode(text)
    if len(tokens) > MAX_EMBED_TOKENS:
        text = enc.decode(tokens[:MAX_EMBED_TOKENS])

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            vec = resp.data[0].embedding
            return array("f", vec).tobytes()
        except Exception as exc:
            logging.error(
                "OpenAI embedding call failed, attempt %s/%s: %s",
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
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
        ids = session.scalars(
            select(Publication.id).where(
                Publication.abstract.is_not(None),
                Publication.abstract != "",
                Publication.abstract_embedding.is_(None),
            )
        ).all()

    write_q: queue.Queue = queue.Queue()

    stmt = (
        update(Publication)
        .where(Publication.id == bindparam("id"))
        .where(Publication.abstract.is_not(None))
        .where(Publication.abstract_embedding.is_(None))
        .values(abstract_embedding=bindparam("emb"))
    )

    def _flush(pending: list[tuple[int, object]]) -> None:
        rows = [{"id": pub_id, "emb": emb} for pub_id, emb in pending]
        for attempt in range(8):
            try:
                with Session.begin() as session:
                    session.execute(stmt, rows)
                return
            except OperationalError as e:
                if "database is locked" not in str(e).lower():
                    raise
                time.sleep(
                    min(0.5, 0.02 * (2**attempt) + random.random() * 0.01)
                )
        with Session.begin() as session:
            session.execute(stmt, rows)

    def writer() -> None:
        pending: list[tuple[int, object]] = []
        while True:
            item = write_q.get()
            if item is None:
                break
            pending.append(item)

            try:
                while len(pending) < BATCH_SIZE:
                    item2 = write_q.get_nowait()
                    if item2 is None:
                        item = None
                        break
                    pending.append(item2)
            except queue.Empty:
                item = "empty"

            if pending and (
                len(pending) >= BATCH_SIZE or item == "empty" or item is None
            ):
                _flush(pending)
                pending.clear()

        if pending:
            _flush(pending)

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    def worker(pub_id: int) -> None:
        client = OpenAI()
        with Session() as session:
            pub = session.get(Publication, pub_id)
            if not pub or not pub.abstract or pub.abstract_embedding:
                return
            abstract = pub.abstract
        emb = embed_text(abstract, client, args.max_retries)
        write_q.put((pub_id, emb))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as exe:
        list(tqdm(exe.map(worker, ids), total=len(ids)))

    write_q.put(None)
    wt.join()


if __name__ == "__main__":
    main()
