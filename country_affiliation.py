from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from models import Base, Publication, PublicationPrimaryAuthorLocation


def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


db_path = "2025_11_09_researchgate.sqlite"

engine = create_engine(
    f"sqlite:///{db_path}",
    future=True,
    connect_args={"timeout": 60},
)
event.listen(engine, "connect", _set_sqlite_pragma)

Base.metadata.create_all(engine)

with Session(engine) as session:
    author_affiliation_set = (
        session.execute(select(Publication.author_affiliations).limit(10))
        .scalars()
        .all()
    )

    for affiliation_string in author_affiliation_set:
        s = affiliation_string

        if "]" in s:
            s = s.split("]", 1)[1]

        first_affiliation = s.split(";", 1)[0].strip()
        print(first_affiliation)
