from math import ceil

from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session
from tqdm.auto import tqdm

from models import (
    Base,
    DataType,
    Publication,
    PublicationToDataType,
    PublicationToSatellite,
    Satellite,
)


def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


db_path = "2025_11_09_researchgate.sqlite"

engine = create_engine(
    f"sqlite:///{db_path}",
    future=True,
    connect_args={"timeout": 60},
)
event.listen(engine, "connect", _set_sqlite_pragma)

Base.metadata.create_all(engine)

insert_chunk_size = 10000

with Session(engine) as session:
    print("Loading publications...")
    publications = session.execute(
        select(Publication.id, Publication.abstract)
    ).all()
    publications = [
        (pub_id, abstract.lower())
        for pub_id, abstract in tqdm(
            publications, desc="Normalize publications"
        )
        if abstract
    ]
    print(f"Loaded {len(publications)} publications with abstracts")

    print("Loading satellites...")
    satellites = session.execute(select(Satellite.id, Satellite.name)).all()
    satellites = [
        (sat_id, name.lower())
        for sat_id, name in tqdm(satellites, desc="Normalize satellites")
    ]
    print(f"Loaded {len(satellites)} satellites")

    print("Loading data types...")
    data_types = session.execute(select(DataType.id, DataType.name)).all()
    data_types = [
        (data_type_id, name.lower())
        for data_type_id, name in tqdm(data_types, desc="Normalize data types")
    ]
    print(f"Loaded {len(data_types)} data types")

    satellite_matches = set()
    data_type_matches = set()

    print("Scanning abstracts...")
    for pub_id, abstract in tqdm(publications, desc="Match terms", unit="pub"):
        for sat_id, sat_name in satellites:
            if sat_name in abstract:
                satellite_matches.add((pub_id, sat_id))

        for data_type_id, data_type_name in data_types:
            if data_type_name in abstract:
                data_type_matches.add((pub_id, data_type_id))

    print(f"Satellite matches found: {len(satellite_matches)}")
    print(f"Data type matches found: {len(data_type_matches)}")

    satellite_rows = [
        {"publication_id": pub_id, "satellite_id": sat_id}
        for pub_id, sat_id in satellite_matches
    ]
    data_type_rows = [
        {"publication_id": pub_id, "data_type_id": data_type_id}
        for pub_id, data_type_id in data_type_matches
    ]

    if satellite_rows:
        satellite_stmt = insert(PublicationToSatellite).on_conflict_do_nothing(
            index_elements=["publication_id", "satellite_id"]
        )
        total_sat_chunks = ceil(len(satellite_rows) / insert_chunk_size)
        for chunk in tqdm(
            chunked(satellite_rows, insert_chunk_size),
            total=total_sat_chunks,
            desc="Insert satellite refs",
            unit="chunk",
        ):
            session.execute(satellite_stmt, chunk)
        session.commit()

    if data_type_rows:
        data_type_stmt = insert(PublicationToDataType).on_conflict_do_nothing(
            index_elements=["publication_id", "data_type_id"]
        )
        total_data_chunks = ceil(len(data_type_rows) / insert_chunk_size)
        for chunk in tqdm(
            chunked(data_type_rows, insert_chunk_size),
            total=total_data_chunks,
            desc="Insert data type refs",
            unit="chunk",
        ):
            session.execute(data_type_stmt, chunk)
        session.commit()

    print("Done")
