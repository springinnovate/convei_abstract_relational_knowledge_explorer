from datetime import datetime
import csv
import logging

from tqdm import tqdm
from sqlalchemy import create_engine, event, select, func, and_
from sqlalchemy.orm import Session, aliased

from models import (
    Base,
    Satellite,
    PublicationToSatellite,
)


def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


db_path = "2025_11_09_researchgate.sqlite"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

logging.info("creating engine")
engine = create_engine(
    f"sqlite:///{db_path}",
    future=True,
    connect_args={"timeout": 60},
)
event.listen(engine, "connect", _set_sqlite_pragma)
Base.metadata.create_all(engine)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
output_path = f"satellite_pair_counts_{timestamp}.csv"

pts1 = aliased(PublicationToSatellite)
pts2 = aliased(PublicationToSatellite)

with Session(engine) as session:
    logging.info("loading satellites")
    satellites = session.execute(
        select(Satellite.id, Satellite.name).order_by(Satellite.name)
    ).all()
    logging.info("loaded %d satellites", len(satellites))

    logging.info("querying co-occurrence counts")
    count_rows = session.execute(
        select(
            pts1.satellite_id,
            pts2.satellite_id,
            func.count(func.distinct(pts1.publication_id)),
        )
        .join(
            pts2,
            and_(
                pts1.publication_id == pts2.publication_id,
                pts1.satellite_id <= pts2.satellite_id,
            ),
        )
        .group_by(pts1.satellite_id, pts2.satellite_id)
    )

    counts = {}
    for row in tqdm(count_rows, desc="Loading pair counts"):
        counts[(row[0], row[1])] = row[2]
    logging.info("loaded %d satellite pairs with co-occurrences", len(counts))

    logging.info("writing csv to %s", output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["satellite"] + [satellite.name for satellite in satellites]
        )

        for i, satellite_row in enumerate(
            tqdm(satellites, desc="Writing matrix rows")
        ):
            row = [satellite_row.name]
            for j, satellite_col in enumerate(satellites):
                if j < i:
                    row.append("")
                else:
                    row.append(
                        counts.get((satellite_row.id, satellite_col.id), 0)
                    )
            writer.writerow(row)

logging.info("wrote %s", output_path)
