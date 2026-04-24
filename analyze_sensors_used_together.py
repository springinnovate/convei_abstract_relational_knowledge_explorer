from datetime import datetime
import csv
import logging

from tqdm import tqdm
from sqlalchemy import create_engine, event, select, func, and_
from sqlalchemy.orm import Session, aliased

from models import (
    Base,
    DataType,
    PublicationToDataType,
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
output_path = f"data_type_pair_counts_{timestamp}.csv"

ptd1 = aliased(PublicationToDataType)
ptd2 = aliased(PublicationToDataType)

with Session(engine) as session:
    logging.info("loading data types")
    data_types = session.execute(
        select(DataType.id, DataType.name).order_by(DataType.name)
    ).all()
    logging.info("loaded %d data types", len(data_types))

    logging.info("querying co-occurrence counts")
    count_rows = session.execute(
        select(
            ptd1.data_type_id,
            ptd2.data_type_id,
            func.count(func.distinct(ptd1.publication_id)),
        )
        .join(
            ptd2,
            and_(
                ptd1.publication_id == ptd2.publication_id,
                ptd1.data_type_id <= ptd2.data_type_id,
            ),
        )
        .group_by(ptd1.data_type_id, ptd2.data_type_id)
    )

    counts = {}
    for row in tqdm(count_rows, desc="Loading pair counts"):
        counts[(row[0], row[1])] = row[2]
    logging.info("loaded %d data type pairs with co-occurrences", len(counts))

    logging.info("writing csv to %s", output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data_type"] + [name for _, name in data_types])

        for i, (row_id, row_name) in enumerate(
            tqdm(data_types, desc="Writing matrix rows")
        ):
            row = [row_name]
            for j, (col_id, _) in enumerate(data_types):
                if j < i:
                    row.append("")
                else:
                    row.append(counts.get((row_id, col_id), 0))
            writer.writerow(row)

logging.info("wrote %s", output_path)
