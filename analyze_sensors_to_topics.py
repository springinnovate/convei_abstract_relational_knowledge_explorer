from datetime import datetime
import csv
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from affiliation_vector_transform import power_normalize
from models import (
    Base,
    Publication,
    Satellite,
    BaseTopics,
    BaseTopicToPublicationDistance,
)


def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


db_path = "2025_11_09_researchgate.sqlite"
reports_dir = Path("topic_mapping_reports")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

engine = create_engine(
    f"sqlite:///{db_path}",
    future=True,
    connect_args={"timeout": 60},
)
event.listen(engine, "connect", _set_sqlite_pragma)
Base.metadata.create_all(engine)

with Session(engine) as session:
    logging.info("Loading sensors")
    sensor_names = [
        x[0].casefold() for x in session.execute(select(Satellite.name)).all()
    ]

    logging.info("Loading topics")
    topics_id_name = session.execute(
        select(BaseTopics.id, BaseTopics.short_name)
    ).all()
    topic_id_to_index = {
        topic_id: index for index, (topic_id, _) in enumerate(topics_id_name)
    }

    logging.info("Loading publications")
    publications_id_abstract = session.execute(
        select(Publication.id, Publication.abstract)
    ).all()

    logging.info("Loading topic-publication distances")
    topic_pub_distances = session.execute(
        select(
            BaseTopicToPublicationDistance.base_topic_id,
            BaseTopicToPublicationDistance.publication_id,
            BaseTopicToPublicationDistance.semantic_similarity,
        )
    ).all()

    logging.info("Indexing publication abstracts")
    publication_to_abstract = {
        pub_id: abstract.casefold()
        for pub_id, abstract in tqdm(publications_id_abstract, desc="Abstracts")
    }

    logging.info("Matching sensors to publications by direct text search")
    publication_to_sensor_idxs = {}
    for pub_id, abstract in tqdm(
        publication_to_abstract.items(), desc="Sensor matching"
    ):
        publication_to_sensor_idxs[pub_id] = [
            i
            for i, sensor_cf in enumerate(sensor_names)
            if sensor_cf in abstract
        ]

    logging.info("Initializing output matrix")
    matrix = {
        topic_name: {sensor: 0.0 for sensor in sensor_names}
        for _, topic_name in topics_id_name
    }
    topic_names = [topic_name for _, topic_name in topics_id_name]

    logging.info("Building publication-level topic vectors")
    publication_to_topic_scores = {}
    for topic_id, pub_id, semantic_similarity in tqdm(
        topic_pub_distances, desc="Topic vectors"
    ):
        scores = publication_to_topic_scores.setdefault(
            pub_id, np.zeros(len(topics_id_name), dtype=np.float64)
        )
        scores[topic_id_to_index[topic_id]] = float(semantic_similarity)

    logging.info("Accumulating transformed topic-to-sensor closeness")
    for pub_id, topic_scores in tqdm(
        publication_to_topic_scores.items(), desc="Accumulating"
    ):
        sensor_idxs = publication_to_sensor_idxs[pub_id]
        if not sensor_idxs:
            continue
        topic_weights = power_normalize(topic_scores)
        for sensor_idx in sensor_idxs:
            sensor_name = sensor_names[sensor_idx]
            for topic_name, topic_weight in zip(topic_names, topic_weights):
                if topic_name is None or topic_weight <= 0.0:
                    continue
                matrix[topic_name][sensor_name] += float(topic_weight)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / f"sensor_to_topic_closeness_{timestamp}.csv"

    logging.info("Writing CSV to %s", output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", *sensor_names])
        for _, topic_name in tqdm(topics_id_name, desc="Writing rows"):
            writer.writerow(
                [
                    topic_name,
                    *[matrix[topic_name][sensor] for sensor in sensor_names],
                ]
            )

    logging.info("Done")
