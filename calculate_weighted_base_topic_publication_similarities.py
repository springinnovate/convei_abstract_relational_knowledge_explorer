"""
Exports top-N publications per base topic to a CSV, ranked by a weighted semantic
similarity score.

The weighted score for a (base_topic, publication) pair is computed as:

  transformed_weight(base_topic, publication)
  * transformed_weight(weight_topic, publication)

Where:
- transformed_weight(...) comes from fourth-power normalizing the full
  publication-level topic vector from BaseTopicToPublicationDistance

The output CSV contains one row per selected publication per base topic with:
- base topic (short name)
- weighted semantic similarity
- publication title
- publication abstract

Expected database schema:
- BaseTopics: base topic records (id, text, short_name, ...)
- BaseTopicToPublicationDistance: distances/similarities between base topics and
  publications (base_topic_id, publication_id, semantic_similarity, ...)
- Publication: publication records (id, title, abstract, ...)

Example:
  python export_top_pubs_by_base_topic.py
"""

import csv
from collections import defaultdict

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from affiliation_vector_transform import power_normalize
from models import BaseTopics, BaseTopicToPublicationDistance, Publication

engine = create_engine("sqlite:///2025_11_09_researchgate.sqlite")
SessionLocal = sessionmaker(bind=engine)


def load_transformed_topic_weights(session, topic_ids: list[int]):
    topic_index_by_id = {topic_id: index for index, topic_id in enumerate(topic_ids)}
    rows = session.execute(
        select(
            BaseTopicToPublicationDistance.publication_id,
            BaseTopicToPublicationDistance.base_topic_id,
            BaseTopicToPublicationDistance.semantic_similarity,
        ).order_by(
            BaseTopicToPublicationDistance.publication_id,
            BaseTopicToPublicationDistance.base_topic_id,
        )
    ).all()

    weights_by_publication = {}
    current_publication_id = None
    current_scores = None

    def flush_publication() -> None:
        if current_publication_id is not None and current_scores is not None:
            weights_by_publication[current_publication_id] = power_normalize(
                current_scores
            )

    for publication_id, topic_id, semantic_similarity in rows:
        if publication_id != current_publication_id:
            flush_publication()
            current_publication_id = publication_id
            current_scores = np.zeros(len(topic_ids), dtype=np.float64)
        current_scores[topic_index_by_id[topic_id]] = float(semantic_similarity)

    flush_publication()
    return weights_by_publication, topic_index_by_id


def export_top_n_per_base_topic_csv(
    session, n: int, output_path: str, weight_topic_text: str
) -> None:
    """Export top-N publications per base topic to a CSV using a weighted score.

    For each base topic, selects up to N publications and ranks them by:

      transformed_weight(base_topic, publication)
      * transformed_weight(weight_topic, publication)

    The weight topic is identified by matching BaseTopics.text to weight_topic_text.

    The CSV is written with a header row:
      ['base topic', 'weighted semantic similarity', 'title', 'abstract']

    Args:
        session: An active SQLAlchemy session bound to the target database.
        n: Maximum number of publications to export per base topic.
        output_path: Filesystem path where the CSV will be written.
        weight_topic_text: Exact BaseTopics.text string used to find the weight topic.

    Raises:
        sqlalchemy.exc.NoResultFound: If weight_topic_text does not match any topic.
        sqlalchemy.exc.MultipleResultsFound: If weight_topic_text matches multiple topics.
        OSError: If output_path cannot be opened for writing.
    """
    weight_topic_id = session.execute(
        select(BaseTopics.id).where(BaseTopics.text == weight_topic_text)
    ).scalar_one()

    base_topics = session.execute(
        select(BaseTopics.id, BaseTopics.text, BaseTopics.short_name).order_by(
            BaseTopics.text.asc()
        )
    ).all()
    topic_ids = [topic_id for topic_id, _, _ in base_topics]
    weights_by_publication, topic_index_by_id = load_transformed_topic_weights(
        session, topic_ids
    )
    weight_topic_index = topic_index_by_id[weight_topic_id]

    scores_by_topic = defaultdict(list)
    for publication_id, topic_weights in weights_by_publication.items():
        weight_topic_score = topic_weights[weight_topic_index]
        if weight_topic_score <= 0.0:
            continue
        for topic_id in topic_ids:
            topic_score = topic_weights[topic_index_by_id[topic_id]]
            weighted_score = topic_score * weight_topic_score
            if weighted_score > 0.0:
                scores_by_topic[topic_id].append(
                    (float(weighted_score), publication_id)
                )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["base topic", "weighted semantic similarity", "title", "abstract"]
        )

        selected_publication_ids = {
            publication_id
            for topic_scores in scores_by_topic.values()
            for _, publication_id in sorted(
                topic_scores, key=lambda item: (-item[0], item[1])
            )[:n]
        }
        publication_by_id = {
            publication_id: (title, abstract)
            for publication_id, title, abstract in session.execute(
                select(Publication.id, Publication.title, Publication.abstract).where(
                    Publication.id.in_(selected_publication_ids)
                )
            )
        }

        for bt_id, _, base_topic_short_name in base_topics:
            top_rows = sorted(
                scores_by_topic[bt_id], key=lambda item: (-item[0], item[1])
            )[:n]
            for weighted_score, publication_id in top_rows:
                title, abstract = publication_by_id[publication_id]
                writer.writerow(
                    [base_topic_short_name, weighted_score, title, abstract]
                )


with SessionLocal() as session:
    export_top_n_per_base_topic_csv(
        session,
        n=100,
        output_path="top_pubs_by_base_topic.csv",
        weight_topic_text="Satellites that Observe and Image the Earth's Surface",
    )
