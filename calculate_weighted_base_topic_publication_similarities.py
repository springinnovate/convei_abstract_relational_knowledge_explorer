"""
Exports top-N publications per base topic to a CSV, ranked by a weighted semantic
similarity score.

The weighted score for a (base_topic, publication) pair is computed as:

  clamp01(sim(base_topic, publication)) * clamp01(sim(weight_topic, publication))

Where:
- sim(...) is the stored semantic similarity in BaseTopicToPublicationDistance
- clamp01(...) clamps similarities into the [0.0, 1.0] range

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
from sqlalchemy import create_engine, select, desc, case, literal
from sqlalchemy.orm import sessionmaker, aliased

from models import BaseTopics, BaseTopicToPublicationDistance, Publication

engine = create_engine("sqlite:///2025_11_09_researchgate.sqlite")
SessionLocal = sessionmaker(bind=engine)


def clamp01(expr):
    """Clamp a SQLAlchemy numeric expression to the inclusive [0.0, 1.0] range.

    Args:
        expr: A SQLAlchemy expression that evaluates to a numeric value.

    Returns:
        A SQLAlchemy CASE expression that yields:
        - 0.0 when expr < 0.0
        - 1.0 when expr > 1.0
        - expr otherwise
    """
    return case(
        (expr < 0.0, 0.0),
        (expr > 1.0, 1.0),
        else_=expr,
    )


def export_top_n_per_base_topic_csv(
    session, n: int, output_path: str, weight_topic_text: str
) -> None:
    """Export top-N publications per base topic to a CSV using a weighted score.

    For each base topic, selects up to N publications and ranks them by:

      clamp01(sim(base_topic, publication)) * clamp01(sim(weight_topic, publication))

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

    D = BaseTopicToPublicationDistance
    W = aliased(BaseTopicToPublicationDistance)

    base_topics = session.execute(
        select(BaseTopics.id, BaseTopics.text).order_by(BaseTopics.text.asc())
    ).all()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["base topic", "weighted semantic similarity", "title", "abstract"]
        )

        for bt_id, bt_text in base_topics:
            weighted_score = clamp01(D.semantic_similarity) * clamp01(
                W.semantic_similarity
            )

            rows = session.execute(
                select(
                    BaseTopics.text,
                    BaseTopics.short_name,
                    weighted_score.label("weighted_score"),
                    Publication.title,
                    Publication.abstract,
                )
                .join(D, D.base_topic_id == BaseTopics.id)
                .join(Publication, Publication.id == D.publication_id)
                .join(
                    W,
                    (W.publication_id == D.publication_id)
                    & (W.base_topic_id == literal(weight_topic_id)),
                )
                .where(BaseTopics.id == bt_id)
                .order_by(desc(weighted_score), Publication.id.asc())
                .limit(n)
            ).all()

            for (
                base_topic,
                base_topic_short_name,
                wsim,
                title,
                abstract,
            ) in rows:
                writer.writerow([base_topic_short_name, float(wsim), title, abstract])


with SessionLocal() as session:
    export_top_n_per_base_topic_csv(
        session,
        n=100,
        output_path="top_pubs_by_base_topic.csv",
        weight_topic_text="Satellites that Observe and Image the Earth's Surface",
    )
