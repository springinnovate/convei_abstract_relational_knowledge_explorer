import csv
from sqlalchemy import create_engine, select, desc, case, literal
from sqlalchemy.orm import sessionmaker, aliased

from models import BaseTopics, BaseTopicToPublicationDistance, Publication

engine = create_engine("sqlite:///2025_11_09_researchgate.sqlite")
SessionLocal = sessionmaker(bind=engine)


def clamp01(expr):
    return case(
        (expr < 0.0, 0.0),
        (expr > 1.0, 1.0),
        else_=expr,
    )


def export_top_n_per_base_topic_csv(
    session, n: int, output_path: str, weight_topic_text: str
) -> None:
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
