from __future__ import annotations

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Publication(Base):
    __tablename__ = "publications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text)

    doi: Mapped[str | None] = mapped_column(String(255), index=True)

    published_in_type: Mapped[str] = mapped_column(String(32), nullable=False)
    published_in_name: Mapped[str | None] = mapped_column(Text)

    authors: Mapped[str] = mapped_column(Text, nullable=False)
    author_affiliations: Mapped[str | None] = mapped_column(Text)
    author_emails: Mapped[str | None] = mapped_column(Text)

    publication_year: Mapped[int | None] = mapped_column(Integer)
    publication_month: Mapped[int | None] = mapped_column(Integer)
    publication_day: Mapped[int | None] = mapped_column(Integer)

    satellite_type: Mapped[str | None] = mapped_column(Text)
    type_evidence: Mapped[str | None] = mapped_column(Text)

    abstract_embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    raw_topics: Mapped[list["RawTopics"]] = relationship(
        secondary="raw_topic_to_pub",
        back_populates="publications",
        lazy="selectin",
    )

    base_topic_distances: Mapped[list["BaseTopicToPublicationDistance"]] = relationship(
        back_populates="publication",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_publications_year_type", "publication_year", "published_in_type"),
    )


class RawTopics(Base):
    __tablename__ = "raw_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    count: Mapped[int] = mapped_column(Integer, nullable=True, primary_key=False)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    include_in_analysis: Mapped[bool] = mapped_column(
        Boolean,
        nullable=True,
    )

    publications: Mapped[list["Publication"]] = relationship(
        secondary="raw_topic_to_pub",
        back_populates="raw_topics",
        lazy="selectin",
    )


class RawTopicToPublication(Base):
    __tablename__ = "raw_topic_to_pub"

    topic_id: Mapped[int] = mapped_column(
        ForeignKey("raw_topics.id", ondelete="CASCADE"),
        primary_key=True,
    )
    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"),
        primary_key=True,
    )

    __table_args__ = (
        UniqueConstraint("topic_id", "publication_id", name="uq_raw_topic_pub"),
        Index("ix_raw_topic_pub_pub_id", "publication_id", "topic_id"),
        Index("ix_raw_topic_pub_topic_id", "topic_id", "publication_id"),
    )


class BaseTopics(Base):
    __tablename__ = "base_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    publication_distances: Mapped[
        list["BaseTopicToPublicationDistance"]
    ] = relationship(
        back_populates="base_topic",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class BaseTopicToPublicationDistance(Base):
    __tablename__ = "base_topic_to_pub_distance"

    base_topic_id: Mapped[int] = mapped_column(
        ForeignKey("base_topics.id", ondelete="CASCADE"),
        primary_key=True,
    )
    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"),
        primary_key=True,
    )

    semantic_similarity: Mapped[float] = mapped_column(Float, nullable=False)

    base_topic: Mapped["BaseTopics"] = relationship(
        back_populates="publication_distances",
        lazy="selectin",
    )
    publication: Mapped["Publication"] = relationship(
        back_populates="base_topic_distances",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "base_topic_id", "publication_id", name="uq_base_topic_pub_distance"
        ),
        Index(
            "ix_base_topic_pub_distance_bt",
            "base_topic_id",
            "semantic_similarity",
            "publication_id",
        ),
        Index(
            "ix_base_topic_pub_distance_pub",
            "publication_id",
            "semantic_similarity",
            "base_topic_id",
        ),
    )
