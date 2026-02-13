from __future__ import annotations

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
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
    abstract_embedding: Mapped[bytes | None] = mapped_column(LargeBinary)

    satellite_type: Mapped[str | None] = mapped_column(Text)
    type_evidence: Mapped[str | None] = mapped_column(Text)

    raw_topics: Mapped[list["RawTopics"]] = relationship(
        secondary="raw_topic_to_pub",
        back_populates="publications",
        lazy="selectin",
    )
    normalized_topics: Mapped[list["NormalizedTopics"]] = relationship(
        secondary="normalized_topic_to_pub",
        back_populates="publications",
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


class NormalizedTopics(Base):
    __tablename__ = "normalized_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    publications: Mapped[list["Publication"]] = relationship(
        secondary="normalized_topic_to_pub",
        back_populates="normalized_topics",
        lazy="selectin",
    )


class NormalizedTopicToPublication(Base):
    __tablename__ = "normalized_topic_to_pub"

    topic_id: Mapped[int] = mapped_column(
        ForeignKey("normalized_topics.id", ondelete="CASCADE"),
        primary_key=True,
    )
    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"),
        primary_key=True,
    )

    __table_args__ = (
        UniqueConstraint("topic_id", "publication_id", name="uq_norm_topic_pub"),
        Index("ix_norm_topic_pub_pub_id", "publication_id", "topic_id"),
        Index("ix_norm_topic_pub_topic_id", "topic_id", "publication_id"),
    )
