from __future__ import annotations

from sqlalchemy import Integer, String, Text, LargeBinary, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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

    __table_args__ = (
        Index(
            "ix_publications_year_type", "publication_year", "published_in_type"
        ),
    )
