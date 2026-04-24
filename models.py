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


class PublicationToSatellite(Base):
    __tablename__ = "publication_to_satellite"

    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"),
        primary_key=True,
    )
    satellite_id: Mapped[int] = mapped_column(
        ForeignKey("satellite_type.id", ondelete="CASCADE"),
        primary_key=True,
    )

    publication: Mapped["Publication"] = relationship(
        back_populates="publication_satellites"
    )
    satellite: Mapped["Satellite"] = relationship(
        back_populates="publication_satellites"
    )


class PublicationToDataType(Base):
    __tablename__ = "publication_to_data_type"

    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"),
        primary_key=True,
    )
    data_type_id: Mapped[int] = mapped_column(
        ForeignKey("data_type.id", ondelete="CASCADE"),
        primary_key=True,
    )

    publication: Mapped["Publication"] = relationship(
        back_populates="publication_data_types"
    )
    data_type: Mapped["DataType"] = relationship(
        back_populates="publication_data_types"
    )


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

    abstract_embedding: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True
    )

    base_topic_distances: Mapped[list["BaseTopicToPublicationDistance"]] = (
        relationship(
            back_populates="publication",
            cascade="all, delete-orphan",
            lazy="selectin",
        )
    )

    affiliation_type_distances: Mapped[
        list["AffiliationTypeToPublicationDistance"]
    ] = relationship(
        back_populates="publication",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    primary_author_locations: Mapped[
        list["PublicationPrimaryAuthorLocation"]
    ] = relationship(
        back_populates="publication",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    author_locations: Mapped[list["PublicationAuthorLocation"]] = relationship(
        back_populates="publication",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    publication_satellites: Mapped[list["PublicationToSatellite"]] = (
        relationship(
            back_populates="publication",
            cascade="all, delete-orphan",
            lazy="selectin",
        )
    )
    satellites: Mapped[list["Satellite"]] = relationship(
        secondary="publication_to_satellite",
        back_populates="publications",
        lazy="selectin",
        viewonly=True,
    )

    publication_data_types: Mapped[list["PublicationToDataType"]] = (
        relationship(
            back_populates="publication",
            cascade="all, delete-orphan",
            lazy="selectin",
        )
    )
    data_types: Mapped[list["DataType"]] = relationship(
        secondary="publication_to_data_type",
        back_populates="publications",
        lazy="selectin",
        viewonly=True,
    )

    __table_args__ = (
        Index(
            "ix_publications_year_type", "publication_year", "published_in_type"
        ),
    )


class Satellite(Base):
    __tablename__ = "satellite_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    publication_satellites: Mapped[list["PublicationToSatellite"]] = (
        relationship(
            back_populates="satellite",
            cascade="all, delete-orphan",
            lazy="selectin",
        )
    )
    publications: Mapped[list["Publication"]] = relationship(
        secondary="publication_to_satellite",
        back_populates="satellites",
        lazy="selectin",
        viewonly=True,
    )


class DataType(Base):
    __tablename__ = "data_type"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    publication_data_types: Mapped[list["PublicationToDataType"]] = (
        relationship(
            back_populates="data_type",
            cascade="all, delete-orphan",
            lazy="selectin",
        )
    )
    publications: Mapped[list["Publication"]] = relationship(
        secondary="publication_to_data_type",
        back_populates="data_types",
        lazy="selectin",
        viewonly=True,
    )


class BaseTopics(Base):
    __tablename__ = "base_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    publication_distances: Mapped[list["BaseTopicToPublicationDistance"]] = (
        relationship(
            back_populates="base_topic",
            cascade="all, delete-orphan",
            lazy="selectin",
        )
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


class AffiliationType(Base):
    __tablename__ = "affiliation_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    publication_distances: Mapped[
        list["AffiliationTypeToPublicationDistance"]
    ] = relationship(
        back_populates="affiliation_type",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    author_affiliation_distances: Mapped[
        list["PublicationAuthorLocationAffiliationTypeDistance"]
    ] = relationship(
        back_populates="affiliation_type",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class AffiliationTypeToPublicationDistance(Base):
    __tablename__ = "affiliation_type_to_pub_distance"

    affiliation_type_id: Mapped[int] = mapped_column(
        ForeignKey("affiliation_types.id", ondelete="CASCADE"),
        primary_key=True,
    )
    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"),
        primary_key=True,
    )

    semantic_similarity: Mapped[float] = mapped_column(Float, nullable=False)

    affiliation_type: Mapped["AffiliationType"] = relationship(
        back_populates="publication_distances",
        lazy="selectin",
    )
    publication: Mapped["Publication"] = relationship(
        back_populates="affiliation_type_distances",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "affiliation_type_id",
            "publication_id",
            name="uq_affiliation_type_pub_distance",
        ),
        Index(
            "ix_affiliation_type_pub_distance_at",
            "affiliation_type_id",
            "semantic_similarity",
            "publication_id",
        ),
        Index(
            "ix_affiliation_type_pub_distance_pub",
            "publication_id",
            "semantic_similarity",
            "affiliation_type_id",
        ),
    )


class PublicationAuthorLocationAffiliationTypeDistance(Base):
    __tablename__ = "publication_author_location_to_affiliation_type_distance"

    publication_author_location_id: Mapped[int] = mapped_column(
        ForeignKey("publication_author_locations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    affiliation_type_id: Mapped[int] = mapped_column(
        ForeignKey("affiliation_types.id", ondelete="CASCADE"),
        primary_key=True,
    )
    semantic_similarity: Mapped[float] = mapped_column(Float, nullable=False)

    publication_author_location: Mapped["PublicationAuthorLocation"] = relationship(
        back_populates="affiliation_type_distances",
        lazy="selectin",
    )
    affiliation_type: Mapped["AffiliationType"] = relationship(
        back_populates="author_affiliation_distances",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "publication_author_location_id",
            "affiliation_type_id",
            name="uq_author_location_affiliation_type_distance",
        ),
        Index(
            "ix_author_location_affiliation_type_distance_loc",
            "publication_author_location_id",
            "semantic_similarity",
            "affiliation_type_id",
        ),
        Index(
            "ix_author_location_affiliation_type_distance_type",
            "affiliation_type_id",
            "semantic_similarity",
            "publication_author_location_id",
        ),
    )


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    publication_primary_author_locations: Mapped[
        list["PublicationPrimaryAuthorLocation"]
    ] = relationship(
        back_populates="location",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    publication_author_locations: Mapped[list["PublicationAuthorLocation"]] = (
        relationship(
            back_populates="location",
            cascade="all, delete-orphan",
            lazy="selectin",
        )
    )


class PublicationPrimaryAuthorLocation(Base):
    __tablename__ = "publication_primary_author_locations"

    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"),
        primary_key=True,
    )
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id", ondelete="RESTRICT"),
        primary_key=True,
    )

    publication: Mapped["Publication"] = relationship(
        back_populates="primary_author_locations",
        lazy="selectin",
    )
    location: Mapped["Location"] = relationship(
        back_populates="publication_primary_author_locations",
        lazy="selectin",
    )


class PublicationAuthorLocation(Base):
    __tablename__ = "publication_author_locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"),
        nullable=False,
    )
    author_name: Mapped[str] = mapped_column(Text, nullable=False, default="")
    author_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_author_group: Mapped[str] = mapped_column(
        Text, nullable=False, default=""
    )
    affiliation_text: Mapped[str] = mapped_column(Text, nullable=False)
    affiliation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id", ondelete="RESTRICT"),
        nullable=False,
    )

    publication: Mapped["Publication"] = relationship(
        back_populates="author_locations",
        lazy="selectin",
    )
    location: Mapped["Location"] = relationship(
        back_populates="publication_author_locations",
        lazy="selectin",
    )
    affiliation_type_distances: Mapped[
        list["PublicationAuthorLocationAffiliationTypeDistance"]
    ] = relationship(
        back_populates="publication_author_location",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "publication_id",
            "author_name",
            "location_id",
            "affiliation_index",
            name="uq_publication_author_location",
        ),
        Index(
            "ix_publication_author_location_pub",
            "publication_id",
            "author_name",
            "affiliation_index",
        ),
        Index(
            "ix_publication_author_location_loc",
            "location_id",
            "author_name",
            "publication_id",
        ),
    )
