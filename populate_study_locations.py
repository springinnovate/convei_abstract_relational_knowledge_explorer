import argparse
import re

from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session
from tqdm import tqdm

from models import Base, Location, Publication, PublicationStudyLocation


LOCATION_ALIASES = {
    "usa": "United States",
    "u s a": "United States",
    "u.s.a": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "pr china": "China",
    "p r china": "China",
    "peoples r china": "China",
    "peoples republic of china": "China",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="2025_11_09_researchgate.sqlite")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = text.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def compile_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase)
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def build_matchers(location_rows: list[tuple[int, str]]):
    matchers: list[tuple[re.Pattern[str], int, str, str]] = []
    location_name_to_id = {
        location_name.casefold(): location_id
        for location_id, location_name in location_rows
    }

    for alias, canonical_name in LOCATION_ALIASES.items():
        location_id = location_name_to_id.get(canonical_name.casefold())
        if location_id is None:
            continue
        matchers.append(
            (compile_pattern(normalize_text(alias)), location_id, alias, "alias")
        )

    for location_id, location_name in location_rows:
        normalized_name = normalize_text(location_name)
        if not normalized_name:
            continue
        matchers.append(
            (
                compile_pattern(normalized_name),
                location_id,
                location_name,
                "direct",
            )
        )

    matchers.sort(key=lambda item: len(item[2]), reverse=True)
    return matchers


def build_rows_for_publication(
    publication_id: int,
    abstract: str | None,
    matchers: list[tuple[re.Pattern[str], int, str, str]],
) -> list[dict]:
    if not abstract:
        return []

    normalized_abstract = normalize_text(abstract)
    if not normalized_abstract:
        return []

    seen_keys: set[tuple[int, str]] = set()
    rows_to_insert: list[dict] = []

    for pattern, location_id, matched_text, match_method in matchers:
        if pattern.search(normalized_abstract):
            key = (location_id, matched_text)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows_to_insert.append(
                {
                    "publication_id": publication_id,
                    "location_id": location_id,
                    "matched_text": matched_text,
                    "match_method": match_method,
                }
            )

    return rows_to_insert


def main() -> None:
    args = parse_args()

    engine = create_engine(f"sqlite:///{args.db}", future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        location_rows = session.execute(select(Location.id, Location.name)).all()
        publication_query = (
            select(Publication.id, Publication.abstract)
            .where(Publication.abstract.is_not(None))
            .where(Publication.abstract != "")
            .order_by(Publication.id)
        )
        if args.limit is not None:
            publication_query = publication_query.limit(args.limit)
        publication_rows = session.execute(publication_query).all()

    matchers = build_matchers(location_rows)

    total_rows_inserted = 0
    total_publications_with_matches = 0
    pending_rows: list[dict] = []

    with Session(engine) as session:
        for publication_id, abstract in tqdm(
            publication_rows,
            desc="Linking study locations from abstracts",
        ):
            rows_to_insert = build_rows_for_publication(
                publication_id,
                abstract,
                matchers,
            )
            if rows_to_insert:
                total_publications_with_matches += 1
                pending_rows.extend(rows_to_insert)

            if len(pending_rows) >= args.batch_size:
                insert_statement = sqlite_insert(PublicationStudyLocation).values(
                    pending_rows
                )
                insert_statement = insert_statement.on_conflict_do_nothing(
                    index_elements=[
                        "publication_id",
                        "location_id",
                        "matched_text",
                    ]
                )
                result = session.execute(insert_statement)
                session.commit()
                total_rows_inserted += result.rowcount or 0
                pending_rows.clear()

        if pending_rows:
            insert_statement = sqlite_insert(PublicationStudyLocation).values(
                pending_rows
            )
            insert_statement = insert_statement.on_conflict_do_nothing(
                index_elements=[
                    "publication_id",
                    "location_id",
                    "matched_text",
                ]
            )
            result = session.execute(insert_statement)
            session.commit()
            total_rows_inserted += result.rowcount or 0

    print(f"publications_processed={len(publication_rows)}")
    print(f"publications_with_matches={total_publications_with_matches}")
    print(f"rows_inserted={total_rows_inserted}")


if __name__ == "__main__":
    main()
