import argparse
import re

from rapidfuzz import fuzz, process
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session
from tqdm import tqdm

from models import Base, Location, Publication, PublicationAuthorLocation


US_STATE_NAMES = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "District of Columbia",
]

US_STATE_ABBREVIATIONS = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
]

US_STATE_NAME_REGEX = re.compile(
    "|".join(
        re.escape(state_name)
        for state_name in sorted(US_STATE_NAMES, key=len, reverse=True)
    ),
    re.IGNORECASE,
)
US_STATE_ABBREVIATION_REGEX = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(US_STATE_ABBREVIATIONS) + r")(?![A-Za-z])"
)

LOCATION_ALIASES = {
    "usa": "United States",
    "u s a": "United States",
    "u.s.a": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    "pr china": "China",
    "p r china": "China",
    "peoples r china": "China",
    "peoples republic of china": "China",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="2025_11_09_researchgate.sqlite")
    parser.add_argument("--score-cutoff", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def split_segments_outside_brackets(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    bracket_depth = 0

    for char in text:
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1

        if char == ";" and bracket_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def parse_affiliation_segments(author_affiliations: str) -> list[dict]:
    lines = [line.strip() for line in author_affiliations.splitlines() if line.strip()]
    if not lines:
        return []

    has_bracketed_lines = any("[" in line and "]" in line for line in lines)
    source_lines = (
        [line for line in lines if "[" in line and "]" in line]
        if has_bracketed_lines
        else lines
    )

    segments: list[dict] = []
    affiliation_index = 0

    for line in source_lines:
        for segment in split_segments_outside_brackets(line):
            match = re.match(r"^\[(.*?)\]\s*(.*)$", segment)
            raw_author_group = ""
            authors: list[str] = []
            affiliation_text = segment.strip()

            if match:
                raw_author_group = match.group(1).strip()
                affiliation_text = match.group(2).strip()
                authors = [
                    author.strip()
                    for author in raw_author_group.split(";")
                    if author.strip()
                ]

            if not affiliation_text:
                continue

            segments.append(
                {
                    "affiliation_index": affiliation_index,
                    "raw_author_group": raw_author_group,
                    "authors": authors,
                    "affiliation_text": affiliation_text,
                }
            )
            affiliation_index += 1

    return segments


def normalize_lookup_text(text: str) -> str:
    normalized = text.casefold()
    normalized = normalized.replace(".", " ")
    normalized = normalized.replace(",", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def match_location_id(
    affiliation_text: str,
    location_name_to_id: dict[str, int],
    location_names: list[str],
    united_states_location_id: int | None,
    score_cutoff: int,
) -> tuple[int | None, str]:
    normalized_text = normalize_lookup_text(affiliation_text)

    for alias, canonical_location_name in LOCATION_ALIASES.items():
        if alias in normalized_text:
            location_id = location_name_to_id.get(
                canonical_location_name.casefold()
            )
            if location_id is not None:
                return location_id, f"alias:{canonical_location_name}"

    direct_matches = [
        location_name
        for location_name in location_names
        if location_name in normalized_text
    ]
    if direct_matches:
        direct_matches.sort(key=len, reverse=True)
        location_name = direct_matches[0]
        return location_name_to_id[location_name], "direct"

    if united_states_location_id is not None and (
        US_STATE_NAME_REGEX.search(affiliation_text)
        or US_STATE_ABBREVIATION_REGEX.search(affiliation_text.upper())
    ):
        return united_states_location_id, "us-heuristic"

    match = process.extractOne(
        normalized_text,
        location_names,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )
    if match:
        location_name = match[0]
        return location_name_to_id[location_name], "fuzzy"

    return None, "unmatched"


def build_rows_for_publication(
    publication_id: int,
    author_affiliations: str | None,
    location_name_to_id: dict[str, int],
    location_names: list[str],
    united_states_location_id: int | None,
    score_cutoff: int,
) -> tuple[list[dict], dict[str, int]]:
    if not author_affiliations:
        return [], {"matched_segments": 0, "unmatched_segments": 0}

    rows_to_insert: list[dict] = []
    stats = {"matched_segments": 0, "unmatched_segments": 0}

    for segment in parse_affiliation_segments(author_affiliations):
        location_id, _ = match_location_id(
            segment["affiliation_text"],
            location_name_to_id,
            location_names,
            united_states_location_id,
            score_cutoff,
        )

        if location_id is None:
            stats["unmatched_segments"] += 1
            continue

        stats["matched_segments"] += 1

        authors = segment["authors"] or [""]
        for author_index, author_name in enumerate(authors):
            rows_to_insert.append(
                {
                    "publication_id": publication_id,
                    "author_name": author_name,
                    "author_index": author_index if author_name else None,
                    "raw_author_group": segment["raw_author_group"],
                    "affiliation_text": segment["affiliation_text"],
                    "affiliation_index": segment["affiliation_index"],
                    "location_id": location_id,
                }
            )

    return rows_to_insert, stats


def main() -> None:
    args = parse_args()

    engine = create_engine(f"sqlite:///{args.db}", future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        location_rows = session.execute(select(Location.id, Location.name)).all()
        publication_query = (
            select(Publication.id, Publication.author_affiliations)
            .where(Publication.author_affiliations.is_not(None))
            .where(Publication.author_affiliations != "")
            .order_by(Publication.id)
        )
        if args.limit is not None:
            publication_query = publication_query.limit(args.limit)
        publication_rows = session.execute(publication_query).all()

    location_name_to_id = {
        location_name.casefold(): location_id
        for location_id, location_name in location_rows
    }
    location_names = list(location_name_to_id.keys())
    united_states_location_id = location_name_to_id.get("united states")

    total_rows_inserted = 0
    total_matched_segments = 0
    total_unmatched_segments = 0
    pending_rows: list[dict] = []

    with Session(engine) as session:
        for publication_id, author_affiliations in tqdm(
            publication_rows,
            desc="Linking all author affiliations to locations",
        ):
            rows_to_insert, stats = build_rows_for_publication(
                publication_id,
                author_affiliations,
                location_name_to_id,
                location_names,
                united_states_location_id,
                args.score_cutoff,
            )
            pending_rows.extend(rows_to_insert)
            total_matched_segments += stats["matched_segments"]
            total_unmatched_segments += stats["unmatched_segments"]

            if len(pending_rows) >= args.batch_size:
                insert_statement = sqlite_insert(PublicationAuthorLocation).values(
                    pending_rows
                )
                insert_statement = insert_statement.on_conflict_do_nothing(
                    index_elements=[
                        "publication_id",
                        "author_name",
                        "location_id",
                        "affiliation_index",
                    ]
                )
                result = session.execute(insert_statement)
                session.commit()
                total_rows_inserted += result.rowcount or 0
                pending_rows.clear()

        if pending_rows:
            insert_statement = sqlite_insert(PublicationAuthorLocation).values(
                pending_rows
            )
            insert_statement = insert_statement.on_conflict_do_nothing(
                index_elements=[
                    "publication_id",
                    "author_name",
                    "location_id",
                    "affiliation_index",
                ]
            )
            result = session.execute(insert_statement)
            session.commit()
            total_rows_inserted += result.rowcount or 0

    print(f"publications_processed={len(publication_rows)}")
    print(f"matched_affiliation_segments={total_matched_segments}")
    print(f"unmatched_affiliation_segments={total_unmatched_segments}")
    print(f"rows_inserted={total_rows_inserted}")


if __name__ == "__main__":
    main()
