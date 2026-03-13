import re

from tqdm import tqdm
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from models import Location, Publication, PublicationPrimaryAuthorLocation

db_path = "2025_11_09_researchgate.sqlite"
engine = create_engine(f"sqlite:///{db_path}")

state_names = [
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

state_abbreviations = [
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

state_name_regex = re.compile(
    "|".join(
        re.escape(state_name)
        for state_name in sorted(state_names, key=len, reverse=True)
    ),
    re.IGNORECASE,
)

state_abbreviation_regex = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(state_abbreviations) + r")(?![A-Za-z])"
)


def extract_first_affiliation(author_affiliations: str | None) -> str:
    if not author_affiliations:
        return ""

    affiliation_text = author_affiliations

    if "]" in affiliation_text:
        affiliation_text = affiliation_text.split("]", 1)[1]

    return affiliation_text.split(";", 1)[0].strip()


with Session(engine) as session:
    united_states_location_id = session.execute(
        select(Location.id).where(Location.name == "United States")
    ).scalar_one()
    print(united_states_location_id)

rows_to_insert = []
insert_batch_size = 5000

with Session(engine) as session:
    publications_without_locations = session.execute(
        select(Publication.id, Publication.author_affiliations)
        .outerjoin(
            PublicationPrimaryAuthorLocation,
            PublicationPrimaryAuthorLocation.publication_id == Publication.id,
        )
        .where(PublicationPrimaryAuthorLocation.publication_id.is_(None))
        .where(Publication.author_affiliations.is_not(None))
        .where(Publication.author_affiliations != "")
    ).all()

    for publication_id, author_affiliations in tqdm(
        publications_without_locations,
        desc="Linking missing affiliations to United States",
    ):
        first_affiliation = extract_first_affiliation(author_affiliations)

        has_state_name_match = bool(state_name_regex.search(first_affiliation))
        has_state_abbreviation_match = bool(
            state_abbreviation_regex.search(first_affiliation.upper())
        )

        if has_state_name_match or has_state_abbreviation_match:
            rows_to_insert.append(
                {
                    "publication_id": publication_id,
                    "location_id": united_states_location_id,
                }
            )

        if len(rows_to_insert) >= insert_batch_size:
            insert_statement = sqlite_insert(
                PublicationPrimaryAuthorLocation
            ).values(rows_to_insert)
            insert_statement = insert_statement.on_conflict_do_nothing(
                index_elements=["publication_id", "location_id"]
            )
            session.execute(insert_statement)
            session.commit()
            rows_to_insert.clear()

    if rows_to_insert:
        insert_statement = sqlite_insert(
            PublicationPrimaryAuthorLocation
        ).values(rows_to_insert)
        insert_statement = insert_statement.on_conflict_do_nothing(
            index_elements=["publication_id", "location_id"]
        )
        session.execute(insert_statement)
        session.commit()
