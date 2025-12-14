from __future__ import annotations

import argparse
import csv
import glob

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from models import Base, Publication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("pattern")
    return parser.parse_args()


MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def parse_date(
    pd_str: str | None, py_str: str | None
) -> tuple[str | None, int | None, int | None, int | None]:
    try:
        year = int(py_str)
        month = None
        day = None
        if pd_str:
            s = pd_str.strip()
            if s:
                mkey = s[:3].upper()
                month = MONTHS.get(mkey)
                parts = s.split()
                if len(parts) > 1 and parts[1].isdigit():
                    d = int(parts[1])
                    if 1 <= d <= 31:
                        day = d
        return year, month, day
    except Exception:
        return None, None, None


def parse_published_in_type(pt: str | None, dt: str | None) -> str:
    pt = (pt or "").strip().upper()
    dt = (dt or "").strip()
    if pt == "J":
        return "journal"
    if pt == "B":
        return "book"
    if pt == "S":
        return "series"
    if pt == "P":
        return "patent"
    if dt:
        return dt.lower()
    return "other"


def iter_publications_from_tsv(path: str):
    csv.field_size_limit(2**31 - 1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                af = (row.get("AF") or "").strip()
                au = (row.get("AU") or "").strip()
                authors = af or au or ""
                c1 = (row.get("C1") or "").strip()
                c3 = (row.get("C3") or "").strip()
                if c1 and c3:
                    author_affiliations = f"{c1}\n{c3}"
                else:
                    author_affiliations = c1 or c3 or None
                em = (row.get("EM") or "").strip()
                author_emails = em or None
                pd_str = row.get("PD")
                py_str = row.get("PY")
                payload = parse_date(pd_str, py_str)
                try:
                    (
                        publication_year,
                        publication_month,
                        publication_day,
                    ) = payload
                except ValueError as e:
                    print(f"error {path} on {payload}")
                    raise
                pt = row.get("PT")
                dt = row.get("DT")
                published_in_type = parse_published_in_type(pt, dt)
                published_in_name = (row.get("SO") or "").strip() or None
                title = (row.get("TI") or "").strip()
                abstract = (row.get("AB") or "").strip() or None
                doi = (row.get("DI") or "").strip() or None
                yield {
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "published_in_type": published_in_type,
                    "published_in_name": published_in_name,
                    "authors": authors,
                    "author_affiliations": author_affiliations,
                    "author_emails": author_emails,
                    "publication_year": publication_year,
                    "publication_month": publication_month,
                    "publication_day": publication_day,
                }
    except Exception as e:
        print(f"error when parsing {path} {e}")
        raise


def main() -> None:
    args = parse_args()

    engine = create_engine(f"sqlite:///{args.db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    files = glob.glob(args.pattern)

    for path in tqdm(files):
        with Session() as session:
            for publication_data in iter_publications_from_tsv(path):
                publication = Publication(**publication_data)
                session.add(publication)
            session.commit()


if __name__ == "__main__":
    main()
