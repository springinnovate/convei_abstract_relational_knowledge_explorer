import argparse
import re

from sqlalchemy import event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from tqdm.auto import tqdm
from sqlalchemy import func

from models import Publication


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", required=True)
    p.add_argument("--batch-size", type=int, default=2000)
    return p.parse_args()


PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "Sentinel",
        re.compile(
            r"\bsentinel[-\s]?(?:[1-6])\b|\bcopernicus sentinel[-\s]?(?:[1-6])\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Landsat",
        re.compile(r"\blandsat\b|\blandsat[-\s]?(?:[1-9])\b", re.IGNORECASE),
    ),
    (
        "MODIS",
        re.compile(r"\bmodis\b|\b(?:mod|myd|mcd)\d{2,3}\b", re.IGNORECASE),
    ),
    ("ASTER", re.compile(r"\baster\b", re.IGNORECASE)),
    (
        "VIIRS",
        re.compile(
            r"\bviirs\b|\bsuomi\s+npp\b|\bnoaa[-\s]?(?:20|21)\b", re.IGNORECASE
        ),
    ),
    ("WorldView", re.compile(r"\bworldview\b|\bworld view\b", re.IGNORECASE)),
    ("QuickBird", re.compile(r"\bquickbird\b|\bquick bird\b", re.IGNORECASE)),
    (
        "PlanetScope",
        re.compile(
            r"\bplanetscope\b|\bplanet\s*scope\b|\bplanetlabs?\b", re.IGNORECASE
        ),
    ),
    ("IKONOS", re.compile(r"\bikonos\b", re.IGNORECASE)),
    ("NOAA", re.compile(r"\bnoaa\b", re.IGNORECASE)),
    (
        "GOES",
        re.compile(r"\bgoes(?:[-\s]?(?:[0-9]{1,2}|r|s|t|u))?\b", re.IGNORECASE),
    ),
    ("METEOSAT", re.compile(r"\bmeteosat\b", re.IGNORECASE)),
    ("SPOT", re.compile(r"\bspot(?:[-\s]?(?:[1-7]))?\b", re.IGNORECASE)),
    ("RADARSAT", re.compile(r"\bradarsat(?:[-\s]?(?:1|2))?\b", re.IGNORECASE)),
    (
        "TerraSAR",
        re.compile(r"\bterrasar[-\s]?x\b|\btandem[-\s]?x\b", re.IGNORECASE),
    ),
    (
        "COSMO-SkyMed",
        re.compile(
            r"\bcosmo[-\s]?skymed\b|\bcosmo\s+sky\s*med\b", re.IGNORECASE
        ),
    ),
    (
        "ALOS",
        re.compile(
            r"\balos(?:[-\s]?(?:1|2))?\b|\bpalsar(?:[-\s]?2)?\b", re.IGNORECASE
        ),
    ),
    ("Envisat", re.compile(r"\benvisat\b|\basar\b", re.IGNORECASE)),
    ("GHGSat", re.compile(r"\bghgsat\b", re.IGNORECASE)),
    ("SkySat", re.compile(r"\bskysat\b|\bsky[-\s]?sat\b", re.IGNORECASE)),
    ("PRISMA", re.compile(r"\bprisma\b", re.IGNORECASE)),
]


def detect_satellite_types(
    title: str | None, abstract: str | None
) -> str | None:
    parts: list[str] = []
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    if not parts:
        return None
    text = " ".join(parts)
    found: list[str] = []
    for label, pat in PATTERNS:
        if pat.search(text):
            found.append(label)
    if not found:
        return None
    return ",".join(sorted(set(found)))


def main() -> None:
    args = parse_args()

    engine = create_engine(
        f"sqlite:///{args.db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    Session = sessionmaker(bind=engine, future=True, expire_on_commit=False)

    with Session() as session:
        stmt = (
            select(Publication)
            .where(Publication.satellite_type.is_(None))
            .order_by(Publication.id)
        )

        total = session.execute(
            select(func.count()).select_from(
                select(Publication.id)
                .where(Publication.satellite_type.is_(None))
                .subquery()
            )
        ).scalar_one()

        offset = 0
        batch_size = args.batch_size

        with tqdm(total=total) as pbar:
            while True:
                batch = (
                    session.execute(stmt.limit(batch_size).offset(offset))
                    .scalars()
                    .all()
                )
                if not batch:
                    break

                for pub in batch:
                    label = detect_satellite_types(pub.title, pub.abstract)
                    if label is not None:
                        pub.satellite_type = label

                session.commit()
                offset += batch_size
                pbar.update(len(batch))


if __name__ == "__main__":
    main()
