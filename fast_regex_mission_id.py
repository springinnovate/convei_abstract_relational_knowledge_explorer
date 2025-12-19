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


import re

EO_GROUPS: dict[str, str] = {
    "Sentinel": r"""
        (?:copernicus\s+)?sentinel[-\s]?(?:[1-6](?:[ab]|[cd])?|5p|5)\b
        |s(?:1|2|3|4|5|6)(?:[-\s]?(?:a|b|c|d|p))?\b(?=\s*(?:data|imagery|image|products?|product|scene|tile|l1c|l2a|grd|slc|msi|olci|slstr|sar))
        |msi\b(?=\s*(?:sentinel|s2|l1c|l2a|tile|imagery|data|sensor|instrument))
        |olci\b(?=\s*(?:sentinel|s3|imagery|data|sensor|instrument))
        |slstr\b(?=\s*(?:sentinel|s3|imagery|data|sensor|instrument))
        |tropomi\b
        |s5p\b
    """,
    "Landsat": r"""
        landsat(?:[-\s]?(?:[1-9]))?\b
        |etm\+\b
        |oli\b
        |tirs\b
        |oli[-\s]?tirs\b
    """,
    "MODIS": r"""
        modis\b
        |(?:mod|myd|mcd)\d{2,3}\b
    """,
    "ASTER": r"""
        aster(?:\s*gdem)?\b
    """,
    "VIIRS": r"""
        viirs\b
        |suomi\s+npp\b
        |noaa[-\s]?(?:20|21)\b
        |jpss(?:[-\s]?\d+)?\b
        |joint\s+polar\s+satellite\s+system\b
        |npp\b(?=\s*(?:viirs|suomi|satellite|mission|platform))
    """,
    "Terra": r"""
        terra\b(?=\s*(?:modis|aster|nasa|satellite|platform|data|imagery))
    """,
    "Aqua": r"""
        aqua\b(?=\s*(?:modis|nasa|satellite|platform|data|imagery))
    """,
    "Maxar": r"""
        world\s*view\b
        |worldview\b
        |wv[-\s]?(?:0?1|0?2|0?3|0?4)\b
        |wv0?(?:1|2|3|4)\b
        |geoeye(?:[-\s]?1)?\b
        |ge[-\s]?1\b
    """,
    "QuickBird": r"""
        quick\s*bird\b
        |quickbird\b
    """,
    "IKONOS": r"""
        ikonos\b
    """,
    "Planet": r"""
        planetscope\b
        |planet\s*scope\b
        |planetlabs?\b
        |dove\b(?=\s*(?:satellites?|imagery|constellation|data))
        |super[-\s]?dove\b
        |superdove\b
        |rapideye\b
        |skysat\b
        |sky[-\s]?sat\b
    """,
    "Pléiades": r"""
        pl[ée]iades(?:[-\s]?(?:neo|1a|1b))?\b
        |pleiades\s+neo\b
    """,
    "SPOT": r"""
        spot(?:[-\s]?(?:[1-7]))?\b
    """,
    "OtherOpticalCommercial": r"""
        blacksky\b
        |satellogic\b
        |newsat\b
        |triplesat\b
        |eros\b(?=\s*(?:satellite|imagery|earth\s+observation))
    """,
    "GOES": r"""
        goes(?:[-\s]?(?:\d{1,2}|r|s|t|u))?\b
    """,
    "METEOSAT": r"""
        meteosat\b
        |mtg\b
        |meteosat\s+third\s+generation\b
        |msg\b(?=\s*(?:meteosat|seviri|satellite))
        |meteosat\s+second\s+generation\b
        |seviri\b
    """,
    "MetOp": r"""
        metop(?:[-\s]?(?:a|b|c))?\b
    """,
    "Himawari": r"""
        himawari[-\s]?(?:8|9)\b
    """,
    "INSAT": r"""
        insat(?:[-\s]?\d+(?:[a-z])?)?\b
    """,
    "GeoKOMPSAT": r"""
        gk[-\s]?2a\b
        |gk2a\b
        |geokompsat[-\s]?2a\b
        |geo\s*kompsat[-\s]?2a\b
    """,
    "Envisat": r"""
        envisat\b
        |asar\b
    """,
    "ERS": r"""
        ers[-\s]?(?:1|2)\b
    """,
    "RADARSAT": r"""
        radarsat(?:[-\s]?(?:1|2))?\b
    """,
    "RCM": r"""
        radarsat\s+constellation\s+mission\b
        |rcm(?:[-\s]?(?:1|2|3))?\b
    """,
    "TerraSAR": r"""
        terrasar[-\s]?x\b
        |tandem[-\s]?x\b
        |terrasar[-\s]?l\b
    """,
    "COSMO-SkyMed": r"""
        cosmo[-\s]?skymed\b
        |cosmo\s+sky\s*med\b
    """,
    "ALOS": r"""
        alos(?:[-\s]?(?:1|2|3|4))?\b
        |daichi(?:[-\s]?(?:1|2))?\b
        |palsar(?:[-\s]?2)?\b
    """,
    "SAOCOM": r"""
        saocom(?:[-\s]?(?:1a|1b))?\b
    """,
    "PAZ": r"""
        paz\b(?=\s*(?:satellite|sar|mission))
        |paz\s+satellite\b
    """,
    "KOMPSAT": r"""
        kompsat(?:[-\s]?(?:1|2|3|3a|5|6|7))?\b
    """,
    "RISAT": r"""
        risat(?:[-\s]?(?:1|2|2b|2br1|\d+))?\b
    """,
    "ICEYE": r"""
        iceye\b
    """,
    "Capella": r"""
        capella\s+space\b
        |capella\b(?=\s*(?:space|sar|satellites?|constellation))
    """,
    "Umbra": r"""
        umbra\s+space\b
        |umbra\b(?=\s*(?:space|sar|satellites?|constellation))
    """,
    "NovaSAR": r"""
        novasar(?:[-\s]?1)?\b
    """,
    "NISAR": r"""
        nisar\b
    """,
    "Gaofen": r"""
        gaofen[-\s]?(?:1|2|3|4|5|6|7)\b
        |gao\s*fen[-\s]?(?:1|2|3|4|5|6|7)\b
        |gf[-\s]?(?:1|2|3|4|5|6|7)\b
    """,
    "ZiYuan": r"""
        ziyuan[-\s]?(?:1|3)\b
        |zi\s*yuan[-\s]?(?:1|3)\b
        |zy[-\s]?(?:1|3)\b
    """,
    "Jilin": r"""
        jilin[-\s]?1\b
        |jl[-\s]?1\b
    """,
    "SuperView": r"""
        superview(?:[-\s]?(?:1|2|neo))?\b
        |super\s*view\b
    """,
    "HJ": r"""
        hj[-\s]?(?:1|2)(?:[-\s]?[a-z])?\b
        |huanjing\b
    """,
    "Tianhui": r"""
        tianhui\b
        |th[-\s]?(?:1|2|3)\b
    """,
    "CBERS": r"""
        cbers(?:[-\s]?(?:1|2|3|4|4a))?\b
    """,
    "Cartosat": r"""
        cartosat(?:[-\s]?(?:1|2|2a|2b|3))?\b
        |cartosat\s+dem\b
    """,
    "PRISMA": r"""
        prisma\b
    """,
    "EnMAP": r"""
        enmap\b
    """,
    "PROBA-V": r"""
        proba[-\s]?v\b
    """,
    "GHGSat": r"""
        ghgsat\b
    """,
    "PACE": r"""
        nasa\s+pace\b
        |pace\s+mission\b
        |pace\b(?=\s*(?:mission|satellite|spacecraft))
        |ocean\s+color\s+instrument\b
        |oci\b
    """,
}

PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        label,
        re.compile(rf"(?ix)\b(?:{alt.strip()})", re.IGNORECASE | re.VERBOSE),
    )
    for label, alt in EO_GROUPS.items()
]

EO_IMAGING_ANY = re.compile(
    rf"(?ix)\b(?:{'|'.join(f'(?:{alt.strip()})' for alt in EO_GROUPS.values())})",
    re.IGNORECASE | re.VERBOSE,
)


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
