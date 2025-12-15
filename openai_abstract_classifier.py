from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
import argparse
import json
import logging

from sqlalchemy import event, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from models import Publication


logging.basicConfig(
    level=logging.INFO,
    format="[%(filename)s:%(lineno)d] %(levelname)s %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

load_dotenv()
CLIENT = OpenAI(timeout=600.0)

SATELLITE_LABELS = [
    "Sentinel",
    "Landsat",
    "MODIS",
    "ASTER",
    "VIIRS",
    "WorldView",
    "QuickBird",
    "PlanetScope",
    "IKONOS",
    "NOAA",
    "GOES",
    "METEOSAT",
    "SPOT",
    "RADARSAT",
    "TerraSAR",
    "COSMO-SkyMed",
    "ALOS",
    "Envisat",
    "GHGSat",
    "SkySat",
    "PRISMA",
]

GENERIC_LABELS = [
    "GENERIC_SATELLITE",
    "GENERIC_REMOTE_SOURCED",
    "NOT_SATELLITE_RELATED",
    "UNKNOWN",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", required=True)
    p.add_argument("--limit", type=int)
    return p.parse_args()


def build_prompt(title: str | None, abstract: str | None) -> str:
    text_title = title or ""
    text_abstract = abstract or ""
    labels_str = ", ".join(SATELLITE_LABELS)
    generic_str = ", ".join(GENERIC_LABELS)
    return (
        "You are a remote sensing expert.\n"
        "Given the title and abstract of a research article, determine:\n"
        "1) Which, if any, specific satellite platforms or sensors from this allowed list are clearly used in the work:\n"
        f"{labels_str}\n"
        "2) A coarse label describing whether the work clearly uses satellite data, some other remote sensing source, "
        "or is not really about satellite/remote sensing data.\n\n"
        "VERY IMPORTANT:\n"
        '- You MUST always provide a non-empty "evidence" string.\n'
        "- For GENERIC_SATELLITE or GENERIC_REMOTE_SOURCED, quote the exact phrases that indicate satellite/remote sensing use "
        '(e.g. "satellite imagery", "remote sensing data", "aerial photographs", "drone imagery").\n'
        "- For NOT_SATELLITE_RELATED or UNKNOWN, explicitly state that the abstract does not mention satellite or remote sensing "
        "and summarise what it *does* talk about.\n\n"
        "Rules:\n"
        "- Only return satellite names from the allowed list. Do not invent new satellite names.\n"
        '- If the abstract only mentions generic phrases like "satellite data", "remote sensing", or "Earth observation" '
        'without naming a specific satellite from the list, then satellites must be an empty list and generic_label should be "GENERIC_SATELLITE".\n'
        "- If the work clearly uses non-satellite remote sensing (e.g., aerial imagery, UAV/Drone, airborne LiDAR) but not satellites, "
        'set generic_label to "GENERIC_REMOTE_SOURCED".\n'
        '- If the work is not really about satellite or remote sensing data at all, set generic_label to "NOT_SATELLITE_RELATED".\n'
        '- If you cannot tell, set generic_label to "UNKNOWN".\n\n'
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "satellites": [list of zero or more strings, each one from the allowed satellite list],\n'
        '  "generic_label": one of [' + generic_str + "],\n"
        '  "evidence": "non-empty explanation quoting phrases from the abstract that justify the decision"\n'
        "}\n\n"
        "Title:\n"
        f"{text_title}\n\n"
        "Abstract:\n"
        f"{text_abstract}\n"
    )


log = logging.getLogger(__name__)


def call_model(
    client: OpenAI, title: str | None, abstract: str | None
) -> dict | None:
    prompt = build_prompt(title, abstract)
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured information about satellites used in research articles.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        log.warning("openai error: %s", e)
        return None

    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except JSONDecodeError as e:
        log.warning("json decode error: %s", e)
        return None


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
    client = OpenAI()

    with Session() as session:
        base_filter = (Publication.satellite_type.is_(None),)

        total_stmt = select(func.count()).select_from(
            select(Publication.id).where(*base_filter).subquery()
        )
        total = session.execute(total_stmt).scalar_one()
        log.info("rows to send to LLM: %d", total)

        stmt = select(Publication).where(*base_filter).order_by(Publication.id)

        batch_size = 50
        offset = 0
        max_workers = 64
        with tqdm(total=total) as pbar:
            while True:
                if args.limit:
                    remaining = args.limit - offset
                    if remaining <= 0:
                        break
                    pubs_batch = (
                        session.execute(
                            stmt.limit(min(batch_size, remaining)).offset(
                                offset
                            )
                        )
                        .scalars()
                        .all()
                    )
                else:
                    pubs_batch = (
                        session.execute(stmt.limit(batch_size).offset(offset))
                        .scalars()
                        .all()
                    )

                if not pubs_batch:
                    break

                inputs = [
                    (pub.id, pub.title, pub.abstract) for pub in pubs_batch
                ]
                results: dict[int, tuple[str, str]] = {}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            call_model, client, title, abstract
                        ): pub_id
                        for pub_id, title, abstract in inputs
                    }

                    for fut in as_completed(futures):
                        pub_id = futures[fut]
                        result = fut.result()
                        if result is None:
                            pbar.update(1)
                            continue

                        satellites = result.get("satellites") or []
                        generic_label = result.get("generic_label") or "UNKNOWN"
                        sat_value = (
                            ",".join(sorted(set(satellites)))
                            if satellites
                            else generic_label
                        )

                        evidence = result.get("evidence")
                        if not evidence:
                            if generic_label in (
                                "GENERIC_SATELLITE",
                                "GENERIC_REMOTE_SOURCED",
                            ):
                                evidence = "Model classified as {0} but did not provide evidence; abstract likely contains generic remote sensing language.".format(
                                    generic_label
                                )
                            else:
                                evidence = "Model classified as {0} and did not detect satellite or remote sensing related terms.".format(
                                    generic_label
                                )

                        results[pub_id] = (sat_value, evidence)
                        pbar.update(1)

                for pub in pubs_batch:
                    r = results.get(pub.id)
                    if not r:
                        continue
                    pub.satellite_type, pub.type_evidence = r

                session.commit()
                offset += len(pubs_batch)


if __name__ == "__main__":
    main()
