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
    "Terra",
    "Aqua",
    "Maxar",
    "WorldView",
    "GeoEye",
    "QuickBird",
    "IKONOS",
    "Planet",
    "PlanetScope",
    "Planet Labs",
    "Dove",
    "SuperDove",
    "RapidEye",
    "SkySat",
    "Pléiades",
    "SPOT",
    "OtherOpticalCommercial",
    "BlackSky",
    "Satellogic",
    "Newsat",
    "TripleSat",
    "EROS",
    "GOES",
    "METEOSAT",
    "MTG",
    "MSG",
    "SEVIRI",
    "MetOp",
    "Himawari",
    "INSAT",
    "GeoKOMPSAT",
    "Envisat",
    "ASAR",
    "ERS",
    "RADARSAT",
    "RCM",
    "RADARSAT Constellation Mission",
    "TerraSAR",
    "TerraSAR-X",
    "TanDEM-X",
    "TerraSAR-L",
    "COSMO-SkyMed",
    "ALOS",
    "Daichi",
    "PALSAR",
    "SAOCOM",
    "PAZ",
    "KOMPSAT",
    "RISAT",
    "ICEYE",
    "Capella",
    "Umbra",
    "NovaSAR",
    "NISAR",
    "Gaofen",
    "ZiYuan",
    "Jilin",
    "SuperView",
    "HJ",
    "Huanjing",
    "Tianhui",
    "CBERS",
    "Cartosat",
    "PRISMA",
    "EnMAP",
    "PROBA-V",
    "GHGSat",
    "PACE",
    "OCI",
    "TROPOMI",
    "S5P",
    "Suomi NPP",
    "NOAA",
    "JPSS",
    "Joint Polar Satellite System",
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
        "1) Which, if any, specific satellite platforms or sensors are clearly used in the work.\n"
        "2) A coarse label describing whether the work clearly uses satellite data, some other remote sensing source, "
        "or is not really about satellite/remote sensing data.\n\n"
        "You have an allowed label list. Prefer returning labels from this list when they match what the abstract says:\n"
        f"{labels_str}\n\n"
        "VERY IMPORTANT:\n"
        '- You MUST always provide a non-empty "evidence" string.\n'
        "- Evidence must quote exact phrases from the title/abstract that justify your decision.\n"
        "- For GENERIC_SATELLITE or GENERIC_REMOTE_SOURCED, quote the exact phrases that indicate satellite/remote sensing use "
        '(e.g. "satellite imagery", "remote sensing data", "aerial photographs", "drone imagery").\n'
        "- For NOT_SATELLITE_RELATED or UNKNOWN, explicitly state that the abstract does not clearly mention satellite or remote sensing "
        "and summarise what it *does* talk about.\n\n"
        "Decision rules:\n"
        "- First, try to match mentioned platforms/sensors to the allowed label list (case-insensitive, minor punctuation/spacing differences OK).\n"
        "- If a mentioned platform/sensor does NOT match the allowed list, but it is clearly a specific satellite/platform/sensor name "
        '(e.g. a named mission/constellation or a sensor name like "SAR", "MSI", "TROPOMI" when tied to a specific mission), '
        "you may include it in the satellites list as written in the abstract.\n"
        "- Do NOT invent names. Only include names explicitly present in the title/abstract.\n"
        "- If only generic phrases are used (e.g. '\"satellite data\"', '\"satellite imagery\"', '\"remote sensing\"', "
        '"Earth observation"\') with no specific platform/sensor name, then satellites must be an empty list and generic_label must be "GENERIC_SATELLITE".\n'
        "- If the work clearly uses non-satellite remote sensing (e.g., aerial imagery, UAV/drone, airborne LiDAR) but not satellites, "
        'set generic_label to "GENERIC_REMOTE_SOURCED" and satellites must be empty.\n'
        '- If it clearly indicates satellite use but you cannot identify any specific platform/sensor name, set generic_label to "GENERIC_SATELLITE".\n'
        '- If it is not really about satellite or remote sensing data at all, set generic_label to "NOT_SATELLITE_RELATED".\n'
        '- If you cannot tell from the title/abstract, set generic_label to "UNKNOWN".\n\n'
        "Output rules:\n"
        "- The satellites list may contain:\n"
        "  a) zero or more strings from the allowed label list, and/or\n"
        "  b) zero or more verbatim satellite/platform/sensor names quoted from the title/abstract when clearly specific.\n"
        "- If you include a non-allowed name, evidence must quote the exact mention and briefly justify why it is clearly a satellite/platform/sensor.\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "satellites": [list of zero or more strings],\n'
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

        batch_size = 50
        max_workers = 64
        last_id = 0

        with tqdm(total=total) as pbar:
            while True:
                q = (
                    select(Publication)
                    .where(
                        Publication.satellite_type.is_(None),
                        Publication.id > last_id,
                    )
                    .order_by(Publication.id)
                    .limit(batch_size)
                )
                pubs_batch = session.execute(q).scalars().all()
                if not pubs_batch:
                    break

                last_id = pubs_batch[-1].id

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
                            results[pub_id] = (
                                "ERROR",
                                "call_model returned None",
                            )
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
                                evidence = (
                                    f"Model classified as {generic_label} but did not provide evidence; "
                                    "abstract likely contains generic remote sensing language."
                                )
                            else:
                                evidence = (
                                    f"Model classified as {generic_label} and did not detect satellite "
                                    "or remote sensing related terms."
                                )

                        results[pub_id] = (sat_value, evidence)
                        pbar.update(1)

                for pub in pubs_batch:
                    sat_type, ev = results[pub.id]
                    pub.satellite_type = sat_type
                    pub.type_evidence = ev

                session.commit()


if __name__ == "__main__":
    main()
