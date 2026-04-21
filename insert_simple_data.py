from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from models import Base, DataType, Satellite


def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def upsert_values(session, model, values, field="name"):
    values = [v.strip() for v in values if v and v.strip()]
    existing = set(session.scalars(select(getattr(model, field))).all())
    missing = [model(**{field: v}) for v in values if v not in existing]
    session.add_all(missing)
    session.commit()
    return missing


db_path = "2025_11_09_researchgate.sqlite"

engine = create_engine(
    f"sqlite:///{db_path}",
    future=True,
    connect_args={"timeout": 60},
)
event.listen(engine, "connect", _set_sqlite_pragma)

Base.metadata.create_all(engine)

satellite_values = [
    "Sentinel-",
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

data_type_values = [
    "aerial imagery",
    "aerial photography",
    "optical imagery",
    "airborne sensor*",
    "spaceborne",
    "space-based",
    "spacecraft sensor*",
    "multispectral",
    "hyperspectral",
    "SAR",
    "synthetic aperture radar",
    "LiDAR",
    "radar imagery",
    "microwave remote sensing",
    "thermal infrared",
]

with Session(engine) as session:
    inserted_satellites = upsert_values(session, Satellite, satellite_values)
    inserted_data_types = upsert_values(session, DataType, data_type_values)

    print("Inserted satellites:", [row.name for row in inserted_satellites])
    print("Inserted data types:", [row.name for row in inserted_data_types])
