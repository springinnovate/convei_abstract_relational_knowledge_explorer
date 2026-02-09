from sqlalchemy import create_engine, text
from models import Base

DB_PATH = "2025_11_09_researchgate.sqlite"
ENGINE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(ENGINE_URL, future=True)

with engine.begin() as conn:
    conn.execute(text("PRAGMA foreign_keys=ON"))

Base.metadata.create_all(engine)
