"""
Async SQLite database — scan result persistence.
"""

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

db_url = os.getenv("DATABASE_URL")
if db_url:
    DATABASE_URL = db_url
else:
    db_dir_env = os.getenv("DB_DIR", "./data")
    if db_dir_env == ":memory:":
        DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    else:
        DB_DIR = Path(db_dir_env)
        DB_DIR.mkdir(parents=True, exist_ok=True)
        DATABASE_URL = f"sqlite+aiosqlite:///{DB_DIR / 'agrivision.db'}"

engine_kwargs = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(DATABASE_URL, **engine_kwargs)
AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
