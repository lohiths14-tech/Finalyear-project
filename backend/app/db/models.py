"""
ORM model — persists every scan result to SQLite.
"""

from datetime import datetime, timezone
import uuid

from sqlalchemy import Column, DateTime, Float, String, Text

from .database import Base


class ScanResult(Base):
    __tablename__ = "scans"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_type = Column(String, nullable=False)  # 'field' | 'leaf'
    filename = Column(String, nullable=False)
    disease_type = Column(String, nullable=False)
    severity = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    spread_risk = Column(String, nullable=True)  # field-only
    treatment = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    full_results = Column(Text, nullable=False)  # JSON blob

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
