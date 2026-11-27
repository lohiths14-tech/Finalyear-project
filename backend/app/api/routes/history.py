"""
Scan history endpoints — powers the Historical Audit tab in the frontend.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.database import get_db
from ...db.models import ScanResult

router = APIRouter()


# ── Pydantic response schemas ──────────────────────────────────────────────


class ScanSummary(BaseModel):
    id: str
    scan_type: str
    filename: str
    disease_type: str
    severity: float
    confidence: float
    spread_risk: str | None
    treatment: str
    timestamp: str


class ScanDetail(ScanSummary):
    full_results: dict


# ── Routes ─────────────────────────────────────────────────────────────────


@router.get("/history", response_model=list[ScanSummary])
async def list_scans(
    limit: int = Query(50, ge=1, le=200),
    scan_type: str | None = Query(None, pattern="^(field|leaf)$"),
    db: AsyncSession = Depends(get_db),
):
    """Return recent scans, newest first."""
    stmt = select(ScanResult).order_by(desc(ScanResult.timestamp)).limit(limit)
    if scan_type:
        stmt = stmt.where(ScanResult.scan_type == scan_type)
    rows = (await db.execute(stmt)).scalars().all()
    return [
        ScanSummary(
            id=r.id,
            scan_type=r.scan_type,
            filename=r.filename,
            disease_type=r.disease_type,
            severity=r.severity,
            confidence=r.confidence,
            spread_risk=r.spread_risk,
            treatment=r.treatment,
            timestamp=r.timestamp.isoformat() if r.timestamp else "",
        )
        for r in rows
    ]


@router.get("/history/{scan_id}", response_model=ScanDetail)
async def get_scan(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Return full results for a specific scan."""
    row = await db.get(ScanResult, scan_id)
    if not row:
        raise HTTPException(404, f"Scan '{scan_id}' not found.")
    return ScanDetail(
        id=row.id,
        scan_type=row.scan_type,
        filename=row.filename,
        disease_type=row.disease_type,
        severity=row.severity,
        confidence=row.confidence,
        spread_risk=row.spread_risk,
        treatment=row.treatment,
        timestamp=row.timestamp.isoformat() if row.timestamp else "",
        full_results=json.loads(row.full_results),
    )


@router.delete("/history/{scan_id}", status_code=204)
async def delete_scan(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a scan record."""
    row = await db.get(ScanResult, scan_id)
    if not row:
        raise HTTPException(404, f"Scan '{scan_id}' not found.")
    await db.delete(row)
    await db.commit()
