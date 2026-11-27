"""
AgriPulse 2.0 — Agronomy Advisory & Dosage Calculator API
Provides ICAR/IRRI compliant chemical dosages, knapsack sprayer tank breakdowns, and expert consultations.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...core.agronomy_engine import (
    calculate_agronomic_dosage,
)

router = APIRouter(prefix="/agronomy", tags=["Agronomy & Advisory"])


class DosageRequest(BaseModel):
    disease_type: str = Field(..., description="Disease name, e.g., 'Rice Blast'")
    severity: float = Field(..., ge=0.0, le=100.0, description="Severity percentage (0-100)")
    land_area: float = Field(1.0, gt=0.0, description="Size of the land plot")
    area_unit: str = Field("acre", description="Unit of land (acre, hectare, guntha, bigha, cent)")
    sprayer_tank_capacity_l: float = Field(16.0, gt=0.0, description="Capacity of knapsack sprayer in liters")
    growth_stage: str = Field("Tillering", description="Crop growth stage (Seedling, Tillering, Panicle Initiation, Flowering, Maturity)")
    lat: float | None = Field(None, description="Field latitude for live Open-Meteo weather spray verification")
    lon: float | None = Field(None, description="Field longitude for live Open-Meteo weather spray verification")


class ConsultRequest(BaseModel):
    disease_type: str
    severity: float = Field(..., ge=0.0, le=100.0)
    question: str = Field(..., min_length=2, description="Agronomic question from farmer / advisor")


class PrescriptionRequest(BaseModel):
    farmer_name: str | None = "Field Farmer"
    location_name: str | None = "Agro-Ecological Zone"
    disease_type: str
    severity: float
    confidence: float
    land_area: float = 1.0
    area_unit: str = "acre"
    growth_stage: str = "Tillering"


@router.post("/calculate-dosage")
async def get_dosage(req: DosageRequest):
    """
    Calculate ICAR compliant chemical/bio dosage, water dilution requirements, knapsack sprayer pump count,
    and evaluate real-time Open-Meteo weather spray safety.
    """
    try:
        weather_data = None
        if req.lat is not None and req.lon is not None:
            from ...services.external_apis import external_api_service

            weather_data = await external_api_service.get_weather(
                req.lat, req.lon
            )

        dosage_info = calculate_agronomic_dosage(
            disease_type=req.disease_type,
            severity=req.severity,
            land_area=req.land_area,
            area_unit=req.area_unit,
            sprayer_tank_capacity_l=req.sprayer_tank_capacity_l,
            growth_stage=req.growth_stage,
            weather_data=weather_data,
        )
        return dosage_info
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate dosage: {str(e)}"
        )


@router.get("/ai-status")
async def get_ai_status():
    """
    Get active Agronomic Advisory Engine status and intelligence capabilities.
    """
    from ...services.genai_service import genai_service
    return genai_service.get_provider_status()


@router.post("/consult")
async def consult(req: ConsultRequest):
    """
    Consult the ICAR/IRRI knowledge engine with a specific question grounded in verified agronomic protocols.
    """
    try:
        from ...services.genai_service import genai_service
        res = await genai_service.chat_with_agronomist(
            question=req.question,
            disease_type=req.disease_type,
            severity=req.severity,
        )
        return {
            "disease_type": req.disease_type,
            "severity": req.severity,
            "question": req.question,
            "advisory_markdown": res["answer"],
            "provider": res.get("provider"),
            "model": res.get("model"),
            "grounding_source": res.get("grounding_source"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agronomy consultation failed: {str(e)}")


@router.post("/generate-prescription")
async def generate_prescription(req: PrescriptionRequest):
    """
    Generates a structured, printable ICAR Agronomic Advisory Prescription.
    """
    dosage = calculate_agronomic_dosage(
        disease_type=req.disease_type,
        severity=req.severity,
        land_area=req.land_area,
        area_unit=req.area_unit,
        growth_stage=req.growth_stage,
    )

    prescription_id = f"ICAR-AGRI-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

    return {
        "prescription_id": prescription_id,
        "issued_at": datetime.now(UTC).isoformat(),
        "standards_compliance": "ICAR-IRRI Package of Practices (PoP) 2024-2026",
        "patient_field": {
            "farmer_name": req.farmer_name,
            "location": req.location_name,
            "land_area": f"{req.land_area} {req.area_unit} ({dosage['normalized_acres']} acres)",
            "crop_growth_stage": req.growth_stage,
        },
        "diagnosis": {
            "detected_pathogen": dosage["pathogen"],
            "disease_name": dosage["disease_name"],
            "severity_score": f"{dosage['severity_pct']}%",
            "model_confidence": f"{req.confidence:.1f}%",
        },
        "treatment_plan": {
            "primary_chemical_spray": dosage["primary_treatment"],
            "water_required_liters": dosage["water_volume_liters"],
            "knapsack_tanks_required": f"{dosage['total_knapsack_tanks']} tanks (16L each)",
            "chemical_per_tank": dosage["primary_treatment"]["quantity_per_tank"],
            "biological_alternative": dosage["biological_alternative"],
            "cultural_measures": dosage["cultural_preventive_action"],
            "spray_timing": dosage["spray_schedule"],
            "safety_ppe": dosage["safety_ppe_checklist"],
        },
    }
