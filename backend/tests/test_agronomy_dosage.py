"""
Unit and API integration tests for AgriPulse 2.0 Agronomic Decision Engine & ICAR Dosage Calculator.
"""

from httpx import ASGITransport, AsyncClient
import pytest

from app.core.agronomy_engine import (
    calculate_agronomic_dosage,
    consult_agronomist,
)
from app.main import app


def test_agronomy_dosage_calculation_acres():
    """Verify ICAR standard dosage calculations for 2.0 acres of Rice Blast."""
    res = calculate_agronomic_dosage(
        disease_type="Rice Blast",
        severity=45.0,
        land_area=2.0,
        area_unit="acre",
        sprayer_tank_capacity_l=16.0,
        growth_stage="Panicle Initiation",
    )

    assert res["disease_name"] == "Rice Blast"
    assert res["normalized_acres"] == 2.0
    # 2 acres * 200 L/acre = 400 L
    assert res["water_volume_liters"] == 400.0
    # ceil(400 / 16) = 25 tanks
    assert res["total_knapsack_tanks"] == 25
    # 0.6 g/L * 400 L = 240 g
    assert "240.0 g" in res["primary_treatment"]["total_quantity_required"]
    assert "FRAC 16.1" in res["primary_treatment"]["frac_code"]
    assert res["primary_treatment"]["pre_harvest_interval_days"] == 30


def test_agronomy_dosage_unit_conversions():
    """Verify unit conversions for Guntha, Hectare, Bigha, and Cent."""
    # 40 gunthas = 1 acre
    res_guntha = calculate_agronomic_dosage(
        disease_type="Brown Spot",
        severity=25.0,
        land_area=40.0,
        area_unit="guntha",
    )
    assert res_guntha["normalized_acres"] == 1.0
    assert res_guntha["water_volume_liters"] == 200.0

    # 1 hectare = 2.471 acres -> ~494.2 L
    res_ha = calculate_agronomic_dosage(
        disease_type="Bacterial Blight",
        severity=30.0,
        land_area=1.0,
        area_unit="hectare",
    )
    assert res_ha["normalized_acres"] == 2.471
    assert res_ha["water_volume_liters"] == 494.2


def test_consult_agronomist_queries():
    """Verify evidence-backed ICAR agronomic consultations for various intents."""
    ans_chem = consult_agronomist("Rice Blast", 50.0, "What chemical spray should I use?")
    assert "Tricyclazole 75% WP" in ans_chem
    assert "FRAC" in ans_chem

    ans_bio = consult_agronomist("Brown Spot", 20.0, "Is there an organic or biological control?")
    assert "Trichoderma" in ans_bio or "Pseudomonas" in ans_bio

    ans_loss = consult_agronomist("Bacterial Blight", 60.0, "What is the expected yield loss?")
    assert "Yield Risk Profile" in ans_loss


@pytest.mark.asyncio
async def test_agronomy_api_endpoints():
    """Test FastAPI agronomy endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test /calculate-dosage
        r_dose = await ac.post("/api/v1/agronomy/calculate-dosage", json={
            "disease_type": "Rice Blast",
            "severity": 35.0,
            "land_area": 1.5,
            "area_unit": "acre",
            "sprayer_tank_capacity_l": 16.0,
            "growth_stage": "Tillering",
        })
        assert r_dose.status_code == 200
        data = r_dose.json()
        assert data["total_knapsack_tanks"] == 19
        assert data["water_volume_liters"] == 300.0

        # Test /generate-prescription
        r_rx = await ac.post("/api/v1/agronomy/generate-prescription", json={
            "farmer_name": "Ramesh Kumar",
            "location_name": "Shivamogga, Karnataka",
            "disease_type": "Rice Blast",
            "severity": 42.0,
            "confidence": 98.4,
            "land_area": 2.0,
            "area_unit": "acre",
            "growth_stage": "Booting",
        })
        assert r_rx.status_code == 200
        rx_data = r_rx.json()
        assert "prescription_id" in rx_data
        assert rx_data["patient_field"]["farmer_name"] == "Ramesh Kumar"
        assert "ICAR-IRRI" in rx_data["standards_compliance"]
