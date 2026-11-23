"""
AgriPulse 2.0 — Agronomic Advisory & ICAR/IRRI Knowledge Engine
Grounded in ICAR (Indian Council of Agricultural Research) and IRRI Rice Knowledge Bank guidelines.
Includes FRAC/IRAC anti-resistance codes, Pre-Harvest Intervals (PHI), growth stages, and knapsack tank dosage calculations.
"""

from typing import Dict, Any, List, Optional
import math

# Unit conversion to Acres
AREA_CONVERSIONS_TO_ACRES = {
    "acre": 1.0,
    "acres": 1.0,
    "hectare": 2.47105,
    "hectares": 2.47105,
    "ha": 2.47105,
    "guntha": 0.025,       # 1 acre = 40 gunthas
    "gunthas": 0.025,
    "bigha": 0.625,        # 1 bigha = ~0.625 acres (standard northern/central)
    "bighas": 0.625,
    "cent": 0.01,          # 1 acre = 100 cents (South India)
    "cents": 0.01,
    "sq_meter": 0.000247105,
    "sq_ft": 0.0000229568,
}

AGRONOMIC_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "Rice Blast": {
        "pathogen": "Magnaporthe oryzae (fungus)",
        "optimal_conditions": "High relative humidity (>90%), leaf wetness duration >12h, temperatures 20-28°C, excess nitrogen (>120 kg N/ha).",
        "primary_chemical": {
            "name": "Tricyclazole 75% WP",
            "dose_per_liter": 0.6,          # grams per liter
            "unit": "g",
            "frac_code": "FRAC 16.1 (Melanin Biosynthesis Inhibitor)",
            "phi_days": 30,
            "toxicity_class": "Class III (Slightly Hazardous)",
            "alt_name": "Isoprothiolane 40% EC @ 1.5 ml/L",
        },
        "secondary_chemical": {
            "name": "Kasugamycin 3% SL",
            "dose_per_liter": 2.5,          # ml per liter
            "unit": "ml",
            "frac_code": "FRAC 24 (Hexose Aminoglycoside)",
            "phi_days": 21,
            "toxicity_class": "Class III (Slightly Hazardous)",
        },
        "chemical_control": "1. Spray Tricyclazole 75% WP @ 0.6 g/L water (120 g in 200 L/acre) or Isoprothiolane 40% EC @ 1.5 ml/L.\n2. In severe cases, Kasugamycin 3% SL @ 2.5 ml/L at 10-day intervals.\n3. Spray in early morning or late afternoon for maximum canopy retention.",
        "biological_control": "Foliar spray of Pseudomonas fluorescens (Pf1) @ 10 g/L or Bacillus subtilis @ 5 g/L seed treatment + foliar spray.",
        "cultural_practices": "1. Avoid split nitrogen top-dressing during humid overcast spells.\n2. Maintain 2-3 cm shallow water level.\n3. Use blast-resistant cultivars (e.g., IR64-Sub1, Swarna-Sub1, Improved Samba Mahsuri).",
        "economic_threshold": "1-2% leaf area infected during tillering; >5% infected neck blast flag leaves.",
        "spray_schedule": "First spray at first visible spindle lesions; second spray at panicle emergence (heading) stage.",
        "growth_stage_advice": {
            "Seedling": "Apply seed dressing with Carbendazim 2g/kg or Trichoderma viride 4g/kg.",
            "Tillering": "Spray Tricyclazole 75% WP @ 0.6 g/L upon first spindle lesion appearance.",
            "Panicle Initiation": "Crucial protective spray to prevent destructive neck blast phase.",
            "Flowering": "Apply Isoprothiolane 40% EC @ 1.5 ml/L if overcast rainy weather persists.",
            "Maturity": "Do not spray if within 30 days of harvest (PHI compliance).",
        },
        "ppe_checklist": [
            "Chemical-resistant nitrile gloves",
            "N95 / FFP2 particulate face mask",
            "Eye protection goggles",
            "Long-sleeved shirt, pants, and gumboots",
            "Wash hands and face with soap immediately post-spraying",
        ],
    },
    "Brown Spot": {
        "pathogen": "Bipolaris oryzae / Cochliobolus miyabeanus (fungus)",
        "optimal_conditions": "Nutrient-deficient soils (low K, low Si, low micronutrients), moisture stress, cloudy days with 25-30°C.",
        "primary_chemical": {
            "name": "Mancozeb 75% WP",
            "dose_per_liter": 2.0,          # grams per liter
            "unit": "g",
            "frac_code": "FRAC M03 (Multi-site Dithiocarbamate)",
            "phi_days": 14,
            "toxicity_class": "Class IV (Practically Non-Toxic)",
            "alt_name": "Propiconazole 25% EC @ 1.0 ml/L",
        },
        "secondary_chemical": {
            "name": "Propiconazole 25% EC",
            "dose_per_liter": 1.0,          # ml per liter
            "unit": "ml",
            "frac_code": "FRAC 3 (DMI Triazole)",
            "phi_days": 21,
            "toxicity_class": "Class III (Slightly Hazardous)",
        },
        "chemical_control": "1. Spray Mancozeb 75% WP @ 2.0 g/L or Propiconazole 25% EC @ 1.0 ml/L at boot-leaf stage.\n2. Edifenphos 50% EC @ 1.0 ml/L or Hexaconazole 5% EC @ 2.0 ml/L.",
        "biological_control": "Seed biopriming with Trichoderma viride @ 4 g/kg seed + Pseudomonas fluorescens @ 10 g/L foliar.",
        "cultural_practices": "1. Apply recommended doses of Muriate of Potash (MOP) in two splits (basal + panicle initiation).\n2. Apply Calcium Silicate slag @ 2 tonnes/ha in silicon-deficient soils to thicken silica cells in foliage.\n3. Ensure adequate irrigation to avoid drought stress.",
        "economic_threshold": "2-5 circular brown spots per leaf on >10% of field tillers.",
        "spray_schedule": "Apply at late tillering and repeat 15 days later if cloud cover persists.",
        "growth_stage_advice": {
            "Seedling": "Treat seeds with Thiram or Captan @ 2g/kg.",
            "Tillering": "Apply MOP top-dressing (25 kg/acre) to correct potassium deficiency.",
            "Panicle Initiation": "Spray Propiconazole 25% EC @ 1.0 ml/L to protect flag leaf vigor.",
            "Flowering": "Maintain balanced moisture; avoid soil drying cracks.",
            "Maturity": "Check grains for discoloration; harvest at 20% moisture.",
        },
        "ppe_checklist": [
            "Nitrile gloves",
            "Protective face shield or mask",
            "Gumboots",
            "Wash spray tank thoroughly away from water bodies",
        ],
    },
    "Bacterial Blight": {
        "pathogen": "Xanthomonas oryzae pv. oryzae (bacterium)",
        "optimal_conditions": "Heavy rainfall, high humidity, cyclonic winds creating leaf wounds, high temperature 25-34°C.",
        "primary_chemical": {
            "name": "Copper Oxychloride 50% WP + Streptocycline",
            "dose_per_liter": 2.5,          # g/L (COC) + 0.1 g/L (Streptocycline)
            "unit": "g",
            "frac_code": "FRAC M01 (Inorganic Copper) + FRAC 25 (Glucopyranosyl Antibiotic)",
            "phi_days": 10,
            "toxicity_class": "Class III (Slightly Hazardous)",
            "alt_name": "Copper Hydroxide 77% WP @ 2.0 g/L",
        },
        "secondary_chemical": {
            "name": "Copper Hydroxide 77% WP",
            "dose_per_liter": 2.0,
            "unit": "g",
            "frac_code": "FRAC M01 (Inorganic Copper)",
            "phi_days": 7,
            "toxicity_class": "Class III (Slightly Hazardous)",
        },
        "chemical_control": "1. Spray Copper Oxychloride 50% WP @ 2.5 g/L mixed with Streptocycline @ 0.1 g/L (1 g in 10 L water).\n2. Copper Hydroxide 77% WP @ 2.0 g/L.",
        "biological_control": "Foliar spray of antagonistic bacteriophages or Pseudomonas fluorescens formulation @ 5 g/L.",
        "cultural_practices": "1. Strictly avoid clipped seedling transplanting which creates bacterial entry ports.\n2. Drain standing water from infected plots into sumps (do NOT drain into neighboring uninfected fields).\n3. Delay nitrogen fertilizer application until leaf margins dry and lesions turn straw-colored.",
        "economic_threshold": "Milky bacterial ooze or yellow wavy margins on >5% of hill crowns.",
        "spray_schedule": "Spray immediately at first symptom appearance, repeat after 7-10 days if high winds/rain continue.",
        "growth_stage_advice": {
            "Seedling": "Do not clip leaf tips before transplanting.",
            "Tillering": "Suspend urea application immediately upon noticing wavy yellow margins.",
            "Panicle Initiation": "Spray Copper Oxychloride + Streptocycline during morning hours.",
            "Flowering": "Drain excess water; maintain strict field hygiene.",
            "Maturity": "Ensure clean harvesting machinery to prevent inoculant spread.",
        },
        "ppe_checklist": [
            "Protective face mask and goggles",
            "Full protective body apron / coverall",
            "Nitrile rubber gloves",
        ],
    },
    "Leaf Scald": {
        "pathogen": "Microdochium oryzae / Gerlachia oryzae (fungus)",
        "optimal_conditions": "High rainfall, dense planting canopy with low airflow, prolonged morning dew.",
        "primary_chemical": {
            "name": "Carbendazim 50% WP",
            "dose_per_liter": 1.0,          # g/L
            "unit": "g",
            "frac_code": "FRAC 1 (MBC - Methyl Benzimidazole Carbamate)",
            "phi_days": 21,
            "toxicity_class": "Class III (Slightly Hazardous)",
            "alt_name": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1.0 ml/L",
        },
        "secondary_chemical": {
            "name": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC",
            "dose_per_liter": 1.0,
            "unit": "ml",
            "frac_code": "FRAC 11 (QoI) + FRAC 3 (DMI)",
            "phi_days": 14,
            "toxicity_class": "Class III (Slightly Hazardous)",
        },
        "chemical_control": "1. Spray Carbendazim 50% WP @ 1.0 g/L or Benomyl 50% WP @ 1.0 g/L.\n2. Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1.0 ml/L.",
        "biological_control": "Seed dressing with Trichoderma harzianum @ 5 g/kg seed.",
        "cultural_practices": "1. Maintain 20×15 cm spacing to enhance micro-climate ventilation.\n2. Clean field bunds and destroy wild grass hosts (Echinochloa colona).\n3. Apply balanced fertilizers with adequate phosphorus and potassium.",
        "economic_threshold": "Zonate scald lesions on >15% of middle canopy leaves.",
        "spray_schedule": "Apply at panicle initiation stage and repeat at 50% flowering if wet weather persists.",
        "growth_stage_advice": {
            "Tillering": "Thin out overcrowded hills to improve sunlight penetration.",
            "Panicle Initiation": "Apply Carbendazim 50% WP @ 1.0 g/L.",
            "Flowering": "Monitor upper flag leaves for chevron scald patterns.",
            "Maturity": "Ensure timely harvest to prevent lodging.",
        },
        "ppe_checklist": [
            "Nitrile chemical gloves",
            "Respirator or particulate mask",
            "Eye goggles",
        ],
    },
    "Healthy": {
        "pathogen": "None detected",
        "optimal_conditions": "Optimal agronomic state.",
        "primary_chemical": {
            "name": "No chemical fungicide required",
            "dose_per_liter": 0.0,
            "unit": "g",
            "frac_code": "N/A",
            "phi_days": 0,
            "toxicity_class": "Safe",
            "alt_name": "Prophylactic bio-stimulants only",
        },
        "secondary_chemical": None,
        "chemical_control": "No synthetic chemical fungicide or bactericide is needed. Prophylactic chemical spraying is discouraged to preserve beneficial parasitoids (Trichogramma) and predators (mirid bugs).",
        "biological_control": "Apply liquid biofertilizers (Azospirillum and Phosphobacteria @ 2 L/ha) to enhance root vigor and nutrient uptake.",
        "cultural_practices": "1. Maintain 2-5 cm alternate wetting and drying (AWD) water regime.\n2. Monitor pest/disease traps weekly.\n3. Apply balanced N-P-K (100:50:50 kg/ha) with micronutrient zinc sulphate @ 25 kg/ha basal.",
        "economic_threshold": "N/A — crop vigor is normal.",
        "spray_schedule": "Continue weekly routine scouting.",
        "growth_stage_advice": {
            "Seedling": "Maintain shallow 2 cm water layer.",
            "Tillering": "Ensure weed-free plot and apply second split of N & K.",
            "Panicle Initiation": "Maintain continuous 5 cm water submergence.",
            "Flowering": "Avoid water stress during 100% flowering.",
            "Maturity": "Drain field 10 days prior to harvest for uniform grain drying.",
        },
        "ppe_checklist": [
            "Standard farm work attire",
            "Sun hat and clean boots",
        ],
    },
}


def calculate_agronomic_dosage(
    disease_type: str,
    severity: float,
    land_area: float = 1.0,
    area_unit: str = "acre",
    sprayer_tank_capacity_l: float = 16.0,
    growth_stage: str = "Tillering",
) -> Dict[str, Any]:
    """
    Computes exact ICAR compliant chemical quantities, dilution water, and knapsack sprayer tank counts.
    """
    # Normalize disease
    disease_key = "Healthy"
    for k in AGRONOMIC_KNOWLEDGE:
        if k.lower() in disease_type.lower():
            disease_key = k
            break

    info = AGRONOMIC_KNOWLEDGE.get(disease_key, AGRONOMIC_KNOWLEDGE["Healthy"])
    
    # Convert area to acres
    unit_norm = area_unit.lower().strip()
    multiplier = AREA_CONVERSIONS_TO_ACRES.get(unit_norm, 1.0)
    area_acres = max(0.01, land_area * multiplier)

    # Standard water volume: ~200 Liters / Acre (equivalent to 500 L / Hectare)
    water_volume_liters = round(area_acres * 200.0, 1)
    
    # Knapsack sprayer calculations (e.g. 16L standard manual / battery backpack pump)
    tank_capacity = max(5.0, sprayer_tank_capacity_l)
    total_tanks = math.ceil(water_volume_liters / tank_capacity)

    primary_chem = info.get("primary_chemical", {})
    dose_per_l = primary_chem.get("dose_per_liter", 0.0)
    chem_unit = primary_chem.get("unit", "g")
    
    total_chemical_qty = round(dose_per_l * water_volume_liters, 2)
    per_tank_chemical_qty = round(dose_per_l * tank_capacity, 2)

    # Growth stage specific notes
    stage_notes = info.get("growth_stage_advice", {}).get(
        growth_stage, f"Monitor crop vigor closely during {growth_stage}."
    )

    return {
        "disease_name": disease_key,
        "pathogen": info["pathogen"],
        "severity_pct": round(severity, 1),
        "growth_stage": growth_stage,
        "growth_stage_advice": stage_notes,
        "input_land_area": land_area,
        "input_area_unit": area_unit,
        "normalized_acres": round(area_acres, 3),
        "water_volume_liters": water_volume_liters,
        "sprayer_tank_capacity_liters": tank_capacity,
        "total_knapsack_tanks": total_tanks,
        "primary_treatment": {
            "name": primary_chem.get("name", "N/A"),
            "dose_per_liter": f"{dose_per_l} {chem_unit}/L",
            "total_quantity_required": f"{total_chemical_qty} {chem_unit}",
            "quantity_per_tank": f"{per_tank_chemical_qty} {chem_unit} per {tank_capacity}L tank",
            "frac_code": primary_chem.get("frac_code", "N/A"),
            "pre_harvest_interval_days": primary_chem.get("phi_days", 0),
            "toxicity_class": primary_chem.get("toxicity_class", "N/A"),
            "alternative": primary_chem.get("alt_name", "None"),
        },
        "biological_alternative": info.get("biological_control"),
        "cultural_preventive_action": info.get("cultural_practices"),
        "economic_threshold": info.get("economic_threshold"),
        "spray_schedule": info.get("spray_schedule"),
        "safety_ppe_checklist": info.get("ppe_checklist", []),
    }


def consult_agronomist(disease_type: str, severity: float, question: str) -> str:
    """
    Generate evidence-backed ICAR/IRRI agronomic consultation response.
    """
    q_lower = question.lower()
    
    # Normalize disease key
    disease_key = "Healthy"
    for k in AGRONOMIC_KNOWLEDGE:
        if k.lower() in disease_type.lower():
            disease_key = k
            break

    info = AGRONOMIC_KNOWLEDGE.get(disease_key, AGRONOMIC_KNOWLEDGE["Healthy"])
    sev_str = f"Estimated severity is currently **{severity:.1f}%**"
    
    # Match query intent
    if any(w in q_lower for w in ["chemical", "spray", "fungicide", "medicine", "drug", "dose", "dosage", "active ingredient"]):
        return (
            f"### 🧪 Targeted Chemical Management for {disease_key}\n\n"
            f"{sev_str}.\n\n"
            f"**Recommended Treatment (ICAR/IRRI Protocol):**\n"
            f"{info['chemical_control']}\n\n"
            f"**Application Guidelines:**\n"
            f"- {info['spray_schedule']}\n"
            f"- Use a hollow cone nozzle calibrated to 200 L of water per acre (500 L/ha).\n"
            f"- Add a non-ionic surfactant/sticker (0.5 ml/L) during rainy spells to prevent wash-off.\n"
            f"- **FRAC Resistance Code:** {info.get('primary_chemical', {}).get('frac_code', 'N/A')}\n"
            f"- **Pre-Harvest Interval (PHI):** {info.get('primary_chemical', {}).get('phi_days', 0)} days."
        )
    elif any(w in q_lower for w in ["bio", "organic", "natural", "eco", "safe", "herbal"]):
        return (
            f"### 🌿 Biological & Organic Control for {disease_key}\n\n"
            f"{sev_str}.\n\n"
            f"**Bio-Agent Protocol:**\n"
            f"{info['biological_control']}\n\n"
            f"**Supporting Cultural Measures:**\n"
            f"{info['cultural_practices']}\n\n"
            f"*Note: Biological controls work best as preventatives or when severity is under 15%. For outbreaks above 30%, integrate with targeted fungicides.*"
        )
    elif any(w in q_lower for w in ["prevent", "spread", "culture", "water", "irrigation", "fertilizer", "nitrogen", "urea"]):
        return (
            f"### 🌾 Cultural Practices & Spread Prevention for {disease_key}\n\n"
            f"{sev_str}.\n\n"
            f"**Agronomic Recommendations:**\n"
            f"{info['cultural_practices']}\n\n"
            f"**Disease Environmental Drivers:**\n"
            f"- {info['optimal_conditions']}\n"
            f"- Economic Action Threshold: {info['economic_threshold']}"
        )
    elif any(w in q_lower for w in ["cost", "yield", "damage", "loss", "danger", "threshold"]):
        return (
            f"### 📊 Economic Impact & Risk Assessment for {disease_key}\n\n"
            f"{sev_str}.\n\n"
            f"**Economic Action Threshold (ETL):**\n"
            f"- {info['economic_threshold']}\n\n"
            f"**Yield Risk Profile:**\n"
            f"- At {severity:.1f}% severity, potential yield impact is estimated between "
            f"{'0-3%' if severity < 15 else '5-15%' if severity < 40 else '20-40%'}.\n"
            f"- Immediate intervention is {'optional (monitor closely)' if severity < 15 else 'strongly recommended' if severity < 40 else 'critical to prevent crop lodging and total panicle loss'}."
        )
    else:
        return (
            f"### 🧑‍🌾 Expert Agronomic Advisory: {disease_key}\n\n"
            f"{sev_str}.\n\n"
            f"**1. Chemical Options (ICAR Standards):**\n{info['chemical_control']}\n\n"
            f"**2. Biological & Organic Measures:**\n{info['biological_control']}\n\n"
            f"**3. Cultural & Water Management:**\n{info['cultural_practices']}\n\n"
            f"**4. Spray Timing & Safety:**\n{info['spray_schedule']}"
        )
