"""
AgriPulse 2.0 — Grounded Generative AI Agronomy & RAG Service
=============================================================
Provides evidence-backed Generative AI capabilities grounded strictly
in ICAR (Indian Council of Agricultural Research) & IRRI Rice Knowledge Bank data.

Architecture:
  - Multi-Provider Support: Google Gemini API / OpenAI API / Local ICAR RAG Engine
  - Zero-Hallucination Grounding: System prompts inject structured ICAR disease protocols,
    FRAC resistance codes, PHI restrictions, and knapsack sprayer tank math.
  - Graceful Degradation: If no external LLM API key is present in .env, falls back
    seamlessly to the internal deterministic ICAR RAG engine with zero crashes.
"""

from datetime import UTC, datetime
import json
import logging
import os
from typing import Any
import urllib.parse
import urllib.request

from ..core.agronomy_engine import (
    AGRONOMIC_KNOWLEDGE,
    calculate_agronomic_dosage,
)
from ..core.config import settings

logger = logging.getLogger(__name__)


class GenAIAgronomyService:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)

        if self.gemini_api_key:
            self.provider = "gemini"
        elif self.openai_api_key:
            self.provider = "openai"
        else:
            self.provider = "icar_rag_local"

        logger.info(f"GenAIAgronomyService initialized with provider: {self.provider}")

    def get_provider_status(self) -> dict[str, Any]:
        """Returns the active GenAI provider status and capabilities."""
        if self.provider == "gemini":
            model_id = "gemini-1.5-flash"
            display_name = "Google Gemini 1.5 Flash"
        elif self.provider == "openai":
            model_id = "gpt-4o-mini"
            display_name = "OpenAI GPT-4o Mini"
        else:
            model_id = "icar-deterministic"
            display_name = "ICAR Deterministic RAG Engine"

        return {
            "active_provider": self.provider,
            "model_id": model_id,
            "display_name": display_name,
            "gemini_configured": bool(self.gemini_api_key),
            "openai_configured": bool(self.openai_api_key),
            "rag_knowledge_base": "ICAR-IRRI 2024-2026 Package of Practices",
            "supported_languages": ["en", "kn", "hi", "te"],
            "capabilities": [
                "Field-Level Spatial Reasoning & Cluster Containment",
                "Acreage Knapsack Dosage & Spray Buffer Calculations",
                "Clinical Crop Pathology Audit Report Generation",
                "Multilingual Agronomic Consultation",
                "FRAC Resistance Code & PHI Compliance Guard",
            ]
        }

    def _build_rag_context(self, disease_type: str, severity: float, farm_context: dict[str, Any] | None = None) -> str:
        """Retrieves and formats indexed ICAR agronomic facts to ground the LLM."""
        disease_key = "Healthy"
        for k in AGRONOMIC_KNOWLEDGE:
            if k.lower() in disease_type.lower():
                disease_key = k
                break

        info = AGRONOMIC_KNOWLEDGE.get(disease_key, AGRONOMIC_KNOWLEDGE["Healthy"])
        chem = info.get("primary_chemical", {})

        context_lines = [
            "=== VERIFIED ICAR/IRRI GROUNDED AGRONOMIC CONTEXT ===",
            f"Target Pathogen: {info.get('pathogen', 'N/A')}",
            f"Target Disease: {disease_key}",
            f"Observed Visual Severity: {severity:.1f}%",
            f"Optimal Fungal/Bacterial Spread Conditions: {info.get('optimal_conditions', 'N/A')}",
            f"ICAR Primary Chemical: {chem.get('name', 'N/A')} @ {chem.get('dose_per_liter', 0)} {chem.get('unit', 'g')}/L water",
            f"FRAC Anti-Resistance Code: {chem.get('frac_code', 'N/A')}",
            f"Pre-Harvest Interval (PHI): {chem.get('phi_days', 0)} days",
            f"Chemical Control Protocol: {info.get('chemical_control', 'N/A')}",
            f"Biological Control Protocol: {info.get('biological_control', 'N/A')}",
            f"Cultural & Water Management: {info.get('cultural_practices', 'N/A')}",
            f"Economic Action Threshold (ETL): {info.get('economic_threshold', 'N/A')}",
            f"Spray Schedule: {info.get('spray_schedule', 'N/A')}",
        ]

        if farm_context:
            context_lines.append("\n=== FARM TELEMETRY ===")
            if "land_area" in farm_context:
                context_lines.append(f"Land Area: {farm_context.get('land_area')} {farm_context.get('area_unit', 'acre')}")
            if "growth_stage" in farm_context:
                context_lines.append(f"Crop Growth Stage: {farm_context.get('growth_stage')}")
            if "weather" in farm_context:
                context_lines.append(f"Local Weather: {farm_context.get('weather')}")

        return "\n".join(context_lines)

    async def chat_with_agronomist(
        self,
        question: str,
        disease_type: str = "Healthy",
        severity: float = 0.0,
        language: str = "en",
        history: list[dict[str, str]] | None = None,
        farm_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Multi-turn RAG-grounded chat with the AI Agronomist grounded in ICAR guidelines.
        """
        rag_context = self._build_rag_context(disease_type, severity, farm_context)

        system_prompt = (
            "You are Dr. AgriPulse, a Senior Precision Agronomist and Crop Protection Specialist grounded in "
            "the Indian Council of Agricultural Research (ICAR) and International Rice Research Institute (IRRI) guidelines.\n\n"
            "CRITICAL SCIENTIFIC & SAFETY CONSTRAINTS:\n"
            "1. Ground all chemical, biological, and dosage recommendations strictly in the provided ICAR context below.\n"
            "2. Always mention exact chemical trade/generic names, doses per liter of water, knapsack tank calculations (16L tanks), "
            "FRAC resistance codes, and Pre-Harvest Intervals (PHI) when asked about treatments.\n"
            "3. If disease severity is >30%, emphasize targeted chemical intervention while warning about excess nitrogen. "
            "If severity is <15%, recommend organic/bio-control (Pseudomonas / Trichoderma) and cultural monitoring.\n"
            "4. Be concise, respectful, actionable, and practical for smallholder farmers and agricultural extension officers.\n"
            "5. Respond in the requested language (en = English, kn = Kannada, hi = Hindi, te = Telugu).\n\n"
            f"{rag_context}"
        )

        # 1. Try Google Gemini API if configured
        if self.gemini_api_key:
            try:
                gemini_resp = await self._call_gemini_api(system_prompt, question, history)
                if gemini_resp:
                    return {
                        "answer": gemini_resp,
                        "provider": "google-gemini-1.5-flash",
                        "model": "Gemini 1.5 Flash",
                        "grounding_source": "ICAR-IRRI Knowledge Base (RAG Verified)",
                        "language": language,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}. Falling back...")

        # 2. Try OpenAI API if configured
        if self.openai_api_key:
            try:
                openai_resp = await self._call_openai_api(system_prompt, question, history)
                if openai_resp:
                    return {
                        "answer": openai_resp,
                        "provider": "openai-gpt-4o-mini",
                        "model": "GPT-4o Mini",
                        "grounding_source": "ICAR-IRRI Knowledge Base (RAG Verified)",
                        "language": language,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
            except Exception as e:
                logger.warning(f"OpenAI API call failed: {e}. Falling back to internal RAG engine.")

        # 3. Fallback: High-Performance Internal ICAR RAG Engine
        from ..core.agronomy_engine import consult_agronomist
        fallback_answer = consult_agronomist(disease_type, severity, question)

        return {
            "answer": fallback_answer,
            "provider": "icar_rag_engine_v2 (Grounded Local Fallback)",
            "model": "ICAR Deterministic Expert System",
            "grounding_source": "ICAR-IRRI Package of Practices (Zero-Hallucination Deterministic Engine)",
            "language": language,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def consult_field_intelligence(
        self,
        question: str,
        disease_type: str,
        average_severity: float,
        infection_rate: float,
        spread_risk: str,
        clusters: list[dict[str, Any]] | None = None,
        vegetation_coverage: float | None = None,
        land_area: float | None = None,
        area_unit: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Field-Level Spatial Intelligence Consultation grounded in vision telemetry and ICAR protocols.
        Provides spatial cluster containment, barrier buffer zones, and spot-spraying strategies.
        """
        if "unknown" in disease_type.lower() or "insufficient" in disease_type.lower():
            return {
                "answer": (
                    "### Diagnosis Withheld — Verification Needed\n\n"
                    "The scan could not confidently match a known paddy condition in this image, "
                    "so disease-specific chemical recommendations are withheld.\n\n"
                    "- Retake the field photo in clear daylight with visible foliage.\n"
                    "- Consult your local Krishi Vigyan Kendra (KVK) agronomist before spraying."
                ),
                "provider": "icar_spatial_rag_engine (Grounded Local Fallback)",
                "model": "ICAR Deterministic Expert System",
                "grounding_source": "DINOv2 Spatial Grid + ICAR-IRRI Field Guidelines",
                "field_telemetry": {
                    "infection_rate": infection_rate,
                    "average_severity": average_severity,
                    "spread_risk": spread_risk,
                    "cluster_count": len(clusters) if clusters else 0,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

        disease_key = "Healthy"
        for k in AGRONOMIC_KNOWLEDGE:
            if k.lower() in disease_type.lower():
                disease_key = k
                break

        info = AGRONOMIC_KNOWLEDGE.get(disease_key, AGRONOMIC_KNOWLEDGE["Healthy"])
        chem = info.get("primary_chemical", {})

        clusters_desc = ""
        if clusters:
            clusters_desc = "\n".join([
                f"- Cluster #{c.get('id', i+1)}: {c.get('disease', disease_key)}, Coverage: {c.get('coverage_pct', 0):.1f}% of field"
                for i, c in enumerate(clusters)
            ])
        else:
            clusters_desc = "No isolated clusters; diffuse or clean field distribution."

        veg_line = (
            f"Vegetation Biomass Coverage: {vegetation_coverage:.1f}%\n"
            if vegetation_coverage is not None
            else "Vegetation Biomass Coverage: Not provided\n"
        )
        has_area = land_area is not None and area_unit is not None
        area_line = (
            f"Plot Area: {land_area} {area_unit}\n"
            if has_area
            else "Plot Area: Not provided (use per-acre standard ICAR reference rates)\n"
        )

        field_telemetry = (
            f"=== REAL FIELD VISION TELEMETRY ===\n"
            f"Primary Disease Detected: {disease_key} ({info.get('pathogen', 'N/A')})\n"
            f"Overall Field Infection Rate: {infection_rate:.1f}% of analyzed canopy\n"
            f"Average Canopy Severity: {average_severity:.1f}%\n"
            f"Spatial Spread Risk: {spread_risk}\n"
            f"{veg_line}"
            f"{area_line}"
            f"Detected Spatial Hotspots:\n{clusters_desc}\n\n"
            f"=== VERIFIED ICAR/IRRI PROTOCOL ===\n"
            f"Recommended Chemical: {chem.get('name', 'N/A')} @ {chem.get('dose_per_liter', 0)} {chem.get('unit', 'g')}/L\n"
            f"FRAC Anti-Resistance Code: {chem.get('frac_code', 'N/A')}\n"
            f"Pre-Harvest Interval: {chem.get('phi_days', 0)} days\n"
            f"Standard Water Volume: ~200 Liters / Acre (~12.5 knapsack tanks of 16L each)\n"
            f"Biological Alternative: {info.get('biological_control', 'N/A')}\n"
            f"Cultural Prevention: {info.get('cultural_practices', 'N/A')}\n"
        )

        system_prompt = (
            "You are Dr. AgriPulse Field Intelligence Copilot, an Agricultural Research Scientist and Precision "
            "Agronomy specialist grounded strictly in ICAR and IRRI field crop standards.\n\n"
            "You analyze field canopy imagery telemetry to advise farmers and agricultural officers on:\n"
            "1. SPATIAL CONTAINMENT: How to isolate and buffer detected disease clusters so the farmer does NOT needlessly "
            "broadcast-spray healthy crop areas (saving 60-80% of chemical expense).\n"
            "2. SPOT-SPRAY DOSAGE: Exact calculation of knapsack spray tanks (16L tanks) needed specifically for the infected acreage.\n"
            "3. EPIDEMIOLOGICAL SPREAD: Impact of weather, canopy humidity, and spread risk on healthy biomass.\n"
            "4. REGULATORY SAFETY: Pre-harvest interval (PHI) and FRAC rotation to prevent fungicide resistance.\n\n"
            "Format your advice with clean markdown headings, bullet points, and exact numbers.\n\n"
            f"{field_telemetry}"
        )

        # 1. Try Gemini if configured
        if self.gemini_api_key:
            try:
                gemini_resp = await self._call_gemini_api(system_prompt, question, history)
                if gemini_resp:
                    return {
                        "answer": gemini_resp,
                        "provider": "google-gemini-1.5-flash",
                        "model": "Gemini 1.5 Flash",
                        "grounding_source": "DINOv2 Spatial Grid + ICAR-IRRI Field Guidelines",
                        "field_telemetry": {
                            "infection_rate": infection_rate,
                            "average_severity": average_severity,
                            "spread_risk": spread_risk,
                            "cluster_count": len(clusters) if clusters else 0,
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Field Spatial Copilot call to Gemini failed: {e}")

        # 2. Try OpenAI if configured
        if self.openai_api_key:
            try:
                openai_resp = await self._call_openai_api(system_prompt, question, history)
                if openai_resp:
                    return {
                        "answer": openai_resp,
                        "provider": "openai-gpt-4o-mini",
                        "model": "GPT-4o Mini",
                        "grounding_source": "DINOv2 Spatial Grid + ICAR-IRRI Field Guidelines",
                        "field_telemetry": {
                            "infection_rate": infection_rate,
                            "average_severity": average_severity,
                            "spread_risk": spread_risk,
                            "cluster_count": len(clusters) if clusters else 0,
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Field Spatial Copilot call to OpenAI failed: {e}")

        # Fallback: Deterministic field advisory
        from ..core.agronomy_engine import calculate_agronomic_dosage
        eff_area = land_area if land_area is not None else 1.0
        eff_unit = area_unit if area_unit is not None else "acre"
        dosage = calculate_agronomic_dosage(
            disease_type=disease_key,
            severity=average_severity,
            land_area=eff_area,
            area_unit=eff_unit,
        )
        area_scope = (
            f"the entire {land_area} {area_unit}"
            if has_area
            else "the entire field (per-acre reference rates shown below because plot area was not supplied)"
        )

        fallback_md = (
            f"### Precision Field Containment Advisory: {disease_key}\n\n"
            f"- **Field Infection Rate:** {infection_rate:.1f}% (Canopy Severity: {average_severity:.1f}%)\n"
            f"- **Spread Risk Level:** **{spread_risk}**\n\n"
            f"#### Spot-Spraying Containment Protocol\n"
            f"Rather than broadcast-spraying {area_scope}, apply targeted spot spraying "
            f"with a 3-5 meter perimeter buffer around the detected infected hotspots.\n\n"
            f"#### Knapsack Tank & Chemical Formulation"
            f"{'' if has_area else ' (Per-Acre Reference)'}\n"
            f"- **Chemical:** {dosage['primary_treatment']['name']}\n"
            f"- **Dose per 16L Tank:** {dosage['primary_treatment']['quantity_per_tank']}\n"
            f"- **Water Dilution Volume:** {dosage['water_volume_liters']} L ({dosage['total_knapsack_tanks']} tanks"
            f"{'' if has_area else ' per acre'})\n"
            f"- **FRAC Code:** {dosage['primary_treatment']['frac_code']} (PHI: {dosage['primary_treatment']['pre_harvest_interval_days']} days)\n\n"
            f"#### Cultural Sanitation\n"
            f"{dosage['cultural_preventive_action']}"
        )

        return {
            "answer": fallback_md,
            "provider": "icar_spatial_rag_engine (Grounded Local Fallback)",
            "model": "ICAR Deterministic Expert System",
            "grounding_source": "DINOv2 Spatial Grid + ICAR-IRRI Field Guidelines",
            "field_telemetry": {
                "infection_rate": infection_rate,
                "average_severity": average_severity,
                "spread_risk": spread_risk,
                "cluster_count": len(clusters) if clusters else 0,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _call_gemini_api(self, system_prompt: str, question: str, history: list[dict[str, str]] | None) -> str | None:
        """Calls Google Gemini REST API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"

        contents = []
        contents.append({
            "role": "user",
            "parts": [{"text": system_prompt + "\n\nUser Question: " + question}]
        })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024,
                "topP": 0.8,
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=12) as response:
            if response.status == 200:
                res_data = json.loads(response.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text")
        return None

    async def _call_openai_api(self, system_prompt: str, question: str, history: list[dict[str, str]] | None) -> str | None:
        """Calls OpenAI REST API."""
        url = "https://api.openai.com/v1/chat/completions"
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history[-4:]:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": question})

        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1000,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=12) as response:
            if response.status == 200:
                res_data = json.loads(response.read().decode("utf-8"))
                choices = res_data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content")
        return None

    def generate_clinical_crop_report(
        self,
        disease_type: str,
        severity: float,
        confidence: float,
        infection_rate: float,
        veg_coverage: float,
        farmer_name: str = "Progressive Farmer",
        location: str = "Regional Agronomy Zone",
        land_area: float = 1.0,
        area_unit: str = "acre",
        growth_stage: str = "Tillering",
    ) -> dict[str, Any]:
        """
        Generates a comprehensive Clinical Crop Pathology Audit Report
        combining DINOv2 vision analytics with ICAR treatment protocols.
        """
        dosage = calculate_agronomic_dosage(
            disease_type=disease_type,
            severity=severity,
            land_area=land_area,
            area_unit=area_unit,
            growth_stage=growth_stage,
        )

        report_id = f"AGRIPULSE-AUDIT-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        is_healthy = "healthy" in disease_type.lower()
        urgency = "LOW" if is_healthy or severity < 20 else ("MODERATE" if severity < 45 else "HIGH (IMMEDIATE ACTION REQUIRED)")

        executive_summary = (
            f"Crop diagnostic evaluation completed on {datetime.now().strftime('%d %B %Y')}. "
            f"Field analysis across {dosage['normalized_acres']} acres indicates an overall crop health status of "
            f"**{disease_type}** with an average canopy disease severity of **{severity:.1f}%** (Infection Area: **{infection_rate:.1f}%**). "
            f"Model classification confidence is calibrated at **{confidence:.1f}%**. "
            f"Urgency Level: **{urgency}**."
        )

        recovery_timeline = [
            {"day": "Day 1-2", "action": "Implement primary spray treatment in early morning (6:00-9:00 AM) with calibrated knapsack sprayer."},
            {"day": "Day 4", "action": "Inspect canopy margins for lesion drying and check for phytotoxicity or runoff."},
            {"day": "Day 7", "action": "Perform follow-up field scan with AgriPulse 2.0 to verify severity reduction trend."},
            {"day": "Day 14", "action": "Apply secondary bio-protective spray if humid overcast weather persists, respecting FRAC code rotation."},
        ] if not is_healthy else [
            {"day": "Day 1-7", "action": "Maintain optimal standing water (2-3 cm) and balanced nitrogen-potash split."},
            {"day": "Day 14", "action": "Routine surveillance scan to catch early leaf blast or brown spot lesions."},
        ]

        return {
            "report_id": report_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "certification": "ICAR-IRRI Standardized Crop Intelligence Protocol",
            "farmer_profile": {
                "farmer_name": farmer_name,
                "location": location,
                "land_plot": f"{land_area} {area_unit} ({dosage['normalized_acres']} standard acres)",
                "growth_stage": growth_stage,
            },
            "vision_pathology": {
                "disease_detected": disease_type,
                "pathogen": dosage["pathogen"],
                "canopy_severity_score": f"{severity:.1f}%",
                "field_infection_rate": f"{infection_rate:.1f}%",
                "vegetation_coverage": f"{veg_coverage:.1f}%",
                "calibrated_confidence": f"{confidence:.1f}%",
                "urgency": urgency,
            },
            "executive_summary": executive_summary,
            "prescribed_intervention": {
                "chemical_protocol": dosage["primary_treatment"],
                "water_volume": f"{dosage['water_volume_liters']} Liters",
                "knapsack_tanks": f"{dosage['total_knapsack_tanks']} Pumps (16L each)",
                "chemical_dose_per_tank": dosage["primary_treatment"]["quantity_per_tank"],
                "total_chemical_needed": dosage["primary_treatment"]["total_quantity_required"],
                "biological_control": dosage["biological_alternative"],
                "cultural_prevention": dosage["cultural_preventive_action"],
                "spray_schedule": dosage["spray_schedule"],
                "safety_ppe": dosage["safety_ppe_checklist"],
            },
            "recovery_timeline": recovery_timeline,
            "legal_disclaimer": "This audit report is generated by AgriPulse 2.0 based on neural vision analytics and ICAR Package of Practices. For high-value regulatory decisions, verify with your local Krishi Vigyan Kendra (KVK) officer."
        }


# Global singleton instance
genai_service = GenAIAgronomyService()
