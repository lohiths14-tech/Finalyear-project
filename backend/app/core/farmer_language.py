"""
Translation layer to convert technical ML outputs into actionable farmer-friendly language.
"""  # noqa: E501

DISEASE_MAPPING = {
    "Healthy": {
        "label": "Crop is Healthy",
        "description": "Your crop looks strong and disease-free.",
        "action": "Keep doing what you are doing. Check again in 7 days.",
    },
    "Rice_Blast": {
        "label": "Rice Blast Fungal Infection",
        "description": "Diamond-shaped greyish spots with brown borders caused by Magnaporthe oryzae fungus.",
        "action": "Apply Tricyclazole 75% WP (0.6 g/L) or Isoprothiolane 40% EC (1.5 mL/L). Avoid excess urea fertilizer.",
    },
    "Brown_Spot": {
        "label": "Brown Spot Disease",
        "description": "Oval brown spots with yellow halos on rice leaves, common in potassium/zinc-deficient soil.",
        "action": "Spray Mancozeb 75% WP (2 g/L) or Propiconazole 25% EC (1 mL/L) and apply balanced potash/zinc nutrition.",
    },
    "Leaf_Scald": {
        "label": "Leaf Scald Fungal Infection",
        "description": "Zonate tan-and-brown water-soaked bands spreading from leaf tips downward.",
        "action": "Spray Benomyl or Carbendazim 50% WP (1 g/L) and improve field drainage and plant spacing.",
    },
    "Bacterial_Blight": {
        "label": "Bacterial Leaf Blight",
        "description": "Water-soaked yellow-orange stripes drying from leaf tips along margins.",
        "action": "Drain standing water temporarily. Avoid spraying during rain; consult ICAR/KVK for Copper Hydroxide bactericide.",
    },
    "Bacterial_Leaf_Streak": {
        "label": "Bacterial Leaf Streak",
        "description": "Narrow dark-green translucent streaks between leaf veins turning yellow-orange.",
        "action": "Avoid excess nitrogen fertilizer, improve drainage, and apply copper oxychloride only if cleared by weather.",
    },
    "Bacterial_Panicle_Blight": {
        "label": "Bacterial Panicle Blight",
        "description": "Grain discoloration and spikelet sterility reducing rice panicle filling.",
        "action": "Drain excess water, avoid high-temperature nitrogen top-dressing, and use disease-free seed next season.",
    },
    "Dead_Heart": {
        "label": "Dead Heart (Yellow Stem Borer)",
        "description": "Central tiller leaf dries and pulls out easily due to stem borer feeding inside the stem.",
        "action": "Clip infested shoots, install pheromone traps (5/acre), or apply Chlorantraniliprole 0.4% GR per ICAR dosage.",
    },
    "Downy_Mildew": {
        "label": "Downy Mildew Infection",
        "description": "Pale yellow streaking and stunted leaf twisting from downy mildew.",
        "action": "Drain waterlogged soil and apply Metalaxyl + Mancozeb (2 g/L) on affected nursery/crop patches.",
    },
    "Rice_Hispa": {
        "label": "Rice Hispa Pest Damage",
        "description": "Parallel white translucent streaks on leaves scraped by thorny blue-black beetles.",
        "action": "Clip and destroy leaf tips containing grubs; spray Azadirachtin (Neem) or Lambda-Cyhalothrin if threshold exceeded.",
    },
    "Rice_Tungro": {
        "label": "Rice Tungro Virus",
        "description": "Orange-yellow leaf discoloration from tips and stunted tillering spread by green leafhoppers.",
        "action": "Uproot and bury infected clumps immediately, then control green leafhopper vectors using neem or approved systemic sprays.",
    },
    "Bacterial_spot": {
        "label": "Bacterial Infection",
        "description": "Small dark spots on leaves suggest a bacterial attack.",
        "action": "Spray recommended copper-based fungicide within 48 hours. Avoid watering the leaves directly.",
    },
    "Late_blight": {
        "label": "Serious Late Blight",
        "description": "Large grey-green patches. This spreads very fast in wet weather.",
        "action": "ACTION REQUIRED: Apply fungicide immediately. Remove and burn heavily infected plants to save the rest.",
    },
    "Early_blight": {
        "label": "Early Blight",
        "description": "Target-like brown spots on older leaves.",
        "action": "Apply fungicide. Increase spacing between plants to improve air flow.",
    },
    "Leaf_Mold": {
        "label": "Leaf Mold",
        "description": "Pale green/yellow spots on top of leaves with fuzzy growth underneath.",
        "action": "Reduce humidity in the greenhouse. Ensure plants have enough space.",
    },
    "Yellow_Leaf_Curl_Virus": {
        "label": "Yellow Leaf Virus",
        "description": "Leaves are curling upwards and turning yellow. Spread by tiny whiteflies.",
        "action": "Control whiteflies with soap spray or neem oil. Remove infected plants to stop the spread.",
    },
}

DEFAULT_ADVICE = {
    "label": "Uncertain / Non-Crop Specimen",
    "description": "This image could not be verified against known rice crop leaf pathology.",
    "action": "Capture a sharp close-up of a single paddy leaf in natural daylight or consult your local Krishi Vigyan Kendra.",
}


def get_certainty_label(confidence: float) -> str:
    if confidence >= 85:
        return "High Certainty"
    if confidence >= 65:
        return "Medium Certainty"
    return "Low Certainty"


def translate_to_farmer_language(disease_type: str, confidence: float) -> dict:
    """
    Returns a dictionary with farmer-friendly fields.
    """
    # Clean up parentheses & dashes so e.g. 'Rice Blast (Magnaporthe oryzae)' matches 'Rice_Blast'
    base_name = disease_type.split(" (")[0].strip()
    lookup_key = base_name.replace(" ", "_").replace("-", "_")

    mapping = DISEASE_MAPPING.get(lookup_key, DISEASE_MAPPING.get(disease_type.replace(" ", "_"), DEFAULT_ADVICE))
    certainty = get_certainty_label(confidence)

    return {
        "simple_label": mapping["label"],
        "certainty": certainty,
        "what_is_it": mapping["description"],
        "what_to_do": mapping["action"],
    }
