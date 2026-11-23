"""
Translation layer to convert technical ML outputs into actionable farmer-friendly language.
"""  # noqa: E501

DISEASE_MAPPING = {
    "Healthy": {
        "label": "Crop is Healthy",
        "description": "Your crop looks strong and disease-free.",
        "action": "Keep doing what you are doing. Check again in 7 days.",
    },
    "Bacterial_spot": {
        "label": "Bacterial Infection",
        "description": "Small dark spots on leaves suggest a bacterial attack.",  # noqa: E501
        "action": "Spray recommended copper-based fungicide within 48 hours. Avoid watering the leaves directly.",  # noqa: E501
    },
    "Late_blight": {
        "label": "Serious Late Blight",
        "description": "Large grey-green patches. This spreads very fast in wet weather.",  # noqa: E501
        "action": "ACTION REQUIRED: Apply fungicide immediately. Remove and burn heavily infected plants to save the rest.",  # noqa: E501
    },
    "Early_blight": {
        "label": "Early Blight",
        "description": "Target-like brown spots on older leaves.",
        "action": "Apply fungicide. Increase spacing between plants to improve air flow.",  # noqa: E501
    },
    "Leaf_Mold": {
        "label": "Leaf Mold",
        "description": "Pale green/yellow spots on top of leaves with fuzzy growth underneath.",  # noqa: E501
        "action": "Reduce humidity in the greenhouse. Ensure plants have enough space.",  # noqa: E501
    },
    "Yellow_Leaf_Curl_Virus": {
        "label": "Yellow Leaf Virus",
        "description": "Leaves are curling upwards and turning yellow. Spread by tiny whiteflies.",  # noqa: E501
        "action": "Control whiteflies with soap spray or neem oil. Remove infected plants to stop the spread.",  # noqa: E501
    },
}

DEFAULT_ADVICE = {
    "label": "Possible Disease Detected",
    "description": "The AI noticed something unusual on the leaves.",
    "action": "Consult with a local agricultural officer or show this photo to an expert.",  # noqa: E501
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
    # Clean up the disease string (replace underscores/dashes)
    lookup_key = disease_type.replace(" ", "_").replace("-", "_")

    mapping = DISEASE_MAPPING.get(lookup_key, DEFAULT_ADVICE)
    certainty = get_certainty_label(confidence)

    return {
        "simple_label": mapping["label"],
        "certainty": certainty,
        "what_is_it": mapping["description"],
        "what_to_do": mapping["action"],
    }
