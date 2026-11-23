"""
Multilingual translation layer for AgriPulse.
Supports regional languages for maximum adoption.
"""

# Dictionary for labels and advice
TRANSLATIONS = {
    "en": {
        "Healthy": "Crop is Healthy",
        "Healthy_adv": "Check again in 7 days.",
        "Blight": "Serious Blight",
        "Blight_adv": "ACTION REQUIRED: Apply fungicide immediately. Remove infected plants.",  # noqa: E501
        "NotSure": "Not Sure",
        "NotSure_adv": "Please try again with a closer photo and better lighting.",  # noqa: E501
        "VillageAlert": "Village Warning: Similar disease reported nearby.",
    },
    "hi": {  # Hindi
        "Healthy": "फसल स्वस्थ है",
        "Healthy_adv": "7 दिनों में फिर से जांचें।",
        "Blight": "गंभीर ब्लाइट",
        "Blight_adv": "कार्रवाई आवश्यक: तुरंत कवकनाशी लगाएं। संक्रमित पौधों को हटा दें।",  # noqa: E501
        "NotSure": "निश्चित नहीं है",
        "NotSure_adv": "कृपया बेहतर रोशनी में और करीब से फोटो लेकर फिर से प्रयास करें।",  # noqa: E501
        "VillageAlert": "गांव की चेतावनी: पास के खेतों में इसी तरह की बीमारी की सूचना मिली है।",  # noqa: E501
    },
    "te": {  # Telugu
        "Healthy": "పంట ఆరోగ్యంగా ఉంది",
        "Healthy_adv": "7 రోజుల తర్వాత మళ్ళీ తనిఖీ చేయండి.",
        "Blight": "తీవ్రమైన బ్లైట్",
        "Blight_adv": "చర్య అవసరం: వెంటనే శిలీంద్ర సంహారిణిని వాడండి. సోకిన మొక్కలను తొలగించండి.",  # noqa: E501
        "NotSure": "ఖచ్చితంగా తెలియదు",
        "NotSure_adv": "దయచేసి మంచి వెలుతురులో మరియు దగ్గరగా ఫోటో తీసి మళ్ళీ ప్రయత్నించండి.",  # noqa: E501
        "VillageAlert": "గ్రామ హెచ్చరిక: సమీపంలోని పొలాల్లో ఇటువంటి వ్యాధి నమోదైంది.",  # noqa: E501
    },
}


def get_farmer_translation(disease_key: str, lang: str = "en") -> dict:
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])

    # Map technical disease to simplified keys
    simplified_key = "Healthy" if "Healthy" in disease_key else "Blight"

    return {
        "label": lang_dict.get(simplified_key, lang_dict["Healthy"]),
        "advice": lang_dict.get(
            f"{simplified_key}_adv", lang_dict["Healthy_adv"]
        ),
        "alert": lang_dict.get("VillageAlert", ""),
    }
