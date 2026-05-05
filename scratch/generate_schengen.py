import json
import os

countries = [
    ("Austria", "AT"), ("Belgium", "BE"), ("Bulgaria", "BG"), ("Croatia", "HR"),
    ("Czechia", "CZ"), ("Denmark", "DK"), ("Estonia", "EE"), ("Finland", "FI"),
    ("France", "FR"), ("Germany", "DE"), ("Greece", "GR"), ("Hungary", "HU"),
    ("Italy", "IT"), ("Latvia", "LV"), ("Lithuania", "LT"), ("Luxembourg", "LU"),
    ("Malta", "MT"), ("Netherlands", "NL"), ("Poland", "PL"), ("Portugal", "PT"),
    ("Romania", "RO"), ("Slovakia", "SK"), ("Slovenia", "SI"), ("Spain", "ES"),
    ("Sweden", "SE")
]

# Template based on schengen_passport.json
template = {
    "schema_version": "1.0",
    "country": {
        "name": "",
        "code": ""
    },
    "document_type": "passport",
    "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 400
    },
    "digital_requirements": {
        "required_resolution_px": {"width": 413, "height": 531},
        "min_resolution_px": {"width": 350, "height": 450},
        "max_resolution_px": {"width": 1000, "height": 1300},
        "max_file_size_kb": 500,
        "format": ["jpg", "jpeg"],
        "background_color": "light gray"
    },
    "face_constraints": {
        "head_height_pct": {"min": 70, "max": 80},
        "eye_position": {"from_bottom_pct": {"min": 50, "max": 65}},
        "top_margin_pct": {"min": 5, "recommended": 8},
        "chin_to_bottom_pct": {"min": 5}
    },
    "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": True
    },
    "background_rules": {
        "color": "plain light gray",
        "shadows_allowed": False,
        "texture_allowed": False
    },
    "print_layout": {
        "paper_size": "A4",
        "photos_per_sheet": 8,
        "spacing_mm": 5,
        "cut_guides": True
    },
    "auto_crop_config": {
        "head_top_multiplier": 1.25,
        "eye_line_target_pct": 0.6,
        "safe_padding_pct": 5
    }
}

base_dir = "/Users/navnitrai/Desktop/My/python/config/countries"

for name, code in countries:
    config = template.copy()
    config["country"] = {"name": name, "code": code}
    
    filename = f"{name.lower().replace(' ', '_')}_passport.json"
    filepath = os.path.join(base_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Generated {filename}")
