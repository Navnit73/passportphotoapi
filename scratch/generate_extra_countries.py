import json
import os

data = {
  "countries": [
    {
      "schema_version": "1.0",
      "country": { "name": "Singapore", "code": "SG" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 600
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 400, "height": 514 },
        "min_resolution_px": { "width": 350, "height": 450 },
        "max_resolution_px": { "width": 1000, "height": 1300 },
        "max_file_size_kb": 8192,
        "format": ["jpg", "jpeg", "heic", "heif", "png"],
        "background_color": "white"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": True
      },
      "background_rules": {
        "color": "plain white",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "United Arab Emirates", "code": "AE" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 40,
        "height_mm": 60,
        "aspect_ratio": 0.667,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 472, "height": 709 },
        "min_resolution_px": { "width": 400, "height": 600 },
        "max_resolution_px": { "width": 945, "height": 1417 },
        "max_file_size_kb": 240,
        "format": ["jpg", "jpeg"],
        "background_color": "light gray"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 10 },
        "chin_to_bottom_pct": { "min": 5 }
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
        "photos_per_sheet": 6,
        "spacing_mm": 5,
        "cut_guides": True
      },
      "auto_crop_config": {
        "head_top_multiplier": 1.25,
        "eye_line_target_pct": 0.6,
        "safe_padding_pct": 5
      }
    },
    {
      "schema_version": "1.0",
      "country": { "name": "South Korea", "code": "KR" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 600
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 413, "height": 531 },
        "min_resolution_px": { "width": 350, "height": 450 },
        "max_resolution_px": { "width": 1000, "height": 1300 },
        "max_file_size_kb": 300,
        "format": ["jpg", "jpeg", "png", "gif", "bmp", "pdf"],
        "background_color": "white"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": True
      },
      "background_rules": {
        "color": "plain white",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "United Kingdom", "code": "GB" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 600
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 600, "height": 750 },
        "min_resolution_px": { "width": 600, "height": 750 },
        "max_resolution_px": { "width": 1200, "height": 1500 },
        "max_file_size_kb": 10240,
        "format": ["jpg", "jpeg"],
        "background_color": "light gray or cream"
      },
      "face_constraints": {
        "head_height_pct": { "min": 64, "max": 76 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain light colored (white, cream, or light grey)",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Turkey", "code": "TR" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 50,
        "height_mm": 60,
        "aspect_ratio": 0.833,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 591, "height": 709 },
        "min_resolution_px": { "width": 500, "height": 600 },
        "max_resolution_px": { "width": 1200, "height": 1440 },
        "max_file_size_kb": 240,
        "format": ["jpg", "jpeg"],
        "background_color": "white"
      },
      "face_constraints": {
        "head_height_pct": { "min": 53, "max": 60 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 10 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain white",
        "shadows_allowed": False,
        "texture_allowed": False
      },
      "print_layout": {
        "paper_size": "A4",
        "photos_per_sheet": 6,
        "spacing_mm": 5,
        "cut_guides": True
      },
      "auto_crop_config": {
        "head_top_multiplier": 1.2,
        "eye_line_target_pct": 0.6,
        "safe_padding_pct": 5
      }
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Thailand", "code": "TH" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 413, "height": 531 },
        "min_resolution_px": { "width": 350, "height": 450 },
        "max_resolution_px": { "width": 1000, "height": 1300 },
        "max_file_size_kb": 500,
        "format": ["jpg", "jpeg"],
        "background_color": "white"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain white",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "New Zealand", "code": "NZ" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 900, "height": 1200 },
        "min_resolution_px": { "width": 900, "height": 1200 },
        "max_resolution_px": { "width": 4500, "height": 6000 },
        "max_file_size_kb": 5120,
        "format": ["jpg", "jpeg"],
        "background_color": "plain light colored"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain light colored (high contrast with face)",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Japan", "code": "JP" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 600
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 827, "height": 1063 },
        "min_resolution_px": { "width": 600, "height": 800 },
        "max_resolution_px": { "width": 1200, "height": 1600 },
        "max_file_size_kb": 500,
        "format": ["jpg", "jpeg"],
        "background_color": "white or off-white"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain white or off-white",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Kazakhstan", "code": "KZ" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 413, "height": 531 },
        "min_resolution_px": { "width": 350, "height": 450 },
        "max_resolution_px": { "width": 1000, "height": 1300 },
        "max_file_size_kb": 240,
        "format": ["jpg", "jpeg", "png", "bmp", "gif", "pdf"],
        "background_color": "light gray"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 3, "recommended": 5 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Iraq", "code": "IQ" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 600
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 413, "height": 531 },
        "min_resolution_px": { "width": 350, "height": 450 },
        "max_resolution_px": { "width": 1000, "height": 1300 },
        "max_file_size_kb": 300,
        "format": ["jpg", "jpeg", "png", "bmp", "gif", "pdf"],
        "background_color": "white"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 56, "max": 69 } },
        "top_margin_pct": { "min": 3, "recommended": 5 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain white",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Iran", "code": "IR" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 30,
        "height_mm": 40,
        "aspect_ratio": 0.75,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 600, "height": 400 },
        "min_resolution_px": { "width": 400, "height": 300 },
        "max_resolution_px": { "width": 900, "height": 600 },
        "max_file_size_kb": 500,
        "format": ["jpg", "jpeg"],
        "background_color": "white"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain white",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Indonesia", "code": "ID" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 51,
        "height_mm": 51,
        "aspect_ratio": 1.0,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 400, "height": 600 },
        "min_resolution_px": { "width": 400, "height": 600 },
        "max_resolution_px": { "width": 1200, "height": 1800 },
        "max_file_size_kb": 2048,
        "format": ["jpg", "jpeg", "png"],
        "background_color": "red"
      },
      "face_constraints": {
        "head_height_pct": { "min": 50, "max": 60 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 60 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain red (medium shade, uniform)",
        "shadows_allowed": False,
        "texture_allowed": False
      },
      "print_layout": {
        "paper_size": "A4",
        "photos_per_sheet": 6,
        "spacing_mm": 5,
        "cut_guides": True
      },
      "auto_crop_config": {
        "head_top_multiplier": 1.2,
        "eye_line_target_pct": 0.55,
        "safe_padding_pct": 5
      }
    },
    {
      "schema_version": "1.0",
      "country": { "name": "China", "code": "CN" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 33,
        "height_mm": 48,
        "aspect_ratio": 0.688,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 390, "height": 567 },
        "min_resolution_px": { "width": 354, "height": 472 },
        "max_resolution_px": { "width": 420, "height": 560 },
        "max_file_size_kb": 120,
        "format": ["jpg", "jpeg"],
        "background_color": "white or light blue"
      },
      "face_constraints": {
        "head_height_pct": { "min": 58, "max": 69 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 3, "recommended": 5 },
        "chin_to_bottom_pct": { "min": 7 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain white or light blue",
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
        "head_top_multiplier": 1.2,
        "eye_line_target_pct": 0.6,
        "safe_padding_pct": 5
      }
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Australia", "code": "AU" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 38,
        "height_mm": 48,
        "aspect_ratio": 0.792,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 450, "height": 570 },
        "min_resolution_px": { "width": 400, "height": 510 },
        "max_resolution_px": { "width": 1200, "height": 1500 },
        "max_file_size_kb": 10240,
        "format": ["jpg", "jpeg"],
        "background_color": "neutral or light gray"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "neutral or light grey (no white — must contrast with face)",
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
    },
    {
      "schema_version": "1.0",
      "country": { "name": "Algeria", "code": "DZ" },
      "document_type": "passport",
      "photo_spec": {
        "width_mm": 35,
        "height_mm": 45,
        "aspect_ratio": 0.778,
        "print_dpi": 300
      },
      "digital_requirements": {
        "required_resolution_px": { "width": 413, "height": 531 },
        "min_resolution_px": { "width": 350, "height": 450 },
        "max_resolution_px": { "width": 1000, "height": 1300 },
        "max_file_size_kb": 300,
        "format": ["jpg", "jpeg", "png", "bmp", "gif", "pdf"],
        "background_color": "light colored or white"
      },
      "face_constraints": {
        "head_height_pct": { "min": 70, "max": 80 },
        "eye_position": { "from_bottom_pct": { "min": 50, "max": 65 } },
        "top_margin_pct": { "min": 5, "recommended": 8 },
        "chin_to_bottom_pct": { "min": 5 }
      },
      "pose_rules": {
        "head_tilt_allowed": False,
        "expression": "neutral",
        "eyes_open": True,
        "mouth_closed": True,
        "both_ears_visible": False
      },
      "background_rules": {
        "color": "plain light colored or white",
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
  ]
}

base_dir = "/Users/navnitrai/Desktop/My/python/config/countries"

for config in data["countries"]:
    name = config["country"]["name"]
    filename = f"{name.lower().replace(' ', '_')}_passport.json"
    filepath = os.path.join(base_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Generated {filename}")
