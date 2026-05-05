import json
import os

base_dir = "/Users/navnitrai/Desktop/My/python/config/countries"
files = [f for f in os.listdir(base_dir) if f.endswith(".json")]

# Common Schengen Source (many countries use this)
SCHENGEN_SOURCE = {
    "data_source": "official",
    "source_url": "https://www.schengenvisainfo.com/photo-requirements/",
    "source_authority": "Schengen Visa Information – ICAO Standard"
}

# Standard keys that should be in every file
def standardize_config(config, filename):
    # 1. Add Missing Source Info (especially for the ones I generated earlier)
    if "data_source" not in config:
        if filename in ["us_passport.json", "in_passport.json"]:
            config["data_source"] = "official"
        else:
            # Assume Schengen if not specified
            config.update(SCHENGEN_SOURCE)

    # 2. Add 'space_from_top' if missing (requested in previous context)
    if "photo_spec" in config:
        if "space_from_top" not in config["photo_spec"]:
            config["photo_spec"]["space_from_top"] = 3
            
    # 3. Ensure 'document_type' is consistent
    if "document_type" not in config:
        config["document_type"] = "passport"

    # 4. Ensure print_layout has 'cut_guides'
    if "print_layout" in config:
        if "cut_guides" not in config["print_layout"]:
            config["print_layout"]["cut_guides"] = True

    return config

for filename in files:
    filepath = os.path.join(base_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    updated_config = standardize_config(config, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(updated_config, f, indent=2)
    print(f"Standardized {filename}")
