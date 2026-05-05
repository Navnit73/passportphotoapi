import json
import os

base_dir = "/Users/navnitrai/Desktop/My/python/config/countries"
files = [f for f in os.listdir(base_dir) if f.endswith(".json")]

digital_colors = set()
rule_colors = set()

for filename in files:
    filepath = os.path.join(base_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        config = json.load(f)
        
        if "digital_requirements" in config:
            digital_colors.add(config["digital_requirements"].get("background_color"))
            
        if "background_rules" in config:
            rule_colors.add(config["background_rules"].get("color"))

print("Digital Requirement Colors:", sorted(list(digital_colors)))
print("Background Rule Colors:", sorted(list(rule_colors)))
