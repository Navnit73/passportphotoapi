
import cv2
import numpy as np
import logging
import os
from services.cropper import compute_crop
from services.face_detector import detect_face
from services.hair_detector import detect_hair_crown
from config.countries.loader import get_config, load_all_configs

logging.basicConfig(level=logging.INFO)

# Load all configs at startup
load_all_configs()

def test_cropper(image_path, country_code="IN"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return

    config = get_config(country_code, "passport")
    if config is None:
        print(f"Config not found for {country_code}")
        return
    
    # 1. Face detection
    face = detect_face(image)
    
    # 2. Hair detection with fallback
    try:
        crown_y = detect_hair_crown(image)
    except Exception as e:
        print(f"Hair detection failed: {e}")
        crown_y = face.top_of_head_y
        
    print(f"\n--- Testing {image_path} for {country_code} ---")
    print(f"Face Chin Y: {face.chin_y:.1f}")
    print(f"Face Forehead Y: {face.forehead_y:.1f}")
    print(f"Face Estimated Top Y: {face.top_of_head_y:.1f}")
    print(f"Detected Crown Y: {crown_y:.1f}")
    
    # Use the logic from routes.py
    refined_crown_y = crown_y
    print(f"Refined Crown Y: {refined_crown_y:.1f}")
    
    # 3. Crop
    try:
        result = compute_crop(image, face, refined_crown_y, config)
        print("Crop Successful!")
        print(f"Final Head Pct: {result.head_height_pct}% (Range: {config.face_constraints.head_height_pct.min}-{config.face_constraints.head_height_pct.max}%)")
        print(f"Final Eye Position: {result.eye_from_bottom_pct}% (Range: {config.face_constraints.eye_position.from_bottom_pct.min}-{config.face_constraints.eye_position.from_bottom_pct.max}%)")
        print(f"Final Top Margin: {result.top_margin_pct}% (Min: {config.face_constraints.top_margin_pct.min}%)")
        
        # Save result
        if not os.path.exists("scratch"):
            os.makedirs("scratch")
        output_path = f"scratch/crop_test_{country_code}_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result.image)
        print(f"Saved result to {output_path}")

        # Check for compliance
        if result.head_height_pct < config.face_constraints.head_height_pct.min:
            print("!!! BUG: Head too small")
        if result.head_height_pct > config.face_constraints.head_height_pct.max:
            print("!!! BUG: Head too large")
        if result.eye_from_bottom_pct < config.face_constraints.eye_position.from_bottom_pct.min:
            print("!!! BUG: Eyes too low")
        if result.eye_from_bottom_pct > config.face_constraints.eye_position.from_bottom_pct.max:
            print("!!! BUG: Eyes too high")
            
    except Exception as e:
        print(f"Crop Failed: {e}")

if __name__ == "__main__":
    test_cropper("test_face.jpg", "IN")
    test_cropper("test_portrait.jpg", "US")
    test_cropper("test_face.jpg", "US")
