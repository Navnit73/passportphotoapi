
import requests
import cv2
import numpy as np
import io

API_URL = "http://localhost:8001"
API_KEY = "test_key"
HEADERS = {"X-API-Key": API_KEY}

def test_precrop_validation():
    print("--- Testing /validate with pre-cropped image (600x600) ---")
    # Create a 600x600 dummy image
    img = np.full((600, 600, 3), 255, dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()
    
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"country_code": "US"}
    
    response = requests.post(f"{API_URL}/validate", headers=HEADERS, files=files, data=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        res_json = response.json()
        resolution = res_json.get("resolution", {})
        print(f"Resolution Status: {resolution.get('status')}")
        print(f"Resolution Details: {resolution.get('details')}")
        if resolution.get("status") == "error" and "wrong croped image" in resolution.get("details"):
            print("SUCCESS: Validation correctly flagged pre-cropped image.")
        else:
            print("FAILURE: Validation did not flag pre-cropped image correctly.")
    else:
        print(f"FAILURE: Request failed with {response.text}")

def test_precrop_processing():
    print("\n--- Testing /process with pre-cropped image (600x600) ---")
    img = np.full((600, 600, 3), 255, dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()
    
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"country_code": "US"}
    
    response = requests.post(f"{API_URL}/process", headers=HEADERS, files=files, data=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 400:
        res_json = response.json()
        detail = res_json.get("detail")
        print(f"Error Detail: {detail}")
        if "wrong croped image" in detail:
            print("SUCCESS: Process correctly blocked pre-cropped image.")
        else:
            print("FAILURE: Process returned 400 but message mismatch.")
    else:
        print(f"FAILURE: Expected 400, got {response.status_code}. Response: {response.text}")

def test_valid_processing():
    print("\n--- Testing /process with valid original image (800x600) ---")
    # Note: 800H x 600W is still potentially problematic if target is 600x600, 
    # but since it's NOT exactly 600x600, it should pass this specific check.
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()
    
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"country_code": "US"}
    
    response = requests.post(f"{API_URL}/process", headers=HEADERS, files=files, data=data)
    print(f"Status: {response.status_code}")
    # We expect this to fail later (face detection) but it should pass the initial dimensions check.
    if response.status_code == 400:
        detail = response.json().get("detail")
        if "wrong croped image" in detail:
             print("FAILURE: Process blocked valid image incorrectly.")
        else:
             print(f"SUCCESS: Pass dimensions check (failed later as expected with: {detail})")
    else:
         print(f"Response: {response.status_code}")

if __name__ == "__main__":
    try:
        test_precrop_validation()
        test_precrop_processing()
        test_valid_processing()
    except Exception as e:
        print(f"Error running tests: {e}")
