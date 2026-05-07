import os
import sys
import time
import cv2
import numpy as np
import logging

# Add current directory to path so we can import services
sys.path.append(os.getcwd())

from services.background import _get_rembg_session, correct_background_rembg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_isnet():
    image_path = "test_portrait.jpg"
    if not os.path.exists(image_path):
        logger.error(f"Sample image {image_path} not found.")
        return

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image {image_path}.")
        return

    logger.info("--- Phase 1: First call (includes session init) ---")
    start_time = time.time()
    result1 = correct_background_rembg(image, target_color="white")
    end_time = time.time()
    logger.info(f"First call took {end_time - start_time:.2f} seconds")

    logger.info("--- Phase 2: Second call (should use cached session) ---")
    start_time = time.time()
    result2 = correct_background_rembg(image, target_color="white")
    end_time = time.time()
    logger.info(f"Second call took {end_time - start_time:.2f} seconds")

    # Save results
    output_path = "scratch/result_isnet.jpg"
    cv2.imwrite(output_path, result2)
    logger.info(f"Result saved to {output_path}")

if __name__ == "__main__":
    test_isnet()
