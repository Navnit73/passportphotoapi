"""
Cloudinary storage integration.
"""

import logging
import requests
import numpy as np
import cv2

import cloudinary
import cloudinary.uploader

from config.settings import settings

logger = logging.getLogger(__name__)

# Configure cloudinary using settings
cloudinary.config(
    cloud_name=settings.cloudinary_cloud_name,
    api_key=settings.cloudinary_api_key,
    api_secret=settings.cloudinary_api_secret,
)

def upload_original(file_bytes: bytes, filename: str) -> str:
    """
    Upload an original image to Cloudinary.
    
    Args:
        file_bytes: raw file bytes
        filename: original filename
        
    Returns:
        Secure URL of the uploaded image
    """
    logger.info("Uploading original image to Cloudinary...")
    response = cloudinary.uploader.upload(
        file_bytes,
        folder="passport/uploads",
        use_filename=True,
        unique_filename=True,
    )
    return response.get("secure_url")

def download_image(url: str) -> np.ndarray:
    """
    Download an image from a URL and decode to a CV2 BGR array.
    """
    logger.info(f"Downloading image from {url}...")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    image_array = np.frombuffer(resp.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode downloaded image")
    return image

def upload_results(
    result_id: str,
    photo_bytes: bytes,
    print_sheet_bytes: bytes | None = None,
    preview_bytes: bytes | None = None,
) -> dict:
    """
    Upload processed results to Cloudinary.
    
    Returns:
        dict mapping file types to secure URLs
    """
    urls = {}
    logger.info(f"Uploading results for {result_id} to Cloudinary...")
    
    # 1. Photo
    res = cloudinary.uploader.upload(
        photo_bytes,
        folder="passport/results",
        public_id=f"{result_id}_photo",
    )
    urls["image_url"] = res.get("secure_url")
    
    # 2. Print Sheet
    if print_sheet_bytes:
        res = cloudinary.uploader.upload(
            print_sheet_bytes,
            folder="passport/results",
            public_id=f"{result_id}_print",
        )
        urls["print_sheet_url"] = res.get("secure_url")
        
    # 3. Preview
    if preview_bytes:
        res = cloudinary.uploader.upload(
            preview_bytes,
            folder="passport/results",
            public_id=f"{result_id}_preview",
        )
        urls["preview_url"] = res.get("secure_url")
        
    return urls
