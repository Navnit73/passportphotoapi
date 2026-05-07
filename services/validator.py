"""
Image validation service for passport compliance reports.
"""

import cv2
import numpy as np
import logging
from typing import Optional
from io import BytesIO
from PIL import Image

from api.models import ValidationResponse, ValidationItem
from services.face_detector import detect_face
from services.hair_detector import detect_hair_crown

logger = logging.getLogger(__name__)

def analyze_image(file_bytes: bytes, config, content_type: str) -> ValidationResponse:
    """
    Perform a comprehensive validation check on an uploaded photo.
    """
    # 1. Decode Image
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid or corrupted image file")

    h, w = image.shape[:2]
    summary = []
    
    # --- 1. File Format ---
    fmt = content_type.split("/")[-1].upper()
    file_format = ValidationItem(
        label="File Format",
        value=f"Valid {fmt} format",
        status="success"
    )
    
    # --- 2. File Size ---
    size_kb = len(file_bytes) // 1024
    max_kb = config.digital_requirements.max_file_size_kb
    status = "success" if size_kb <= max_kb else "warning"
    file_size = ValidationItem(
        label="File Size",
        value=f"{size_kb} KB",
        status=status,
        details=f"Exceeds {max_kb} KB (will be compressed during processing)" if status == "warning" else "Within limits",
        target=f"max {max_kb} KB"
    )

    # --- 3. Image Resolution ---
    min_res = config.digital_requirements.min_resolution_px or config.digital_requirements.required_resolution_px
    min_w = min_res.width
    min_h = min_res.height
    res_status = "success" if (w >= min_w and h >= min_h) else "error"
    resolution = ValidationItem(
        label="Image Resolution",
        value=f"{w}x{h} px",
        status=res_status,
        details="Meets minimum requirements" if res_status == "success" else f"Below minimum {min_w}x{min_h}",
        target=f"min {min_w}x{min_h}"
    )

    # --- 3b. DPI Check ---
    try:
        with Image.open(BytesIO(file_bytes)) as pil_img:
            dpi_meta = pil_img.info.get('dpi')
            
        target_dpi = config.photo_spec.print_dpi
        if dpi_meta:
            actual_dpi = int(dpi_meta[0])
            dpi_status = "success" if actual_dpi >= target_dpi else "warning"
            dpi_val = f"{actual_dpi} DPI"
            dpi_details = "Meets printing requirements" if dpi_status == "success" else f"Lower than recommended {target_dpi} DPI"
        else:
            dpi_val = "Not detected"
            dpi_status = "warning"
            dpi_details = "DPI metadata missing (common for web uploads)"
        
        dpi_item = ValidationItem(
            label="Print DPI",
            value=dpi_val,
            status=dpi_status,
            details=dpi_details,
            target=f"{target_dpi} DPI"
        )
    except Exception as e:
        logger.warning(f"Failed to read DPI: {e}")
        dpi_item = ValidationItem(label="Print DPI", value="Error reading metadata", status="warning")

    # --- 4. Lighting ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 40:
        l_val, l_status = "Too dark", "error"
    elif mean_brightness > 230:
        l_val, l_status = "Too bright / Overexposed", "error"
    else:
        l_val, l_status = "Good exposure detected", "success"
        summary.append("Good lighting detected")
    
    lighting = ValidationItem(label="Lighting", value=l_val, status=l_status)

    # --- 5. Face Detection ---
    try:
        face = detect_face(image)
        f_val, f_status = "Exactly 1 face detected", "success"
        summary.append("Face detected successfully")
    except Exception as e:
        f_val, f_status = str(e), "error"
    
    face_detection = ValidationItem(label="Face Detection", value=f_val, status=f_status)

    # --- Biometrics (if face detected) ---
    if f_status == "success":
        # 6. Eye Level
        eye_y = face.eye_center.y
        eye_pct = ((h - eye_y) / h) * 100
        min_eye = config.face_constraints.eye_position.from_bottom_pct.min
        max_eye = config.face_constraints.eye_position.from_bottom_pct.max
        e_status = "success" if (min_eye <= eye_pct <= max_eye) else "warning"
        
        eye_level = ValidationItem(
            label="Eye Level",
            value=f"{eye_pct:.1f}% from bottom",
            status=e_status,
            target=f"{min_eye}–{max_eye}%",
            details="Within range" if e_status == "success" else "Outside recommended range"
        )
        if e_status == "success":
            summary.append(f"Eye level: {eye_pct:.1f}% ✓")
        
        # 7. Head Size
        try:
            crown_y = detect_hair_crown(image)
        except Exception:
            crown_y = face.top_of_head_y
        crown_y = min(crown_y, face.top_of_head_y)
        
        head_height = face.chin_y - crown_y
        head_pct = (head_height / h) * 100
        min_head = config.face_constraints.head_height_pct.min
        max_head = config.face_constraints.head_height_pct.max
        h_status = "success" if (min_head <= head_pct <= max_head) else "warning"
        
        head_size = ValidationItem(
            label="Head Size",
            value=f"{head_pct:.1f}% of image height",
            status=h_status,
            target=f"{min_head}–{max_head}%",
            details="Head is too small" if head_pct < min_head else ("Head is too large" if head_pct > max_head else "Within range")
        )
        if h_status == "success":
            summary.append(f"Head size: {head_pct:.1f}% ✓")
        else:
            summary.append(f"Head size: {head_pct:.1f}% (target: {min_head}–{max_head}%)")

        # 8. Orientation
        # Simple check: eyes should be roughly horizontal
        eye_diff = abs(face.left_eye_center.y - face.right_eye_center.y) / h
        if eye_diff > 0.02:
            o_val, o_status = "Head tilt detected", "warning"
        else:
            o_val, o_status = "Frontal orientation verified", "success"
            summary.append("Frontal orientation ✓")
        
        orientation = ValidationItem(label="Orientation", value=o_val, status=o_status)
    else:
        # Error placeholders
        eye_level = ValidationItem(label="Eye Level", value="N/A", status="error")
        head_size = ValidationItem(label="Head Size", value="N/A", status="error")
        orientation = ValidationItem(label="Orientation", value="N/A", status="error")

    # --- Final Result & Suggestions ---
    items = [
        file_format, file_size, resolution, dpi_item, lighting, 
        face_detection, eye_level, head_size, orientation
    ]
    
    # Calculate overall status
    has_failures = any(item.status != "success" for item in items)
    overall_result = "FAIL" if has_failures else "PASS"
    
    # Generate suggestions
    suggestions = []
    for item in items:
        if item.status != "success":
            if item.label == "Face Detection" and item.status == "error":
                suggestions.append("Ensure your face is clearly visible and look directly at the camera.")
            elif item.label == "Head Size":
                if "small" in (item.details or "").lower():
                    suggestions.append("Move closer to the camera to make your head larger.")
                else:
                    suggestions.append("Move slightly further away from the camera.")
            elif item.label == "Eye Level" and item.status != "success":
                suggestions.append("Adjust your position so your eyes are centered vertically in the frame.")
            elif item.label == "Lighting" and item.status == "error":
                suggestions.append("Ensure even lighting on your face without harsh shadows or bright glares.")
            elif item.label == "Orientation" and item.status != "success":
                suggestions.append("Keep your head straight and avoid tilting it to the side.")
            elif item.label == "Image Resolution" and item.status == "error":
                suggestions.append("Use a higher quality camera or better lighting for a clearer image.")

    return ValidationResponse(
        status="success" if f_status == "success" else "error",
        overall_result=overall_result,
        summary=summary,
        suggestions=suggestions,
        file_format=file_format,
        file_size=file_size,
        resolution=resolution,
        lighting=lighting,
        face_detection=face_detection,
        eye_level=eye_level,
        head_size=head_size,
        orientation=orientation,
        dpi=dpi_item
    )
