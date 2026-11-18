"""
Standalone US passport / visa photo engine.

Independent of services.cropper / services.face_detector / services.hair_detector.
Only the background-removal service (MODNet -> rembg) and the generic
JPEG encoder are shared with the rest of the project.

Run as its own API:   uvicorn us_passport.app:app --port 8001
"""
