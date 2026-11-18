"""US preview: same renderer as the main /process endpoint, with US labels."""

import numpy as np

from services.preview import render_measurement_preview

from . import spec
from .compliance import Metrics
from .measure import Measurement


def create_preview(img: np.ndarray, m: Measurement, mt: Metrics, compliant: bool = True) -> bytes:
    S = img.shape[0]
    cap = f"Head {mt.head_pct:.1f}%  ·  Eyes {mt.eye_pct:.1f}% from bottom  ·  Top margin {mt.top_margin_pct:.1f}%"
    return render_measurement_preview(
        img,
        crown_frac=m.crown_y / S,
        chin_frac=m.face.chin_y / S,
        eye_frac=m.face.eye_y / S,
        width_label="2 in", height_label="2 in",
        head_label="1–1 3/8 in", eye_label="1 1/8–1 3/8 in",
        caption=cap, compliant=compliant,
    )
