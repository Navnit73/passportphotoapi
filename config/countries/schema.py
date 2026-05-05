"""
Unified country configuration schema.

Every country config MUST include:
  - schema_version
  - country (name, code)
  - document_type
  - photo_spec
  - digital_requirements
  - face_constraints
  - background_rules
  - auto_crop_config

All measurements include units. Internally, everything is
normalized to percentage of image height.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─── Photo Spec ───

class PhotoSpec(BaseModel):
    """Physical print dimensions."""
    width_in: Optional[float] = None
    height_in: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    aspect_ratio: float = 1.0
    print_dpi: int = 300


# ─── Digital Requirements ───

class ResolutionPx(BaseModel):
    width: int
    height: int


class DigitalRequirements(BaseModel):
    """Digital file constraints."""
    required_resolution_px: ResolutionPx
    min_resolution_px: Optional[ResolutionPx] = None
    max_resolution_px: Optional[ResolutionPx] = None
    max_file_size_kb: int = 240
    format: list[str] = Field(default_factory=lambda: ["jpg", "jpeg"])
    background_color: str = "white"


# ─── Face Constraints ───

class MinMax(BaseModel):
    min: float
    max: float


class EyePosition(BaseModel):
    from_bottom_pct: MinMax


class TopMargin(BaseModel):
    min: float
    recommended: Optional[float] = None


class ChinToBottom(BaseModel):
    min: float


class FaceConstraints(BaseModel):
    """Biometric face positioning rules."""
    head_height_pct: MinMax
    eye_position: EyePosition
    top_margin_pct: TopMargin
    chin_to_bottom_pct: Optional[ChinToBottom] = None


# ─── Pose Rules ───

class PoseRules(BaseModel):
    """Expression and head position rules."""
    head_tilt_allowed: bool = False
    expression: str = "neutral"
    eyes_open: bool = True
    mouth_closed: bool = True
    both_ears_visible: bool = False


# ─── Background Rules ───

class BackgroundRules(BaseModel):
    """Background color and quality rules."""
    color: str = "plain white"
    shadows_allowed: bool = False
    texture_allowed: bool = False


# ─── Print Layout ───

class PrintLayout(BaseModel):
    """Print sheet configuration."""
    paper_size: str = "4x6_in"
    photos_per_sheet: int = 6
    spacing_mm: float = 5
    cut_guides: bool = True


# ─── Auto Crop Config ───

class AutoCropConfig(BaseModel):
    """Parameters for the automatic crop engine."""
    head_top_multiplier: float = 1.25
    eye_line_target_pct: float = 0.62
    safe_padding_pct: float = 5.0


# ─── Country Identity ───

class CountryInfo(BaseModel):
    name: str
    code: str


# ─── Root Config ───

class CountryConfig(BaseModel):
    """
    Unified country configuration.

    This is the canonical schema that every country MUST conform to.
    The loader normalizes flat JSON specs into this structure.
    """
    schema_version: str = "1.0"
    country: CountryInfo
    document_type: str  # "passport" | "visa"

    photo_spec: PhotoSpec
    digital_requirements: DigitalRequirements
    face_constraints: FaceConstraints
    pose_rules: PoseRules = Field(default_factory=PoseRules)
    background_rules: BackgroundRules = Field(default_factory=BackgroundRules)
    print_layout: PrintLayout = Field(default_factory=PrintLayout)
    auto_crop_config: AutoCropConfig = Field(default_factory=AutoCropConfig)

    # ─── Derived helpers (not stored in JSON) ───

    @property
    def target_width_px(self) -> int:
        return self.digital_requirements.required_resolution_px.width

    @property
    def target_height_px(self) -> int:
        return self.digital_requirements.required_resolution_px.height

    @property
    def target_aspect_ratio(self) -> float:
        return self.target_width_px / self.target_height_px

    @property
    def target_head_pct(self) -> float:
        """Midpoint of allowed head height range (as 0-1 fraction)."""
        mn = self.face_constraints.head_height_pct.min / 100
        mx = self.face_constraints.head_height_pct.max / 100
        return (mn + mx) / 2

    @property
    def target_eye_from_bottom_pct(self) -> float:
        """Midpoint of eye-from-bottom range (as 0-1 fraction)."""
        mn = self.face_constraints.eye_position.from_bottom_pct.min / 100
        mx = self.face_constraints.eye_position.from_bottom_pct.max / 100
        return (mn + mx) / 2

    @property
    def target_eye_from_top_pct(self) -> float:
        """Eye position measured from top (as 0-1 fraction)."""
        return 1.0 - self.target_eye_from_bottom_pct
