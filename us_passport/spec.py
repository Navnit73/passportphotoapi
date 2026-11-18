"""
Official US Department of State photo requirements
(travel.state.gov - "Photo Composition Template", digital image).

All head / eye metrics are fractions of the *image* height (== width, 1:1).
"""

# ── Digital image ──
OUT_PX = 600                    # 2in x 2in @ 300 dpi, also the digital minimum
MIN_PX = 600
MAX_PX = 1200
OUT_DPI = 300
MAX_FILE_KB = 240               # State Dept. digital upload limit
ALLOWED_FORMAT = "JPEG"

# ── Head size: top of hair -> bottom of chin: 1" to 1 3/8" of a 2" photo ──
HEAD_MIN = 0.50
HEAD_MAX = 0.69

# ── Eye height measured from the bottom of the image: 1 1/8" to 1 3/8" ──
# (the official digital template says 56% - 69%)
EYE_MIN = 0.56
EYE_MAX = 0.69

# ── Where the engine aims (centre of the legal window, with safety margin
#    so landmark error of a few % can never push us out of range) ──
TARGET_HEAD = 0.60
TARGET_EYE = 0.625

# Whole head must be visible: keep at least this much space above the hair.
MIN_TOP_MARGIN = 0.03
PREFERRED_TOP_MARGIN = 0.07

# Head must be centred horizontally.
CENTER_TOL_PASS = 0.03
CENTER_TOL_FAIL = 0.06

# Pose
ROLL_PASS, ROLL_FAIL = 2.0, 5.0         # degrees (head tilt)
YAW_PASS, YAW_FAIL = 7.0, 12.0          # degrees (turned left/right)
# MediaPipe's canonical face reads ~10 deg 'pitch' on a perfectly frontal photo,
# so the limits are relative to that bias.
PITCH_PASS, PITCH_FAIL = 20.0, 30.0     # degrees (up/down)
ROLL_CORRECT_MAX = 25.0                 # largest tilt we will straighten

# Background: white or off-white, no shadows / texture
BG_MIN_CHANNEL = 235
BG_MAX_STD = 10.0

# Source must offer enough pixels in the head crop
MIN_SOURCE_CROP_PX = 400                # below this we refuse (too blurry)
WARN_SOURCE_CROP_PX = 600
