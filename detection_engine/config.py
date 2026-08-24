"""
Configuration Module - Driver Fatigue and Drowsiness Detection Engine.

All threshold constants and tunable parameters are defined here.
Edit values here to tune detection sensitivity without touching logic files.
"""

# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR)
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.21            # Fallback: EAR below this = eye closed
EAR_CLOSED_FACTOR = 0.72        # Calibration multiplier: threshold = baseline_ear * factor

# ---------------------------------------------------------------------------
# Mouth Aspect Ratio (MAR) and Yawn Detection
# ---------------------------------------------------------------------------
YAWN_MAR_THRESHOLD = 0.35       # MAR above this = potential yawn
YAWN_MAR_FACTOR = 0.16          # Calibration additive: threshold = baseline_mar + factor
YAWN_MIN_FRAMES = 9             # Min consecutive frames above threshold for a yawn (~0.3s @ 30fps)

# ---------------------------------------------------------------------------
# Blink Detection
# ---------------------------------------------------------------------------
BLINK_MIN_FRAMES = 2            # Min consecutive frames EAR below threshold = valid blink
BLINK_MAX_FRAMES = 12           # Max frames (>this = microsleep/prolonged closure, not a blink)

# ---------------------------------------------------------------------------
# PERCLOS Rolling Window
# ---------------------------------------------------------------------------
PERCLOS_WINDOW_SECONDS = 60.0   # Rolling window for PERCLOS and blink rate calculations

# ---------------------------------------------------------------------------
# Startup Baseline Calibration
# ---------------------------------------------------------------------------
CALIBRATION_DURATION_SECONDS = 6.0   # Seconds to collect baseline samples at startup

# ---------------------------------------------------------------------------
# Emergency Alarm Audio
# ---------------------------------------------------------------------------
ALARM_VOLUME = 1.0                  # Max volume by default; lower if needed
MICROSLEEP_ALARM_SECONDS = 0.8      # Trigger alarm when eyes stay closed continuously for this long
