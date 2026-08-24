"""
Audio alarm controller for driver drowsiness monitoring.

Generates a synthetic emergency-style alarm tone at startup if no audio file exists,
then plays it looped and non-blocking via pygame.mixer when the driver state becomes
critical or drowsy for several consecutive frames.
"""

import logging
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover - optional dependency
    pygame = None

from detection_engine.config import ALARM_VOLUME, MICROSLEEP_ALARM_SECONDS

logger = logging.getLogger(__name__)


class AlarmController:
    """Debounced alarm state machine with optional non-blocking audio playback."""

    def __init__(
        self,
        trigger_seconds: float = MICROSLEEP_ALARM_SECONDS,
        alarm_file: Optional[str] = None,
        initialize_audio: bool = True,
        volume: Optional[float] = None,
    ):
        self.trigger_seconds = max(0.0, trigger_seconds)
        self.volume = ALARM_VOLUME if volume is None else volume
        self.alarm_file = Path(alarm_file) if alarm_file else self._default_alarm_path()
        self._is_alarm_playing = False
        self._audio_ready = False
        self._sound = None
        self.last_transitioned_to_active = False

        if initialize_audio:
            self.initialize()

    @property
    def is_alarm_playing(self) -> bool:
        return self._is_alarm_playing

    @property
    def default_alarm_path(self) -> Path:
        return self.alarm_file

    def _default_alarm_path(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "outputs" / "alarm.wav"

    def initialize(self) -> None:
        """Create the alarm asset if missing and initialize pygame mixer if possible."""
        self._ensure_alarm_file()

        if pygame is None:
            logger.warning("[ALARM] pygame is not installed; audio playback is unavailable.")
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        except Exception as exc:
            logger.warning(f"[ALARM] pygame mixer initialization failed ({exc}).")
            return

        if pygame.mixer.get_init():
            try:
                self._sound = pygame.mixer.Sound(str(self.alarm_file))
                self._sound.set_volume(self.volume)
                self._audio_ready = True
                logger.info(f"[ALARM] Audio ready: {self.alarm_file}")
            except Exception as exc:
                logger.warning(f"[ALARM] Unable to load alarm sound ({exc}).")

    def _ensure_alarm_file(self) -> None:
        self.alarm_file.parent.mkdir(parents=True, exist_ok=True)
        if self.alarm_file.exists():
            return

        self._generate_alarm_wav(self.alarm_file)

    def _generate_alarm_wav(self, output_path: Path) -> None:
        sample_rate = 22050
        duration_seconds = 1.25
        total_samples = int(sample_rate * duration_seconds)
        t = np.arange(total_samples) / sample_rate

        # Two-tone siren with alternating harsh pulses.
        tone_a = np.sin(2 * np.pi * 800 * t)
        tone_b = np.sin(2 * np.pi * 1200 * t)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
        pattern = np.where(np.sin(2 * np.pi * 6.0 * t) >= 0, tone_a, tone_b)
        wave_data = (pattern * envelope * 0.75 * 32767).astype(np.int16)

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(wave_data.tobytes())

    def update_from_eye_state(self, eyes_closed: bool, closure_duration_seconds: float) -> bool:
        """Update the alarm based solely on continuous eye-closure duration."""
        self.last_transitioned_to_active = False

        if not eyes_closed:
            self._stop_alarm()
            return False

        if self._is_alarm_playing:
            return True

        if closure_duration_seconds >= self.trigger_seconds:
            self._start_alarm()
            self.last_transitioned_to_active = True
            return True

        return False

    def _start_alarm(self) -> None:
        if self._is_alarm_playing:
            return

        self._is_alarm_playing = True

        if self._audio_ready and self._sound is not None:
            try:
                self._sound.play(loops=-1)
            except Exception as exc:
                logger.warning(f"[ALARM] Playback failed ({exc}).")

    def _stop_alarm(self) -> None:
        if not self._is_alarm_playing:
            return

        self._critical_frame_count = 0
        self._alert_frame_count = 0
        self._is_alarm_playing = False

        if self._audio_ready and self._sound is not None:
            try:
                self._sound.stop()
            except Exception as exc:
                logger.warning(f"[ALARM] Stop failed ({exc}).")

    def stop(self) -> None:
        self._stop_alarm()

    def close(self) -> None:
        self.stop()
        if pygame is not None and pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
            except Exception:
                pass
