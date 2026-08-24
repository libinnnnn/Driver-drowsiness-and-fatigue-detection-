import unittest

from detection_engine.socket_client import DetectionSocketClient


class DetectionSocketClientPreviewTests(unittest.TestCase):
    def test_emit_metrics_includes_frame_preview_when_present(self):
        client = DetectionSocketClient("http://localhost:5000")
        client.is_connected = True
        client.last_emit_time = 0.0

        captured = {}

        def fake_emit(event, payload):
            captured["event"] = event
            captured["payload"] = payload

        client.sio.emit = fake_emit

        client.emit_metrics(
            {
                "ear": 0.2,
                "mar": 0.3,
                "eye_state": "CLOSED",
                "ml_features": {},
                "frame_preview": "preview-data",
            },
            {
                "risk_level": "LOW",
                "prediction_class": "Alert",
                "fatigue_score": 0.2,
                "probabilities": {},
                "alarm_active": False,
                "microsleep_count": 0,
            },
        )

        self.assertEqual(captured["event"], "metrics_update")
        self.assertEqual(captured["payload"]["frame_preview"], "preview-data")


if __name__ == "__main__":
    unittest.main()
