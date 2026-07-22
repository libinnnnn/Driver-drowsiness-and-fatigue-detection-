import unittest

from detection_engine.audio import AlarmController
from detection_engine.config import MICROSLEEP_ALARM_SECONDS


class AlarmControllerTests(unittest.TestCase):
    def test_alarm_requires_prolonged_eye_closure_to_start(self):
        controller = AlarmController(initialize_audio=False)

        self.assertFalse(controller.update_from_eye_state(False, 0.5))
        self.assertFalse(controller.update_from_eye_state(True, 1.4))
        self.assertTrue(controller.update_from_eye_state(True, MICROSLEEP_ALARM_SECONDS + 0.1))

    def test_alarm_stops_immediately_when_eyes_open_again(self):
        controller = AlarmController(initialize_audio=False)

        self.assertTrue(controller.update_from_eye_state(True, MICROSLEEP_ALARM_SECONDS + 0.1))
        self.assertFalse(controller.update_from_eye_state(False, 0.0))


if __name__ == "__main__":
    unittest.main()
