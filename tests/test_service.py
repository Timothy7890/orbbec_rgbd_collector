from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from rgbd_collector.app import CollectorService
from rgbd_collector.camera import CameraConfig

from test_storage import camera_metadata, make_frame


class FakeCamera:
    def __init__(self) -> None:
        self.index = 0

    def latest(self):
        self.index += 1
        return make_frame(self.index)

    def metadata(self):
        return camera_metadata()

    def status(self):
        return {"running": True, "frames": self.index, "error": None}

    def start(self):
        return None

    def stop(self):
        return None


class CollectorServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_service(self) -> CollectorService:
        service = CollectorService(CameraConfig(), self.root)
        service.camera = FakeCamera()
        return service

    def test_manual_capture_and_session_close(self) -> None:
        service = self.make_service()
        frame_id = service.capture(trigger="manual", session_name="manual")
        self.assertTrue(frame_id.startswith("000001_"))
        closed = service.close_session()
        self.assertIsNotNone(closed)
        self.assertEqual(closed["saved"], 1)
        self.assertIsNone(service.status()["session"])

    def test_interval_recording_stops_at_max_frames(self) -> None:
        service = self.make_service()
        service.start_recording(
            interval_s=0.1, max_frames=3, session_name="interval"
        )
        deadline = time.monotonic() + 2.0
        while service.status()["recording"]["active"] and time.monotonic() < deadline:
            time.sleep(0.02)
        recording = service.status()["recording"]
        self.assertFalse(recording["active"])
        self.assertEqual(recording["captured"], 3)
        closed = service.close_session()
        self.assertEqual(closed["saved"], 3)
        self.assertEqual(closed["failed"], 0)


if __name__ == "__main__":
    unittest.main()
