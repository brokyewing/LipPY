from __future__ import annotations

import time
import unittest
from fastapi.testclient import TestClient

from backend.main import app


class LipPYApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_models(self):
        res_h = self.client.get("/api/health")
        self.assertEqual(res_h.status_code, 200)
        body_h = res_h.json()
        self.assertEqual(body_h["status"], "ok")
        self.assertIn(body_h["mode"], ("checkpoint", "demo"))

        res_m = self.client.get("/api/models")
        self.assertEqual(res_m.status_code, 200)
        models = res_m.json()["models"]
        ids = [m["id"] for m in models]
        self.assertIn("video", ids)
        self.assertIn("av-fusion", ids)
        for m in models:
            self.assertTrue(m["available"])
            self.assertTrue(len(m["decode_modes"]) >= 2)

    def test_samples_and_comparative_inference(self):
        res_s = self.client.get("/api/samples")
        self.assertEqual(res_s.status_code, 200)
        samples = res_s.json()["samples"]
        self.assertGreaterEqual(len(samples), 4)

        res_vid = self.client.get("/api/samples/tr-hedef/video.mp4")
        self.assertEqual(res_vid.status_code, 200)
        self.assertGreater(len(res_vid.content), 10000)

        res_greedy = self.client.post(
            "/api/infer_sample",
            data={"sample_id": "tr-hedef", "model": "video", "decode_mode": "greedy"},
        )
        self.assertEqual(res_greedy.status_code, 200)
        data_greedy = res_greedy.json()
        self.assertTrue(data_greedy["success"])
        self.assertIn("gözden geçirme", data_greedy["text"])

        res_beam = self.client.post(
            "/api/infer_sample",
            data={"sample_id": "tr-hedef", "model": "video", "decode_mode": "beam_lm"},
        )
        self.assertEqual(res_beam.status_code, 200)
        data_beam = res_beam.json()
        self.assertTrue(data_beam["success"])
        self.assertIn("çünkü hedeflerimizi gözden geçirme", data_beam["text"])
        self.assertEqual(data_beam["processing"]["metrics"]["av_fusion_per"], 24.7)

    def test_infer_upload_and_validation(self):
        res_empty = self.client.post(
            "/api/infer",
            files={"file": ("empty.mp4", b"", "video/mp4")},
        )
        self.assertEqual(res_empty.status_code, 400)

        from backend.demo_samples import get_sample_video_path
        vpath = get_sample_video_path("tr-merkez")
        self.assertIsNotNone(vpath)
        video_bytes = vpath.read_bytes()

        res_ok = self.client.post(
            "/api/infer",
            files={"file": ("test_videosu54_0188.mp4", video_bytes, "video/mp4")},
            data={"model": "video", "decode_mode": "beam_lm"},
        )
        self.assertEqual(res_ok.status_code, 200)
        payload = res_ok.json()
        self.assertTrue(payload["success"])
        self.assertIn("kendini çok abartmamak ben merkezli olmamak bencil", payload["text"])

    def test_long_video_demo_and_subtitles(self):
        res_job = self.client.post("/api/infer_long_demo", data={"model": "av-fusion"})
        self.assertEqual(res_job.status_code, 200)
        job_id = res_job.json()["job_id"]

        stream_res = self.client.get(f"/api/jobs/{job_id}/stream")
        self.assertEqual(stream_res.status_code, 200)
        self.assertIn('"status": "done"', stream_res.text)

        st = self.client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(st["status"], "done")
        self.assertEqual(st["n_segments"], 4)

        vtt_res = self.client.get(f"/api/jobs/{job_id}/subtitles.vtt")
        self.assertEqual(vtt_res.status_code, 200)
        self.assertIn("WEBVTT", vtt_res.text)

        srt_res = self.client.get(f"/api/jobs/{job_id}/subtitles.srt")
        self.assertEqual(srt_res.status_code, 200)
        self.assertIn("-->", srt_res.text)


if __name__ == "__main__":
    unittest.main()
