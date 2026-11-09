import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from build_speech_mixtures import TARGET_SR, aligned_crop, choose_crop, generate, place_activity


class MixtureBuilderTests(unittest.TestCase):
    def test_generator_rejects_invalid_count_and_nonfinite_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "missing.json"
            with self.assertRaisesRegex(ValueError, "count"):
                generate(manifest, root / "out", 0, 1.0, 1)
            with self.assertRaisesRegex(ValueError, "finite positive"):
                generate(manifest, root / "out", 1, float("inf"), 1)

    def test_manifest_rejects_non_object_source_entry(self):
        from build_speech_mixtures import read_manifest
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.json"
            path.write_text(json.dumps({"sources": ["not-an-object"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                read_manifest(path)

    def test_short_stem_is_padded_not_looped(self):
        source = np.array([[0.25, -0.25], [0.5, -0.5]], dtype=np.float32)
        result = choose_crop(source, 8, __import__("random").Random(4))
        self.assertEqual(result.shape, (8, 2))
        self.assertEqual(np.count_nonzero(result), 4)
        self.assertTrue(np.all(np.isin(result[:, 0], [0.0, 0.25, 0.5])))

    def test_aligned_crop_preserves_shared_source_timing(self):
        source = np.arange(20, dtype=np.float32).reshape(10, 2)
        crop = aligned_crop(source, 3, 5)
        np.testing.assert_array_equal(crop, source[3:8])
        padded = aligned_crop(source, 8, 5)
        np.testing.assert_array_equal(padded[:2], source[8:10])
        np.testing.assert_array_equal(padded[2:], np.zeros((3, 2), dtype=np.float32))
        offset_crop = aligned_crop(source, 5, 5, stem_start=3)
        np.testing.assert_array_equal(offset_crop, source[2:7])

    def test_transient_activity_is_not_forced_to_clip_start(self):
        stem = np.ones((TARGET_SR, 2), dtype=np.float32)
        starts = set()
        for seed in range(20):
            placed = place_activity(stem, TARGET_SR, __import__("random").Random(seed), "mouse")
            nonzero = np.flatnonzero(np.any(placed != 0, axis=1))
            self.assertGreater(len(nonzero), 0)
            starts.add(int(nonzero[0]))
            self.assertLess(len(nonzero), TARGET_SR)
        self.assertGreater(len(starts), 1)

    def test_generator_respects_event_stem_start_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            speech = np.full((TARGET_SR // 2, 1), 0.05, dtype=np.float32)
            mouse = np.zeros((TARGET_SR // 4, 1), dtype=np.float32)
            mouse[0] = 0.9
            sf.write(root / "speech.wav", speech, TARGET_SR)
            sf.write(root / "mouse.wav", mouse, TARGET_SR)
            manifest = {"sources": [{
                "id": "offset-session", "speech": "speech.wav", "mouse": "mouse.wav",
                "start_seconds": {"mouse": 0.125}
            }]}
            (root / "sources.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "out"
            generate(root / "sources.json", output, 40, 0.5, 17)
            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            event_rows = [row for row in rows if "mouse" in row["active_sources"]]
            self.assertTrue(event_rows, "seed should produce at least one mouse-positive example")
            row = event_rows[0]
            mix = sf.read(output / row["mixture"], always_2d=True, dtype="float32")[0]
            target = sf.read(output / row["target_speech"], always_2d=True, dtype="float32")[0]
            difference = np.max(np.abs(mix - target), axis=1)
            self.assertEqual(int(np.argmax(difference)), round(0.125 * TARGET_SR))

    def test_generator_writes_aligned_stereo_pairs_and_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            speech = np.zeros((TARGET_SR, 1), dtype=np.float32)
            speech[100:900] = 0.1
            click = np.zeros((TARGET_SR // 4, 1), dtype=np.float32)
            click[20:40] = 0.8
            sf.write(root / "speech.wav", speech, TARGET_SR)
            sf.write(root / "mouse.wav", click, TARGET_SR)
            manifest = {"sources": [{"id": "session-1", "speech": "speech.wav",
                                     "mouse": "mouse.wav"}]}
            (root / "sources.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "out"
            generate(root / "sources.json", output, 4, 0.25, 17)
            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            for row in rows:
                mix, mix_sr = sf.read(output / row["mixture"], always_2d=True)
                target, target_sr = sf.read(output / row["target_speech"], always_2d=True)
                self.assertEqual(mix_sr, TARGET_SR)
                self.assertEqual(target_sr, TARGET_SR)
                self.assertEqual(mix.shape, target.shape)
                self.assertEqual(mix.shape, (TARGET_SR // 4, 2))
                self.assertEqual(row["source_id"], "session-1")
                self.assertEqual(row["samples"], len(mix))


if __name__ == "__main__":
    unittest.main()
