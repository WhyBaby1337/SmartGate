"""Small end-to-end checks for paired separator training data and optimization."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from speech_separator import SpeechMaskSeparator
from train_speech_separator import PairedMixtures, grouped_split, load_rows, spectral_loss


class SeparatorTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def _make_dataset(self, root):
        sample_rate = 48000
        samples = 4800
        rows = []
        time = np.arange(samples, dtype=np.float32) / sample_rate
        for index in range(3):
            source_id = f"session_{index}"
            speech = (0.1 * np.sin(2 * np.pi * (180 + index * 30) * time)).astype(np.float32)
            noise = (0.015 * np.random.default_rng(index).standard_normal(samples)).astype(np.float32)
            mixture = speech + noise
            sf.write(root / f"{source_id}_mix.wav", np.column_stack((mixture, mixture)), sample_rate)
            sf.write(root / f"{source_id}_target.wav", np.column_stack((speech, speech)), sample_rate)
            rows.append({"id": source_id, "source_id": source_id,
                         "mixture": f"{source_id}_mix.wav",
                         "target_speech": f"{source_id}_target.wav"})
        manifest = root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return manifest

    def test_manifest_split_and_dataset_keep_source_groups_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._make_dataset(root)
            rows = load_rows(manifest)
            train, valid = grouped_split(rows, seed=3)
            train_ids = {row["source_id"] for row in train}
            valid_ids = {row["source_id"] for row in valid}
            self.assertTrue(train_ids)
            self.assertTrue(valid_ids)
            self.assertTrue(train_ids.isdisjoint(valid_ids))
            mixture, target = PairedMixtures(train[:1], root)[0]
            self.assertEqual(tuple(mixture.shape), (4800,))
            self.assertEqual(tuple(target.shape), (4800,))

    def test_manifest_rejects_duplicate_ids_and_missing_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._make_dataset(root)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[1]["id"] = rows[0]["id"]
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate mixture id"):
                load_rows(manifest)

            rows[1]["id"] = "unique_id"
            rows[1]["source_id"] = ""
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                load_rows(manifest)

    def test_dataset_rejects_non_finite_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._make_dataset(root)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            bad_audio = np.zeros(4800, dtype=np.float32)
            bad_audio[10] = np.nan
            sf.write(root / "bad_mix.wav", np.column_stack((bad_audio, bad_audio)), 48000)
            rows[0]["mixture"] = "bad_mix.wav"
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            loaded = load_rows(manifest)
            with self.assertRaisesRegex(ValueError, "Invalid or too-short audio pair"):
                PairedMixtures(loaded[:1], root)[0]

    def test_one_real_optimizer_step_has_finite_loss_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._make_dataset(root)
            rows = load_rows(manifest)
            train, _ = grouped_split(rows, seed=3)
            dataset = PairedMixtures(train, root)
            batch = [dataset[index] for index in range(len(dataset))]
            mixture = torch.stack([item[0] for item in batch])
            target = torch.stack([item[1] for item in batch])
            model = SpeechMaskSeparator(hidden=16, layers=1).train()
            estimate, _, _ = model(mixture)
            loss = torch.mean(torch.abs(estimate - target)) + 0.5 * spectral_loss(estimate, target)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(grad).all().item() for grad in gradients))


if __name__ == "__main__":
    unittest.main()
