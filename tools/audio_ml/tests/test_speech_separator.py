"""Runtime tests for the first speech-separator prototype."""
import unittest

import torch

from speech_separator import CausalMaskNet, SpeechMaskSeparator


class SpeechSeparatorRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_waveform_length_and_finite_values_at_stft_boundaries(self):
        model = SpeechMaskSeparator(hidden=16, layers=1).eval()
        for length in (960, 961, 1000, 1439, 1440, 1441, 48000):
            with self.subTest(length=length):
                mixture = torch.randn(1, length)
                with torch.no_grad():
                    estimate, mask, state = model(mixture)
                self.assertEqual(tuple(estimate.shape), (1, length))
                self.assertTrue(torch.isfinite(estimate).all().item())
                self.assertTrue(torch.isfinite(mask).all().item())
                self.assertEqual(tuple(state.shape), (1, 1, 16))

    def test_forward_backward_produces_finite_gradients(self):
        model = SpeechMaskSeparator(hidden=16, layers=1).train()
        mixture = torch.randn(2, 4800, requires_grad=True)
        target = torch.randn(2, 4800)
        estimate, mask, _ = model(mixture)
        loss = (estimate - target).abs().mean() + mask.mean() * 0.01
        loss.backward()
        self.assertIsNotNone(mixture.grad)
        self.assertTrue(torch.isfinite(mixture.grad).all().item())
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_recurrent_state_matches_single_pass(self):
        model = CausalMaskNet(bins=33, hidden=16, layers=2).eval()
        features = torch.randn(2, 17, 33)
        with torch.no_grad():
            full_mask, full_state = model(features)
            first_mask, state = model(features[:, :6])
            second_mask, state = model(features[:, 6:11], state)
            third_mask, state = model(features[:, 11:], state)
        chunked_mask = torch.cat((first_mask, second_mask, third_mask), dim=1)
        self.assertTrue(torch.allclose(full_mask, chunked_mask, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(full_state, state, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
