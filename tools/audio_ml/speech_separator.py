"""Small causal recurrent mask estimator for speech-isolation experiments.

This is an offline research component, not production plugin code. The recurrent network
processes frames strictly left-to-right and exposes hidden state for streaming prototypes.
"""
import torch
from torch import nn


class CausalMaskNet(nn.Module):
    """Unidirectional GRU that estimates a conservative magnitude mask per STFT bin."""

    def __init__(self, bins=481, hidden=96, layers=2):
        super().__init__()
        self.bins = bins
        self.input_norm = nn.LayerNorm(bins)
        self.encoder = nn.Sequential(nn.Linear(bins, hidden), nn.PReLU())
        self.recurrent = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
        )
        self.decoder = nn.Sequential(nn.Linear(hidden, bins), nn.Sigmoid())

    def forward(self, log_magnitude, state=None):
        """Return [batch, frames, bins] mask and updated recurrent state."""
        if log_magnitude.ndim != 3 or log_magnitude.shape[-1] != self.bins:
            raise ValueError(f"Expected [batch, frames, {self.bins}] input")
        encoded = self.encoder(self.input_norm(log_magnitude))
        recurrent, state = self.recurrent(encoded, state)
        return self.decoder(recurrent), state


class SpeechMaskSeparator(nn.Module):
    """Causal-frame magnitude-mask separator using a 20 ms window / 10 ms hop at 48 kHz."""

    def __init__(self, n_fft=960, hop_length=480, hidden=96, layers=2):
        super().__init__()
        if n_fft != 960 or hop_length != 480:
            raise ValueError("Initial low-latency profile is fixed at 960/480 samples")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.mask_net = CausalMaskNet(n_fft // 2 + 1, hidden, layers)

    def forward(self, mixture, state=None):
        """Offline differentiable path; input [batch, samples], output speech estimate."""
        if mixture.ndim != 2:
            raise ValueError("Expected mono [batch, samples] waveform")
        if mixture.shape[-1] < self.n_fft:
            raise ValueError("Waveform must contain at least one analysis window")
        spectrum = torch.stft(
            mixture, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.window.to(mixture),
            center=False, return_complex=True,
        )
        magnitude = spectrum.abs()
        features = torch.log1p(magnitude * 10.0).transpose(1, 2)
        mask, state = self.mask_net(features, state)
        estimated = spectrum * mask.transpose(1, 2)
        speech = torch.istft(
            estimated, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.window.to(mixture),
            center=False, length=mixture.shape[-1],
        )
        return speech, mask, state
