"""Small causal recurrent mask estimator for speech-isolation experiments.

This is an offline research component, not production plugin code. The recurrent network
processes frames strictly left-to-right and exposes hidden state for streaming prototypes.
"""
import torch
import torch.nn.functional as F
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
        self.register_buffer("window", torch.hann_window(n_fft).clamp_min(1e-3), persistent=False)
        self.mask_net = CausalMaskNet(n_fft // 2 + 1, hidden, layers)

    def forward(self, mixture, state=None):
        """Offline differentiable path; input [batch, samples], output speech estimate."""
        if mixture.ndim != 2:
            raise ValueError("Expected mono [batch, samples] waveform")
        if mixture.shape[-1] < self.n_fft:
            raise ValueError("Waveform must contain at least one analysis window")
        # With center=False, ISTFT can silently zero-pad the tail when the final
        # analysis frame does not cover the complete input. Right-pad to the next
        # hop boundary, reconstruct the padded signal, then crop to the true length.
        original_length = mixture.shape[-1]
        remainder = (original_length - self.n_fft) % self.hop_length
        right_padding = (self.hop_length - remainder) % self.hop_length
        padded = F.pad(mixture, (0, right_padding)) if right_padding else mixture
        spectrum = torch.stft(
            padded, n_fft=self.n_fft, hop_length=self.hop_length,
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
            center=False, length=padded.shape[-1],
        )
        speech = speech[..., :original_length]
        return speech, mask, state


class StreamingSpeechMaskSeparator:
    """Stateful mono streaming wrapper around a trained/evaluated separator.

    Input chunks may have arbitrary lengths. The wrapper keeps the analysis FIFO,
    GRU state and overlap-add tail between calls. It emits one hop per completed
    analysis frame (20 ms initial frame availability at 48 kHz). Call flush() once
    at end-of-stream to pad the final partial frame and return the remaining tail.
    This is a research reference, not yet wired into the real-time plugin callback.
    """

    def __init__(self, model):
        if not isinstance(model, SpeechMaskSeparator):
            raise TypeError("model must be a SpeechMaskSeparator")
        self.model = model
        self.reset()

    def reset(self):
        self._input = None
        self._ola = None
        self._weight = None
        self._state = None
        self._received = 0
        self._emitted = 0
        self._closed = False

    def _process_frame(self, frame):
        window = self.model.window.to(device=frame.device, dtype=frame.dtype)
        spectrum = torch.fft.rfft(frame * window, n=self.model.n_fft)
        features = torch.log1p(spectrum.abs().unsqueeze(0).unsqueeze(0) * 10.0)
        mask, self._state = self.model.mask_net(features, self._state)
        estimated = spectrum * mask[0, 0]
        reconstructed = torch.fft.irfft(estimated, n=self.model.n_fft) * window
        weight = window.square()

        if self._ola is None:
            self._ola = torch.zeros_like(reconstructed)
            self._weight = torch.zeros_like(weight)
        self._ola = self._ola + reconstructed
        self._weight = self._weight + weight

        hop = self.model.hop_length
        denominator = self._weight[:hop].clamp_min(1e-8)
        output = self._ola[:hop] / denominator
        self._ola = torch.cat((self._ola[hop:], torch.zeros_like(self._ola[:hop])))
        self._weight = torch.cat((self._weight[hop:], torch.zeros_like(self._weight[:hop])))
        self._emitted += hop
        return output

    @torch.no_grad()
    def process_chunk(self, chunk):
        """Consume a mono 1-D tensor and return currently stable output samples."""
        if self._closed:
            raise RuntimeError("stream is closed; call reset() before processing again")
        if chunk.ndim != 1:
            raise ValueError("Expected mono [samples] chunk")
        if chunk.numel() == 0:
            return chunk.clone()
        if self._input is None:
            self._input = chunk.new_empty((0,))
        elif chunk.device != self._input.device or chunk.dtype != self._input.dtype:
            raise ValueError("All chunks in a stream must share device and dtype")
        self._received += int(chunk.numel())
        self._input = torch.cat((self._input, chunk))
        outputs = []
        n_fft, hop = self.model.n_fft, self.model.hop_length
        while self._input.numel() >= n_fft:
            frame = self._input[:n_fft]
            outputs.append(self._process_frame(frame))
            self._input = self._input[hop:]
        if outputs:
            return torch.cat(outputs)
        return chunk.new_empty((0,))

    @torch.no_grad()
    def flush(self):
        """Finish the stream, right-pad like the offline path, and crop to input length."""
        if self._closed:
            raise RuntimeError("stream already flushed")
        self._closed = True
        if self._received == 0:
            return self._input if self._input is not None else torch.empty(0)
        n_fft, hop = self.model.n_fft, self.model.hop_length
        remainder = (self._received - n_fft) % hop if self._received >= n_fft else 0
        right_padding = (hop - remainder) % hop
        if self._received < n_fft:
            right_padding += n_fft - self._received
        if self._input is None:
            raise RuntimeError("stream input state is missing")
        if right_padding or self._input.numel() >= n_fft:
            needed = max(0, n_fft - self._input.numel())
            frame = torch.nn.functional.pad(self._input, (0, needed))
            self._process_frame(frame)
            self._input = self._input.new_empty((0,))
        tail = self._ola / self._weight.clamp_min(1e-8)
        remaining = tail
        self._emitted += int(remaining.numel())
        padding = self._input.new_zeros((0,))
        del padding
        total = torch.cat((remaining.new_empty((0,)), remaining))
        # Only the true input duration is part of the requested waveform.
        start = max(0, self._emitted - int(total.numel()))
        del start
        return total[:max(0, self._received - (self._emitted - int(total.numel())))]
