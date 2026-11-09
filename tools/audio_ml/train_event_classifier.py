#!/usr/bin/env python3
"""Leakage-safe multi-label trainer for Smart Gate acoustic event detection."""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Independent sigmoid outputs allow speech+keyboard and speech+mouse at once.
LABELS = ("speech", "keyboard", "mouse", "other")
SR, FFT, HOP, MELS, FRAMES = 16000, 512, 160, 64, 32


def make_mel_bank():
    hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    edges = hz(np.linspace(mel(30.0), mel(SR / 2), MELS + 2))
    bins = np.floor((FFT + 1) * edges / SR).astype(int)
    bank = np.zeros((MELS, FFT // 2 + 1), np.float32)
    for i in range(1, MELS + 1):
        a, b, c = bins[i - 1:i + 2]
        b, c = max(b, a + 1), max(c, b + 1)
        for k in range(a, min(b, bank.shape[1])):
            bank[i - 1, k] = (k - a) / (b - a)
        for k in range(b, min(c, bank.shape[1])):
            bank[i - 1, k] = (c - k) / (c - b)
    return bank


BANK = make_mel_bank()
WINDOW = np.hanning(FFT).astype(np.float32)
WINDOW_SAMPLES = FFT + (FRAMES - 1) * HOP


def segment_window_range(start, end, feature_count):
    """Return feature-frame indices whose complete receptive fields fit inside [start, end)."""
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("segment bounds must be finite and satisfy 0 <= start < end")
    start_sample = math.ceil(start * SR)
    end_sample = math.floor(end * SR)
    first = max(0, (start_sample + HOP - 1) // HOP)
    last_exclusive = min(
        feature_count - FRAMES + 1,
        (end_sample - WINDOW_SAMPLES) // HOP + 1,
    )
    return first, max(first, last_exclusive)


def features(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != SR:
        g = math.gcd(int(rate), SR)
        mono = resample_poly(mono, SR // g, int(rate) // g).astype(np.float32)
    if mono.size < FFT or not np.isfinite(mono).all():
        raise ValueError(f"Invalid or too-short audio: {path}")
    frames = np.lib.stride_tricks.sliding_window_view(mono, FFT)[::HOP]
    spec = np.fft.rfft(frames * WINDOW, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    logmel = np.log(np.maximum(power @ BANK.T, 1e-10)).astype(np.float32)
    # Per-frame normalization is causal: future audio cannot alter earlier features.
    median = np.median(logmel, axis=1, keepdims=True)
    scale = np.maximum(logmel.std(axis=1, keepdims=True), 0.5)
    return np.clip((logmel - median) / scale, -5, 5), mono.size / SR


def load_dataset(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    xs, ys, groups = [], [], []
    sources = defaultdict(set)
    root = manifest_path.parent
    recording_ids = set()
    for rec in manifest["recordings"]:
        if not isinstance(rec, dict) or not rec.get("id") or not rec.get("path"):
            raise ValueError("Every recording needs a non-empty id and path")
        rid = str(rec["id"])
        if rid in recording_ids:
            raise ValueError(f"Duplicate recording id: {rid}")
        recording_ids.add(rid)
        if not isinstance(rec.get("segments"), list):
            raise ValueError(f"{rid}: segments must be a list")
        feat, duration = features((root / rec["path"]).resolve())
        previous_end = 0.0
        try:
            ordered_segments = sorted(rec["segments"], key=lambda s: float(s["start"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{rid}: each segment needs numeric start/end bounds") from exc
        for seg in ordered_segments:
            if not isinstance(seg, dict):
                raise ValueError(f"{rid}: each segment must be an object")
            try:
                start, end = float(seg["start"]), float(seg["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{rid}: segment needs numeric start/end bounds") from exc
            active = seg.get("labels", [])
            if (not isinstance(active, list) or
                    any(not isinstance(label, str) for label in active) or
                    len(active) != len(set(active))):
                raise ValueError(f"{rid}: labels must be a unique list of strings")
            if any(label not in LABELS for label in active):
                raise ValueError(f"{rid}: unsupported labels {active}")
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
                raise ValueError(f"{rid}: invalid segment {seg}")
            if start < previous_end:
                raise ValueError(f"{rid}: overlapping segments; combine active labels in one segment")
            previous_end = end
            target = np.asarray([float(label in active) for label in LABELS], np.float32)
            first, last = segment_window_range(start, end, len(feat))
            for pos in range(first, last, 16):
                xs.append(feat[pos:pos + FRAMES].T[None])
                ys.append(target)
                groups.append(rid)
                for label in active:
                    sources[label].add(rid)
    missing = {k: len(sources[k]) for k in LABELS if len(sources[k]) < 3}
    if missing:
        raise ValueError("Need >=3 independent recordings with positive examples per label; "
                         f"current counts: {missing}")
    if not xs:
        raise ValueError("No complete 320 ms labelled windows")
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(groups)


def split_by_recording(groups, targets, seed):
    names = sorted(set(groups.tolist()))
    rng = random.Random(seed)
    required = set(range(len(LABELS)))
    for _ in range(300):
        rng.shuffle(names)
        val_groups = set(names[:max(1, len(names) // 4)])
        mask = np.asarray([g in val_groups for g in groups])
        train_positive = set(np.flatnonzero(targets[~mask].sum(axis=0) > 0))
        valid_positive = set(np.flatnonzero(targets[mask].sum(axis=0) > 0))
        if train_positive == required and valid_positive == required:
            return np.flatnonzero(~mask), np.flatnonzero(mask)
    raise ValueError("Cannot create a recording-level split with every label in train and validation")


class EventCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 12, 3, padding=1, bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, len(LABELS)))
    def forward(self, x):
        return self.net(x)


def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()
    predicted = probabilities >= 0.5
    per_label, macro = {}, []
    for i, name in enumerate(LABELS):
        truth, guess = y[:, i] >= 0.5, predicted[:, i]
        tp = int(np.logical_and(truth, guess).sum())
        fp = int(np.logical_and(~truth, guess).sum())
        fn = int(np.logical_and(truth, ~guess).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_label[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "tp": tp, "fp": fp, "fn": fn}
        macro.append(f1)
    return per_label, float(np.mean(macro))


def positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=positive_int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    x, y, groups = load_dataset(args.manifest)
    train, valid = split_by_recording(groups, y, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventCNN().to(device)
    positive = y[train].sum(axis=0)
    negative = len(train) - positive
    pos_weight = torch.tensor(np.clip(negative / np.maximum(positive, 1), 1.0, 12.0),
                              dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train]),
                                      torch.from_numpy(y[train])),
                        batch_size=64, shuffle=True)
    best, state = -1.0, None
    for _ in range(args.epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            # Small log-energy perturbation and feature noise improve gain tolerance.
            bx = bx + (torch.rand(len(bx), 1, 1, 1, device=device) - 0.5) * 0.8
            bx = bx + torch.randn_like(bx) * 0.025
            opt.zero_grad(set_to_none=True)
            loss_fn(model(bx), by).backward()
            opt.step()
        _, score = evaluate(model, x[valid], y[valid], device)
        if score > best:
            best = score
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    metrics, macro_f1 = evaluate(model, x[valid], y[valid], device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model.cpu(), torch.zeros(1, 1, MELS, FRAMES), str(args.output),
                      input_names=["logmel"], output_names=["event_logits"], opset_version=17,
                      dynamic_axes={"logmel": {0: "batch"}, "event_logits": {0: "batch"}},
                      dynamo=False)
    result = {
        "labels": list(LABELS), "output_activation": "sigmoid per label",
        "sample_rate": SR, "mel_bins": MELS, "context_ms": (FFT + (FRAMES - 1) * HOP) * 1000 / SR,
        "train_recordings": sorted(set(groups[train].tolist())),
        "validation_recordings": sorted(set(groups[valid].tolist())),
        "validation_windows": int(len(valid)), "macro_f1": macro_f1,
        "per_label": metrics,
        "warning": "Do not deploy without independent real speech+keyboard and speech+mouse validation."
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
