#!/usr/bin/env python3
"""Leakage-safe trainer for Smart Gate's compact acoustic event CNN."""
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

LABELS = ("silence", "speech", "keyboard", "mouse", "mixed")
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
    med = np.median(logmel, axis=0, keepdims=True)
    scale = np.maximum(logmel.std(axis=0, keepdims=True), 0.5)
    return np.clip((logmel - med) / scale, -5, 5), mono.size / SR


def load_dataset(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    xs, ys, groups = [], [], []
    sources = defaultdict(set)
    root = manifest_path.parent
    for rec in manifest["recordings"]:
        rid = str(rec["id"])
        feat, duration = features((root / rec["path"]).resolve())
        previous_end = 0.0
        for seg in sorted(rec["segments"], key=lambda s: float(s["start"])):
            start, end, label = float(seg["start"]), float(seg["end"]), seg["label"]
            if label not in LABELS or not (0 <= start < end <= duration):
                raise ValueError(f"{rid}: invalid segment {seg}")
            if start < previous_end:
                raise ValueError(f"{rid}: overlapping labels; label the overlap as mixed")
            previous_end = end
            first = max(0, round(start * SR / HOP))
            last = min(len(feat), round(end * SR / HOP))
            for pos in range(first, last - FRAMES + 1, 16):
                xs.append(feat[pos:pos + FRAMES].T[None])
                ys.append(LABELS.index(label))
                groups.append(rid)
                sources[label].add(rid)
    missing = {k: len(sources[k]) for k in LABELS if len(sources[k]) < 3}
    if missing:
        raise ValueError("Need >=3 independent recordings per class for honest holdout; "
                         f"current counts: {missing}")
    if not xs:
        raise ValueError("No complete 320 ms labelled windows")
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64), np.asarray(groups)


def split_by_recording(groups, labels, seed):
    names = sorted(set(groups.tolist()))
    rng = random.Random(seed)
    for _ in range(200):
        rng.shuffle(names)
        nval = max(1, len(names) // 4)
        val_groups = set(names[:nval])
        mask = np.asarray([g in val_groups for g in groups])
        required = set(range(len(LABELS)))
        if set(labels[mask]) == required and set(labels[~mask]) == required:
            return np.flatnonzero(~mask), np.flatnonzero(mask)
    raise ValueError("Cannot make a recording-level split containing every class in both sets")


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
        pred = model(torch.from_numpy(x).to(device)).argmax(1).cpu().numpy()
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for a, b in zip(y, pred):
        matrix[a, b] += 1
    per_class = {}
    for i, name in enumerate(LABELS):
        tp = int(matrix[i, i])
        precision = tp / max(1, int(matrix[:, i].sum()))
        recall = tp / max(1, int(matrix[i, :].sum()))
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / max(1e-12, precision + recall)}
    return matrix, per_class, float(np.mean([v["f1"] for v in per_class.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    x, y, groups = load_dataset(args.manifest)
    train, valid = split_by_recording(groups, y, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train]),
                                      torch.from_numpy(y[train])),
                        batch_size=64, shuffle=True)
    best, state = -1.0, None
    for _ in range(args.epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            bx = bx + torch.randn_like(bx) * 0.025
            opt.zero_grad(set_to_none=True)
            loss_fn(model(bx), by).backward()
            opt.step()
        _, _, score = evaluate(model, x[valid], y[valid], device)
        if score > best:
            best = score
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    matrix, metrics, macro_f1 = evaluate(model, x[valid], y[valid], device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model.cpu(), torch.zeros(1, 1, MELS, FRAMES), str(args.output),
                      input_names=["logmel"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    result = {
        "labels": list(LABELS), "sample_rate": SR, "mel_bins": MELS,
        "context_ms": FRAMES * HOP * 1000 / SR,
        "train_recordings": sorted(set(groups[train].tolist())),
        "validation_recordings": sorted(set(groups[valid].tolist())),
        "validation_windows": int(len(valid)), "macro_f1": macro_f1,
        "per_class": metrics, "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "warning": "Do not deploy without independent real mixed speech+click recordings."
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
