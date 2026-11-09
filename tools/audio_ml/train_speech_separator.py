#!/usr/bin/env python3
"""Offline trainer for the causal recurrent speech-mask prototype."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from speech_separator import SpeechMaskSeparator

SR = 48000


class PairedMixtures(Dataset):
    def __init__(self, rows, root):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        mixture, mix_sr = sf.read(self.root / row["mixture"], dtype="float32", always_2d=True)
        target, target_sr = sf.read(self.root / row["target_speech"], dtype="float32", always_2d=True)
        if mix_sr != SR or target_sr != SR or mixture.shape != target.shape:
            raise ValueError(f"Unaligned pair: {row['id']}")
        # The first prototype is mono; preserve timing by applying the same downmix to both.
        mixture = mixture.mean(axis=1)
        target = target.mean(axis=1)
        return torch.from_numpy(mixture.copy()), torch.from_numpy(target.copy())


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Paired manifest is empty")
    for row in rows:
        if not {"source_id", "mixture", "target_speech"}.issubset(row):
            raise ValueError(f"Invalid manifest row: {row}")
    return rows


def grouped_split(rows, seed):
    sources = sorted({row["source_id"] for row in rows})
    if len(sources) < 3:
        raise ValueError("Need at least three independent source sessions for train/validation")
    rng = random.Random(seed)
    rng.shuffle(sources)
    validation_sources = set(sources[:max(1, round(len(sources) * 0.2))])
    train = [r for r in rows if r["source_id"] not in validation_sources]
    valid = [r for r in rows if r["source_id"] in validation_sources]
    if not train or not valid:
        raise ValueError("Could not create non-empty source-disjoint split")
    return train, valid


def spectral_loss(estimate, target):
    terms = []
    for fft, hop in ((480, 120), (960, 240), (1920, 480)):
        window = torch.hann_window(fft, device=estimate.device).clamp_min(1e-3)
        a = torch.stft(estimate, fft, hop, fft, window, center=False, return_complex=True).abs()
        b = torch.stft(target, fft, hop, fft, window, center=False, return_complex=True).abs()
        terms.append(torch.mean(torch.abs(torch.log1p(a * 10.0) - torch.log1p(b * 10.0))))
    return torch.stack(terms).mean()


def score(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for mixture, target in loader:
            mixture, target = mixture.to(device), target.to(device)
            estimate, _, _ = model(mixture)
            losses.append(float((torch.mean(torch.abs(estimate - target)) +
                                 0.5 * spectral_loss(estimate, target)).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(args.manifest)
    train_rows, valid_rows = grouped_split(rows, args.seed)
    root = args.manifest.parent
    train_loader = DataLoader(PairedMixtures(train_rows, root), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    valid_loader = DataLoader(PairedMixtures(valid_rows, root), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechMaskSeparator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_loss, best_state = float("inf"), None

    for epoch in range(args.epochs):
        model.train()
        running = []
        for mixture, target in train_loader:
            mixture, target = mixture.to(device), target.to(device)
            estimate, _, _ = model(mixture)
            loss = torch.mean(torch.abs(estimate - target)) + 0.5 * spectral_loss(estimate, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(float(loss.detach().cpu()))
        validation_loss = score(model, valid_loader, device)
        print(json.dumps({"epoch": epoch + 1, "train_loss": float(np.mean(running)),
                          "validation_loss": validation_loss}))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "sample_rate": SR, "n_fft": 960,
                "hop_length": 480, "hidden": 96, "layers": 2,
                "validation_loss": best_loss,
                "train_source_ids": sorted({r["source_id"] for r in train_rows}),
                "validation_source_ids": sorted({r["source_id"] for r in valid_rows}),
                "warning": "Research checkpoint only; not validated for live plugin use."},
               args.output)
    print(json.dumps({"checkpoint": str(args.output), "best_validation_loss": best_loss,
                      "train_examples": len(train_rows), "validation_examples": len(valid_rows),
                      "device": str(device), "status": "offline research only"}, indent=2))


if __name__ == "__main__":
    main()
