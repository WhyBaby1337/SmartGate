#!/usr/bin/env python3
"""Generate paired noisy mixtures and clean-speech targets for Smart Gate.

Input manifest describes independently recorded stems. Each output example contains
an identical-time noisy mixture and its clean speech reference. Source recording IDs
are retained so train/validation/test splits can be made without source leakage.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 48000
DEFAULT_SECONDS = 2.0


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all() or audio.shape[0] == 0:
        raise ValueError(f"Invalid audio: {path}")
    if sr != TARGET_SR:
        divisor = math.gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // divisor, int(sr) // divisor, axis=0)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return np.asarray(audio, dtype=np.float32)


def choose_crop(audio, length, rng):
    if len(audio) >= length:
        start = rng.randrange(0, len(audio) - length + 1)
        return audio[start:start + length].copy()
    repeats = int(math.ceil(length / len(audio)))
    tiled = np.tile(audio, (repeats, 1))
    start = rng.randrange(0, len(tiled) - length + 1)
    return tiled[start:start + length].copy()


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def scale_to_snr(noise, speech, snr_db, rng):
    noise_rms = rms(noise)
    if noise_rms < 1e-7:
        return noise
    speech_rms = rms(speech)
    if speech_rms < 1e-7:
        return noise * (10.0 ** (rng.uniform(-8.0, 8.0) / 20.0))
    desired = speech_rms / (10.0 ** (snr_db / 20.0))
    return noise * (desired / noise_rms)


def read_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("sources"), list) or not data["sources"]:
        raise ValueError("Manifest must contain a non-empty 'sources' array")
    root = path.parent
    result = []
    for item in data["sources"]:
        if not item.get("id") or not item.get("speech"):
            raise ValueError("Every source needs a unique id and speech path")
        entry = {"id": str(item["id"])}
        for key in ("speech", "keyboard", "mouse", "background"):
            value = item.get(key)
            entry[key] = (root / value).resolve() if value else None
        result.append(entry)
    ids = [item["id"] for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Source IDs must be unique")
    return result


def generate(manifest_path, output_dir, count, seconds, seed):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    sources = read_manifest(manifest_path)
    cache = {}
    for source in sources:
        for key in ("speech", "keyboard", "mouse", "background"):
            path = source[key]
            if path and path not in cache:
                cache[path] = load_audio(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    mixture_dir = output_dir / "mixtures"
    target_dir = output_dir / "targets"
    mixture_dir.mkdir(exist_ok=True)
    target_dir.mkdir(exist_ok=True)
    length = max(1, round(seconds * TARGET_SR))
    index_path = output_dir / "manifest.jsonl"
    records = []

    for number in range(count):
        source = rng.choice(sources)
        speech_path = source["speech"]
        speech_source = cache[speech_path]
        speech = choose_crop(speech_source, length, rng)
        # A fraction of examples are deliberately non-speech to teach zero-output cases.
        has_speech = rng.random() >= 0.15
        if not has_speech:
            speech.fill(0.0)

        mixture = speech.copy()
        active = []
        if has_speech:
            active.append("speech")

        for label in ("keyboard", "mouse", "background"):
            path = source[label]
            if path is None:
                continue
            probability = {"keyboard": 0.58, "mouse": 0.58, "background": 0.35}[label]
            if rng.random() >= probability:
                continue
            stem = choose_crop(cache[path], length, rng)
            snr_range = {
                "keyboard": (-8.0, 18.0),
                "mouse": (-8.0, 18.0),
                "background": (0.0, 24.0),
            }[label]
            snr = rng.uniform(*snr_range)
            stem = scale_to_snr(stem, speech, snr, rng) if has_speech else stem * (10 ** (rng.uniform(-20, -4) / 20))
            mixture += stem
            active.append(label)

        # Shared gain preserves the target-to-mixture relationship. Soft peak control
        # avoids hard clipping while keeping the clean reference synchronized.
        peak = float(np.max(np.abs(mixture)))
        shared_gain = min(1.0, 0.98 / max(peak, 1e-8))
        mixture *= shared_gain
        target = speech * shared_gain

        # Tiny independent sensor-like noise prevents the model learning digital silence.
        mixture += np_rng.normal(0.0, 1e-5, mixture.shape).astype(np.float32)
        stem_id = f"{number:07d}"
        mix_rel = Path("mixtures") / f"{stem_id}.wav"
        target_rel = Path("targets") / f"{stem_id}.wav"
        sf.write(output_dir / mix_rel, mixture, TARGET_SR, subtype="PCM_16")
        sf.write(output_dir / target_rel, target, TARGET_SR, subtype="PCM_16")
        records.append({
            "id": stem_id,
            "source_id": source["id"],
            "mixture": mix_rel.as_posix(),
            "target_speech": target_rel.as_posix(),
            "active_sources": active,
            "sample_rate": TARGET_SR,
            "samples": length,
            "duration_seconds": seconds,
        })

    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "examples": len(records),
        "sources": len(sources),
        "unique_source_ids": sorted({r["source_id"] for r in records}),
        "output_manifest": str(index_path),
        "sample_rate": TARGET_SR,
        "duration_seconds": seconds,
        "seed": seed,
        "note": "Split by source_id before training; never randomly split generated clips."
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.count < 1 or args.seconds <= 0:
        parser.error("--count and --seconds must be positive")
    generate(args.manifest, args.output_dir, args.count, args.seconds, args.seed)


if __name__ == "__main__":
    main()
