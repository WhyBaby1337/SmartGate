#!/usr/bin/env python3
"""Generate paired noisy mixtures and clean-speech targets from isolated stems.

Stems within one source bundle must share a recording clock. Short stems are zero-padded,
never looped; transient events are placed at randomized offsets to avoid fixed alignment.
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
    """Random crop long stems; place short stems once with silence padding."""
    if len(audio) >= length:
        start = rng.randrange(0, len(audio) - length + 1)
        return audio[start:start + length].copy()
    result = np.zeros((length, audio.shape[1]), dtype=np.float32)
    offset = rng.randrange(0, length - len(audio) + 1)
    result[offset:offset + len(audio)] = audio
    return result
def aligned_crop(audio, start, length):
    """Crop a stem on a shared recording clock, padding beyond its boundaries."""
    result = np.zeros((length, audio.shape[1]), dtype=np.float32)
    source_start = max(0, int(start))
    source_end = min(len(audio), int(start) + length)
    if source_end > source_start:
        destination_start = source_start - int(start)
        result[destination_start:destination_start + source_end - source_start] = audio[source_start:source_end]
    return result




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


def place_activity(stem, length, rng, label):
    """Return a zero-padded stem with randomized duration and offset."""
    if label == "background":
        fraction = rng.uniform(0.65, 1.0)
    else:
        fraction = rng.uniform(0.08, 0.55)
    active_length = max(1, min(length, round(length * fraction)))
    crop = choose_crop(stem, active_length, rng)
    result = np.zeros((length, stem.shape[1]), dtype=np.float32)
    offset = rng.randrange(0, length - active_length + 1)
    result[offset:offset + active_length] = crop
    return result


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


def generate(manifest_path, output_dir, count, seconds, seed, randomize_event_offsets=False):
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
        # All stems in a source bundle share one recording clock. Select one
        # common crop position so speech/click/background alignment is retained.
        source_length = max(len(cache[source[key]]) for key in
                            ("speech", "keyboard", "mouse", "background") if source[key] is not None)
        max_start = max(0, source_length - length)
        shared_start = rng.randrange(max_start + 1) if max_start else 0
        speech = aligned_crop(cache[source["speech"]], shared_start, length)
        has_speech = rng.random() >= 0.15
        if not has_speech:
            speech.fill(0.0)

        mixture = speech.copy()
        active = ["speech"] if has_speech else []
        for label in ("keyboard", "mouse", "background"):
            path = source[label]
            if path is None:
                continue
            probability = {"keyboard": 0.58, "mouse": 0.58, "background": 0.35}[label]
            if rng.random() >= probability:
                continue
            stem = aligned_crop(cache[path], shared_start, length)
            # Temporal jitter is explicit augmentation, never the default:
            # independently shifting stems would destroy their real alignment.
            if randomize_event_offsets:
                stem = place_activity(stem, length, rng, label)
            snr_range = {
                "keyboard": (-8.0, 18.0),
                "mouse": (-8.0, 18.0),
                "background": (0.0, 24.0),
            }[label]
            if has_speech:
                stem = scale_to_snr(stem, speech, rng.uniform(*snr_range), rng)
            else:
                stem *= 10 ** (rng.uniform(-20, -4) / 20)
            mixture += stem
            if rms(stem) >= 1e-7:
                active.append(label)

        peak = float(np.max(np.abs(mixture)))
        shared_gain = min(1.0, 0.98 / max(peak, 1e-8))
        mixture *= shared_gain
        target = speech * shared_gain
        mixture += np_rng.normal(0.0, 1e-5, mixture.shape).astype(np.float32)

        stem_id = f"{number:07d}"
        mix_rel = Path("mixtures") / f"{stem_id}.wav"
        target_rel = Path("targets") / f"{stem_id}.wav"
        sf.write(output_dir / mix_rel, mixture, TARGET_SR, subtype="PCM_16")
        sf.write(output_dir / target_rel, target, TARGET_SR, subtype="PCM_16")
        records.append({
            "id": stem_id, "source_id": source["id"],
            "mixture": mix_rel.as_posix(), "target_speech": target_rel.as_posix(),
            "active_sources": active, "sample_rate": TARGET_SR,
            "samples": length, "duration_seconds": seconds,
        })

    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "examples": len(records), "sources": len(sources),
        "unique_source_ids": sorted({r["source_id"] for r in records}),
        "output_manifest": str(index_path), "sample_rate": TARGET_SR,
        "duration_seconds": seconds, "seed": seed,
        "randomize_event_offsets": randomize_event_offsets,
        "note": "Stems are cropped on a shared source clock by default. Split by source_id before training."
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--randomize-event-offsets", action="store_true",
                        help="Apply deliberate timing-jitter augmentation to event stems")
    args = parser.parse_args()
    if args.count < 1 or args.seconds <= 0:
        parser.error("--count and --seconds must be positive")
    generate(args.manifest, args.output_dir, args.count, args.seconds, args.seed,
             randomize_event_offsets=args.randomize_event_offsets)


if __name__ == "__main__":
    main()
