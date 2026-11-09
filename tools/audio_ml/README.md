<!-- CI-enabled development branch: validate the existing plugin and DSP baseline. -->

# Smart Gate acoustic-event ML training

Offline training pipeline for a future multi-label speech / keyboard / mouse event detector.
It deliberately does not connect an unvalidated model to the audio callback.

## Multi-label outputs

The network has independent sigmoid outputs for `speech`, `keyboard`, `mouse`, and
`other`. This is intentional: a single softmax class called "mixed" would not tell the DSP
whether the overlapping event was a keyboard or mouse click. Silence is represented by all
four labels being inactive.

## Public bootstrap datasets

- **DCASE 2013 Office Live**: real office recordings with speech, keyboard clicks and mouse
  clicks; development set includes annotated event sequences. The DCASE dataset catalogue
  lists CC BY 4.0 for the dataset. This is the most useful real-scene starting point.
  Download: https://dcase.community/challenge2013/download
- **DCASE 2013 Office Synthetic (OS)**: event sequences with overlapping events, including
  keyboard, mouse and speech. The catalogue lists CC BY 4.0. Use annotated windows where
  speech overlaps keyboard or mouse; unrelated overlaps are not target positives.

These datasets are useful for pretraining and regression tests, not a substitute for
recordings from the target microphone, room, keyboard and mouse. Keep attribution and
dataset README files with any derived training release.

## Dataset manifest

Segments are mutually exclusive time ranges, but each range can have multiple simultaneous
labels. Example:

```json
{
  "recordings": [{
    "id": "session_001",
    "path": "audio/session_001.wav",
    "segments": [
      {"start": 0.0, "end": 3.0, "labels": []},
      {"start": 3.0, "end": 7.0, "labels": ["keyboard"]},
      {"start": 7.0, "end": 12.0, "labels": ["speech"]},
      {"start": 12.0, "end": 15.0, "labels": ["speech", "mouse"]}
    ]
  }]
}
```

Install `requirements.txt`, then run:
```bash
python train_event_classifier.py --manifest dataset/manifest.json --output build/event_classifier.onnx
```

The script uses 16 kHz log-mel patches (64 bands, 320 ms context) and a compact CNN.
Validation is grouped by whole recording to prevent adjacent windows leaking between
training and validation. It requires at least three independent recordings with positive
examples per output label, reports per-label precision/recall/F1, and exports ONNX logits.
The model is only exported after a leakage-safe recording-level split is possible.

## First user recording

The 61.176 s recording appears to contain quiet background (0–4 s), keyboard-like activity
(4–10 s), background (10–13 s), mouse-like high-frequency activity (13–22 s), and speech
(about 23–46 s and 47–60 s). These boundaries are provisional and should be reviewed.
It does not provide a confirmed speech+keyboard/mouse overlap segment. Therefore it is
useful as isolated-class material, but cannot validate the most important overlapping case.

Keep private recordings out of Git. Store them locally and share only reviewed labels and
aggregate validation metrics.
