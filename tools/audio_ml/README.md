<!-- CI-enabled development branch: validate the existing plugin and DSP baseline. -->

# Smart Gate acoustic-event ML training

Offline training pipeline for a future speech / keyboard / mouse / silence / overlap classifier.
This branch deliberately does not connect an unvalidated model to the audio callback.

## Required labels
`silence`, `speech`, `keyboard`, `mouse`, `mixed`.
Use `mixed` only for accurately bounded intervals where speech and a mechanical event
actually overlap. Isolated clicks do not teach the model to remove clicks during speech.

## Dataset manifest
```json
{
  "recordings": [{
    "id": "session_001",
    "path": "audio/session_001.wav",
    "segments": [
      {"start": 0.0, "end": 3.0, "label": "silence"},
      {"start": 3.0, "end": 7.0, "label": "keyboard"},
      {"start": 7.0, "end": 12.0, "label": "speech"},
      {"start": 12.0, "end": 15.0, "label": "mixed"}
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
training and validation. It requires at least three independent recordings per class,
including genuine `mixed` recordings, and reports a confusion matrix, per-class recall
and macro-F1. The model is only exported after that leakage-safe split is possible.

## First user recording
The 61.176 s recording appears to contain quiet background (0–4 s), keyboard-like activity
(4–10 s), background (10–13 s), mouse-like high-frequency activity (13–22 s), and speech
(about 23–46 s and 47–60 s). These boundaries are provisional and should be reviewed.
It does not provide a confirmed speech+keyboard/mouse overlap segment. Therefore it is
useful as isolated-class material, but cannot satisfy the mixed-class requirement.

Keep private recordings out of Git. Store them locally and share only reviewed labels and
aggregate validation metrics.
