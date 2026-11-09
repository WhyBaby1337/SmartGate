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

The prototype uses 16 kHz log-mel patches (64 bands, 32 frames). With a 512-sample FFT and 160-sample hop, the actual receptive field is about 342 ms, not 320 ms.
Validation is grouped by whole recording to prevent adjacent windows leaking between
training and validation. It requires at least three independent recordings with positive
examples per output label, reports per-label precision/recall/F1, and exports ONNX logits.

## Correct product architecture

The event CNN is only an auxiliary detector. The actual product goal requires a separate supervised speech-isolation model that estimates clean speech from mixtures, trained against synchronized clean-speech targets. See [SPEECH_ISOLATION_ARCHITECTURE.md](SPEECH_ISOLATION_ARCHITECTURE.md) for the signal path, dataset strategy, latency constraints and acceptance tests. Event classification alone cannot remove a click that overlaps speech.


## Paired speech-mixture generation

The speech-isolation model must learn from paired files: the input mixture and the clean
speech target must have identical sample timing and duration. Use
`build_speech_mixtures.py` to create deterministic training examples from synchronized
source-session stems.

Input manifest example (paths are relative to the manifest):
```json
{
  "sources": [
    {
      "id": "session_a",
      "speech": "stems/session_a_speech.wav",
      "keyboard": "stems/session_a_keyboard.wav",
      "mouse": "stems/session_a_mouse.wav",
      "background": "stems/session_a_room.wav"
    },
    {
      "id": "session_b",
      "speech": "stems/session_b_speech.wav",
      "keyboard": "stems/session_b_keyboard.wav",
      "mouse": "stems/session_b_mouse.wav"
    }
  ]
}
```

Run:
```bash
python build_speech_mixtures.py --manifest stems.json --output-dir generated --count 5000 --seconds 2 --seed 1337
```

Each JSONL row points to a stereo 48 kHz mixture and its synchronized clean-speech target.
Keyboard and mouse stems are independently included with randomized relative levels;
some examples contain speech alone, non-speech events, or background. By default, every stem
is cropped at the same sample offset on the shared source clock, preserving real event/speech
overlap. The optional `--randomize-event-offsets` flag deliberately adds timing jitter and
should be used only as an explicit augmentation, not as the default data alignment. The
generator uses one source-session bundle per example and writes its `source_id` into the index.
All stems default to a shared time origin. For event-only excerpts that begin later in the
session, provide explicit per-stem offsets in seconds, for example:

```json
{
  "id": "session_c",
  "speech": "stems/session_c_speech.wav",
  "mouse": "stems/session_c_mouse_excerpt.wav",
  "start_seconds": {"speech": 0.0, "mouse": 12.4}
}
```

Offsets are measured on the common source-session clock after resampling; they must be finite
and non-negative. The generator computes the session extent from each stem's offset plus
duration, then crops every stem against the same absolute timeline. Do not use offsets to
disguise stems from unrelated recordings. Split by source/session IDs before training: never
randomly split generated clips, or near-identical source material can leak across evaluation
sets.

This generator is a first data-preparation component, not a trained separator. Its quality
depends on genuinely isolated, correctly synchronized stems. Do not treat arbitrary
unrelated recordings as aligned stems. For final evaluation, hold out complete sessions,
speakers, rooms and devices, and include real simultaneous speech+keyboard/mouse recordings.

## Not ready for live audio yet

This is a research/training scaffold, not a production inference path. The current dataset
preprocessing normalizes each complete recording, which is non-causal and cannot be copied
directly into a live plugin. The approximately 342 ms patch also exceeds Smart Gate's current 30 ms lookahead.
Before integration we must implement and test a causal streaming feature frontend, decide on
a low-latency temporal model/state strategy, and measure its CPU cost. Do not ship this model
or connect its scores to attenuation until those pieces and the speech-preservation tests pass.

## First user recording

The 61.176 s recording appears to contain quiet background (0–4 s), keyboard-like activity
(4–10 s), background (10–13 s), mouse-like high-frequency activity (13–22 s), and speech
(about 23–46 s and 47–60 s). These boundaries are provisional and should be reviewed.
It does not provide a confirmed speech+keyboard/mouse overlap segment. Therefore it is
useful as isolated-class material, but cannot validate the most important overlapping case.

Keep private recordings out of Git. Store them locally and share only reviewed labels and
aggregate validation metrics.

## First speech-separation model prototype

`speech_separator.py` now contains a small unidirectional GRU mask estimator. It consumes
20 ms STFT frames at 48 kHz with a 10 ms hop and estimates a per-bin magnitude mask. Its
recurrent state is explicit, so the temporal network itself can be advanced frame by frame.
The companion `train_speech_separator.py` trains against synchronized clean-speech targets
using waveform L1 plus multi-resolution spectral loss. It splits by complete `source_id`
groups and refuses fewer than three independent sessions.

Example:
```bash
python train_speech_separator.py --manifest generated/manifest.jsonl --output build/speech_separator.pt
```

This is the first trainable separation prototype, not a production-ready streaming processor.
The current wrapper is an offline waveform path; streaming overlap-add, block-boundary state,
bounded worker scheduling, ONNX/runtime export, and measured latency/CPU are still missing.
A 20 ms analysis window is only a latency design point, not proof that the full plugin path
meets the 30 ms budget. The model is mono for this first experiment. It must pass speech
preservation and real overlap listening tests before any plugin integration.
