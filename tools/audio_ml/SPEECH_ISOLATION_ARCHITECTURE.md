# Smart Gate neural speech-isolation architecture

## Product objective

The production objective is **clean speech audio**, not merely event labels. The plugin must suppress keyboard and mouse sounds both in pauses and when they overlap speech, while preserving speech onsets, quiet consonants, sibilants, word endings and natural dynamics.

A detector-only model cannot satisfy this objective: it can report that speech and a click coexist, but it cannot remove the click from the waveform. The system therefore needs a supervised speech-enhancement / source-separation model.

## Proposed signal path

1. **Causal streaming frontend**: frame the microphone signal with bounded lookahead and persistent state. No whole-recording normalization, future samples, or clip-level statistics.
2. **Compact speech-isolation network**: consume noisy audio features and estimate the target speech signal (or a complex time-frequency mask). Train against the clean speech stem, not against event labels alone.
3. **Auxiliary event heads**: optionally predict speech, keyboard, mouse and other-event activity. These outputs are useful for diagnostics, confidence and carefully gated behavior, but they are not a substitute for the speech estimate.
4. **Conservative reconstruction / post-processing**: reconstruct speech with overlap-add and smooth only model artifacts. Avoid hard binary masks that produce musical noise or chop consonants.
5. **Existing gate**: remain a separate final stage for residual room noise during true pauses. The gate must not be responsible for removing clicks that overlap active speech.

## Training data and targets

Construct mixtures from synchronized stems:
- clean speech target;
- keyboard noise stem;
- mouse noise stem;
- optional room/background stem.

Create examples for speech alone, speech+keyboard, speech+mouse, speech+both, non-speech keyboard/mouse, and silence/background. Randomize relative levels, event timing, microphone distance, room impulse responses, device examples and background conditions. Keep source recordings grouped: no speaker/session/source recording may leak across train, validation and test splits.

DCASE Office Synthetic can provide useful overlapping event material, and Office Live can provide real-scene evaluation. Before using either, inspect the actual annotation/audio layout and verify which annotated intervals truly contain speech overlapping keyboard or mouse. Generic event overlap is not sufficient. Public data is bootstrap material; final evaluation needs recordings from the target microphone, room, keyboard and mouse.

For every mixture, retain the clean speech reference. The principal target is the clean speech waveform. Event labels are auxiliary multi-label targets and may be inactive simultaneously (silence).

## Streaming and latency constraints

- The current plugin lookahead budget is approximately 30 ms; the model path must fit within the actual measured plugin latency budget.
- Use causal or explicitly bounded-lookahead inference with persistent state across audio blocks.
- Avoid the existing research classifier's full-recording normalization and long patch inference in the real-time callback.
- Measure end-to-end algorithmic latency, block-time CPU, worst-case spikes and memory on representative Windows hardware at 48 kHz, stereo, and 64/128/256-sample buffers.
- Keep inference off the audio callback if it requires a worker; define bounded queues, stale-frame behavior and safe fallback. Never block the audio thread waiting for inference.

A low CPU target (2–3% on the user's meter) is a product requirement to measure, not a guarantee. If a model misses the budget, reduce model size or use a staged/conditional compute design; do not silently disable suppression or compromise speech quality.

## Losses and evaluation

Train the speech estimate with a combination of scale-aware waveform loss and multi-resolution spectral loss. Evaluate:
- speech quality/intelligibility and intrusive distortion against the clean reference;
- residual keyboard/mouse energy during speech and during pauses;
- false suppression of sibilants, plosives, breaths, quiet words, phrase starts and endings;
- separate speech-only, keyboard-only, mouse-only, speech+keyboard, speech+mouse and speech+both test sets;
- unseen speakers, sessions, devices and rooms.

Report per-condition metrics and listening examples. A single aggregate F1 or event-classification accuracy is not proof of successful source separation. Do not claim success from synthetic data alone.

## Safe implementation sequence

1. Keep the current multi-label CNN as an offline research detector only.
2. Inspect and import public dataset annotations; create leakage-safe stem/mixture manifests.
3. Build a reproducible mixture generator that preserves clean speech targets.
4. Implement and train a small causal speech-enhancement model offline.
5. Evaluate on recording-level held-out public data and separately recorded target-device data.
6. Prototype streaming inference outside the production audio path; verify latency, CPU and block-boundary continuity.
7. Integrate behind an opt-in development flag only after speech-preservation and regression tests pass.
8. Keep the existing DSP/Gate fallback available until real-world listening tests pass.

## Non-goals and guardrails

- Do not treat event classification as source separation.
- Do not connect unvalidated model probabilities directly to aggressive attenuation.
- Do not train or validate only on isolated clicks; overlap with speech is the key case.
- Do not normalize using future audio or use non-causal processing in a real-time path.
- Do not merge experimental ML work into the stable branch or ship a model before reproducible tests and listening validation.
