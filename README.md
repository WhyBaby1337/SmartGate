# Smart Gate

A native VST3 gate/expander focused on preserving a learned target source while reducing unwanted background sound.

## Status

Clean project bootstrap. First milestone: reproducible Windows VST3 build and deterministic DSP tests. Target-aware analysis will be introduced behind explicit, tested interfaces. Silero VAD, if integrated, is speech-activity evidence, not speaker identity.

## Architecture

1. Deterministic DSP core.
2. Target learning and acoustic profile.
3. Asynchronous analysis worker; never block the audio callback.
4. Decision engine combining target similarity, speech activity, and noise evidence.
5. Gate engine with attack, hold, release, lookahead, stereo linking, and host latency reporting.
6. Plugin UI with Learn/Reset, confidence, gain reduction, and clear AI-off states.

## Engineering rules

- No heap allocation, locks, or blocking calls in the audio callback.
- Every DSP behavior must have deterministic tests.
- AI failure must degrade safely to the non-AI path.
- Never describe heuristic or VAD output as target-source recognition.
- Document third-party code and model licenses before distribution.
