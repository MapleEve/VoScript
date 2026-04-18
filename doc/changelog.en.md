# Changelog

[简体中文](./changelog.zh.md) | **English**

## 0.1.0 — initial public release

- First public release of the private transcription backend used by
  [OpenPlaud](https://github.com/MapleEve/openplaud).
- Async job pipeline: `queued → converting → transcribing → identifying → completed`.
- faster-whisper `large-v3` + pyannote `3.1` + ECAPA-TDNN speaker embeddings.
- Persistent voiceprint DB with cosine-similarity auto-match.
- Optional `API_KEY` bearer auth on all `/api/*` routes.
- Portable `docker-compose.yml` (data/model paths configurable via env).
- Dependency pins to keep `pyannote.audio==3.1.1` usable:
  - `numpy<2` (pyannote 3.1.1 uses `np.NaN`, removed in numpy 2.x).
  - `huggingface_hub<0.24` (keeps the `use_auth_token` kwarg pyannote calls).
