# Changelog

[简体中文](./changelog.zh.md) | **English**

## 0.3.0 — WhisperX alignment + sqlite voiceprint DB + WeSpeaker (2026-04-18)

Three independent core upgrades released together.

### WhisperX forced alignment
- `app/pipeline.py`'s `transcribe` + `align_segments` now use `whisperx`'s
  forced-alignment pipeline.
- Each segment in the result carries an optional
  `words: [{word, start, end, score}, …]` field — **word-level timestamps**.
- Still CTranslate2-based internally, so the existing local
  `/models/faster-whisper-<size>` cache continues to satisfy cold starts
  without round-tripping HuggingFace.
- First boot downloads a wav2vec2 alignment model (cached under `/cache`).
  Alignment for Chinese audio can fail on some utterances — when that
  happens we gracefully fall back to segment-level timestamps (no
  `words[]`) instead of failing the whole job.
- Version pin: `whisperx==3.1.6`. It is the only WhisperX series
  compatible with our frozen `torch==2.4.1` + `pyannote==3.1.1`; later
  releases require newer torch and newer pyannote. 3.1.x is marked
  "yanked" on PyPI but pip installs yanked packages when the version is
  pinned explicitly in requirements.

### Voiceprint store moved to sqlite + sqlite-vec
- `app/voiceprint_db.py` no longer uses `index.json + *.npy` files.
- Single `voiceprints.db` (sqlite 3) + sqlite-vec's `vec0` virtual
  table for vector similarity. Top-k nearest-neighbour search runs in
  `O(log N)` through the extension.
- All writes run inside sqlite transactions with WAL enabled; the old
  hand-rolled `tempfile + os.replace` atomic writer is gone.
- Concurrency: Python-level `threading.RLock` + sqlite WAL.
- **Automatic migration** on first boot: if `index.json` + `.npy` files
  exist, they're imported into sqlite in a single transaction and
  `index.json` is renamed to `index.json.migrated.bak`. The `.npy`
  files are left in place for rollback.
- If sqlite-vec fails to load (build without the extension), falls back
  to a Python-side cosine full scan.

### WeSpeaker ResNet34 replaces ECAPA-TDNN
- `pyannote/wespeaker-voxceleb-resnet34-LM` replaces speechbrain's ECAPA.
- Dependency diet: `speechbrain` is no longer required. WeSpeaker's
  wrapper class ships with `pyannote.audio`.
- Embedding dimension goes from 192 (ECAPA) to ~256 (WeSpeaker).
- **Breaking**: the new embedding space is not cosine-comparable with
  ECAPA's. **Every voiceprint enrolled under 0.2.x has to be
  re-enrolled** — comparing old ECAPA vectors against new WeSpeaker
  queries yields essentially random cosine similarities.
- **HF gated model**: accept the agreement at
  <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM>
  before first use — same process as for speaker-diarization-3.1 and
  segmentation-3.0.
- The per-speaker chunk selection (≥1 s, top-10 longest, mean pooled)
  is unchanged.

### Upgrade-breaking changes
- Bigger container image: WhisperX pulls wav2vec2 deps + the alignment
  model weights (~1 GB extra cache on first run).
- **Old voiceprints must be re-enrolled.** Either delete
  `data/voiceprints/voiceprints.db` or `DELETE /api/voiceprints/{id}`
  one by one, then re-enroll.
- One more HuggingFace gated model to agree to (WeSpeaker).

### Stays compatible
- HTTP contract unchanged (`segments[i].words` is a new optional field
  — old clients ignore it).
- No new env vars.
- Container name `voscript`, port, data-dir layout, non-root user, and
  API_KEY enforcement all unchanged.

## 0.2.1 — Renamed to voscript (2026-04-18)

Decoupled from OpenPlaud(Maple) — the service stands on its own, so it now
has its own name:

- **Repo**: `MapleEve/openplaud-voice-transcribe` → `MapleEve/voscript`
  (GitHub keeps a permanent 301 from the old slug, existing clones keep
  working).
- **Docker service / container name**: `voice-transcribe` → `voscript`
  (`docker logs voscript`, `docker exec voscript …`).
- **Image name**: compose produces `voscript-voscript:latest` automatically.
- **README / docs**: repositioned as an independent transcription service,
  with OpenPlaud(Maple) called out as one known consumer rather than the
  service's identity.
- **HTTP contract, file layout, env vars, on-disk data layout: unchanged.**
  Existing clients keep working with zero code changes.

## 0.2.0 — Post red-team hardening (2026-04-18)

Full hardening pass following real-audio end-to-end testing and an
independent penetration test.

### Security hardening
- **Container no longer runs as root.** A non-root `app` user (uid/gid
  1000 by default, overridable via `APP_UID`/`APP_GID`). An RCE inside
  the container only owns the service's own uid — it can't read other
  root-owned files on the host.
- **Upload size limit.** `/api/transcribe` now reads the upload in 1 MiB
  chunks and aborts with `HTTP 413` when the total exceeds
  `MAX_UPLOAD_BYTES` (default 2 GiB, configurable). The partial file is
  removed so disk doesn't leak.
- **Upload filename sanitization.** `PurePosixPath(filename).name`
  strips any client-supplied directory components — an attacker-set
  `filename=../../etc/passwd.wav` becomes `passwd.wav` before the path
  is built.
- **`ffmpeg` argv hardening.** `--` is inserted before the input path
  so a filename starting with `-` can't be interpreted as a flag.
- **Constant-time key comparison.** `hmac.compare_digest` replaces
  plain `!=` for the bearer/key check. Eliminates any theoretical
  timing side channel.
- **Tighter public-path matching.** `/docs`, `/redoc`, `/openapi.json`
  moved from `startswith()` prefixes to exact-match only. `/docsXYZ`
  used to slip past auth; now it correctly returns 401.
- **Concurrency-safe, atomic `VoiceprintDB`.** Every mutation holds
  `threading.Lock`; `index.json` and every `.npy` write goes through
  `tempfile` + `os.replace` so a crash mid-write can't corrupt the
  index.
- **`np.load(..., allow_pickle=False)` everywhere.** Closes the
  pickle-deserialization RCE path.

### Features & config
- New env var `MAX_UPLOAD_BYTES` (default `2147483648`, i.e. 2 GiB)
  configurable in `.env` / compose.
- New env vars `APP_UID` / `APP_GID` (default 1000) for hosts where
  the bind-mount owner isn't uid 1000.
- HuggingFace cache migrated from the container's `/root/.cache/huggingface`
  to `/cache`. `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`, and
  `XDG_CACHE_HOME` all point at `/cache` so the non-root user can
  write it.
- `docker-compose.yml` mounts `${MODEL_CACHE_DIR}` both at `/cache`
  (read-write, HF hub cache) and `/models:ro` (read-only, for
  `pipeline.py`'s local-first checkpoint resolver), so cold starts
  don't round-trip through HuggingFace.

### Bug fixes
- `_convert_to_wav` now shells out to `ffmpeg` directly instead of
  going through `pydub.AudioSegment`. Fixes `KeyError('codec_type')`
  that pydub hits on newer ffmpeg's Opus/container output
  ([jiaaro/pydub#638](https://github.com/jiaaro/pydub/issues/638)).
- `GET /` now reaches browsers when `API_KEY` is set. Previously the
  bundled Web UI was effectively unreachable because browsers can't
  attach a Bearer header to a direct navigation; the UI's own
  fetch-to-`/api/*` calls still carry the key.
- On the OpenPlaud(Maple) side, `VoiceTranscribeProvider` now keeps
  the raw `speaker_label` (`SPEAKER_XX`) in `speakerSegments[].speaker`,
  so re-enrollment still works after an automatic voiceprint match.

### Breaking
- The container's HF cache path moved from `/root/.cache/huggingface`
  to `/cache`. If you had a host directory bind-mounted at the old
  path, update your compose (or just use this repo's new
  `docker-compose.yml`, which handles both mounts for you).
- Requests beyond `MAX_UPLOAD_BYTES` now fail with `413` instead of
  silently succeeding. Default 2 GiB fits every real-world audio we
  tested.

## 0.1.0 — Initial public release

- First public release of the private transcription backend used by
  [OpenPlaud(Maple)](https://github.com/MapleEve/openplaud).
- Async job pipeline: `queued → converting → transcribing → identifying → completed`.
- faster-whisper `large-v3` + pyannote `3.1` + ECAPA-TDNN speaker embeddings.
- Persistent voiceprint DB with cosine-similarity auto-match.
- Optional `API_KEY` bearer auth on all `/api/*` routes.
- Portable `docker-compose.yml` (data/model paths configurable via env).
- Dependency pins to keep `pyannote.audio==3.1.1` usable:
  - `numpy<2` (pyannote 3.1.1 uses `np.NaN`, removed in numpy 2.x).
  - `huggingface_hub<0.24` (keeps the `use_auth_token` kwarg pyannote calls).
