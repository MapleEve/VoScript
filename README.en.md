# openplaud-voice-transcribe

[简体中文](./README.md) | **English**

Self-hosted GPU transcription service with persistent speaker voiceprints.
Designed as the private backend for [**OpenPlaud(Maple)**](https://github.com/MapleEve/openplaud),
but usable as a stand-alone FastAPI service.

```
Audio  ──►  faster-whisper large-v3  (transcription)
        ──►  pyannote 3.1             (speaker diarization)
        ──►  ECAPA-TDNN               (speaker embeddings)
        ──►  VoiceprintDB             (cosine match vs. enrolled speakers)
        ──►  timestamped text with identified speaker names
```

## Documentation

All detailed docs live in [`doc/`](./doc/). Chinese is the default, every
page has an English counterpart:

| Topic | 中文 | English |
| --- | --- | --- |
| Quickstart | [quickstart.zh.md](./doc/quickstart.zh.md) | [quickstart.en.md](./doc/quickstart.en.md) |
| API reference | [api.zh.md](./doc/api.zh.md) | [api.en.md](./doc/api.en.md) |
| **Install guide for AI agents** | [ai-install.zh.md](./doc/ai-install.zh.md) | [ai-install.en.md](./doc/ai-install.en.md) |
| **Usage guide for AI agents** | [ai-usage.zh.md](./doc/ai-usage.zh.md) | [ai-usage.en.md](./doc/ai-usage.en.md) |
| Security policy | [security.zh.md](./doc/security.zh.md) | [security.en.md](./doc/security.en.md) |
| Changelog | [changelog.zh.md](./doc/changelog.zh.md) | [changelog.en.md](./doc/changelog.en.md) |

First-time deployers: start with the [Quickstart](./doc/quickstart.en.md).
AI agents integrating the API: read the [AI usage guide](./doc/ai-usage.en.md).
AI agents deploying the service for a user: read the
[AI install guide](./doc/ai-install.en.md).

## Why a separate repo

OpenPlaud(Maple) is a single-user control panel. The heavy work — loading
whisper/pyannote, keeping models resident in GPU memory, running diarization,
maintaining a voiceprint database — stays behind a private HTTP API so that
the public panel never ships a GPU model or a raw embedding to the browser.

This repo is that private API. OpenPlaud(Maple) submits audio to it, polls
for the job, stores the transcript locally, and calls the voiceprint
endpoints when the user enrolls a speaker.

## Features

- Async job pipeline (`queued → converting → transcribing → identifying → completed`)
- Chinese + multilingual transcription (faster-whisper large-v3)
- Speaker diarization (pyannote 3.1)
- Persistent voiceprints: enroll once, auto-match in future recordings
  (cosine similarity ≥ 0.75)
- Stable HTTP contract consumed by OpenPlaud(Maple)'s
  [`voice-transcribe-provider.ts`](https://github.com/MapleEve/openplaud/blob/main/src/lib/transcription/providers/voice-transcribe-provider.ts)
  and [`voice-transcribe/client.ts`](https://github.com/MapleEve/openplaud/blob/main/src/lib/voice-transcribe/client.ts)
- Optional Bearer / `X-API-Key` auth on every `/api/*` route (constant-time compare)
- Container runs as a **non-root user**; uploads enforced by `MAX_UPLOAD_BYTES`; voiceprint DB is concurrency-safe with atomic writes. Full hardening list in [`doc/security.en.md`](./doc/security.en.md)
- Minimal built-in web UI at `/` for manual testing

## 30-second start

```bash
git clone https://github.com/MapleEve/openplaud-voice-transcribe.git
cd openplaud-voice-transcribe

cp .env.example .env
# edit .env — at minimum set HF_TOKEN and API_KEY

docker compose up -d --build
curl -sf http://localhost:8780/healthz
```

Full steps + troubleshooting in [`doc/quickstart.en.md`](./doc/quickstart.en.md).

## Wiring into OpenPlaud(Maple)

In OpenPlaud(Maple) → Settings → Transcription, set:

- **Private transcription base URL**: `http://<host>:8780`
- **Private transcription API key**: the same `API_KEY` from `.env`

The OpenPlaud(Maple) worker then routes every recording through this service.
API details in [`doc/api.en.md`](./doc/api.en.md).

## License

MIT — see [LICENSE](./LICENSE).
