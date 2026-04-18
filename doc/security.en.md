# Security Policy

[简体中文](./security.zh.md) | **English**

## Supported Versions

Only `main` is supported. Run the latest published image or rebuild from `main`.

## Threat Model

This service uploads audio, runs speaker diarization, and stores speaker
voiceprints. It is designed for **trusted deployments**. By default it exposes
the following data to any client that can reach `:8780`:

- Every uploaded audio file under `data/uploads/`
- Every transcript under `data/transcriptions/`
- Every enrolled voiceprint (persistent speaker embedding)

Treat the service as if it were an internal database.

## Required Hardening

1. **Set `API_KEY`**. Without it the service accepts unauthenticated requests
   and logs a startup warning. Any deployment that is not on a fully trusted
   LAN segment MUST set this env var to a long random string. Clients send it
   as `Authorization: Bearer <key>` or `X-API-Key: <key>`.
2. **Never commit `.env`**. Only `.env.example` belongs in git.
3. **Do not expose `:8780` to the public Internet.** Put it behind a VPN, a
   reverse proxy with TLS, or at minimum an IP allow-list. `API_KEY` alone is
   not a substitute for transport encryption.
4. **Keep your HuggingFace token out of logs and out of images.** It is read
   from `HF_TOKEN` at runtime and used only to download pyannote models.
5. **Back up `data/voiceprints/`**. Losing it means you have to re-enroll
   every speaker.

## Reporting a Vulnerability

Please open a private security advisory on GitHub, or email the maintainer
listed in the repo. Do not file public issues for unpatched vulnerabilities.
