# Quickstart

[简体中文](./quickstart.zh.md) | **English**

This guide is for first-time deployers. Expect 15–30 minutes, most of it
waiting for model weights to download.

## 0. Prerequisites

- A Linux host with an NVIDIA GPU (≥ 12 GB VRAM recommended; RTX 3090/4090/A10
  or better is a safe bet).
- Docker 24+.
- **NVIDIA Container Toolkit** (without it, `docker run --gpus all` fails):
  ```bash
  # Ubuntu example
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- A HuggingFace account, and:
  1. Click **Agree and access repository** at
     <https://huggingface.co/pyannote/speaker-diarization-3.1>.
  2. Do the same at <https://huggingface.co/pyannote/segmentation-3.0>.
  3. Create a **read** token at <https://huggingface.co/settings/tokens>
     (starts with `hf_`).

> Both models are gated. If you skip this, the service will hang on first
> boot trying to download them.

## 1. Clone the repo

```bash
git clone https://github.com/MapleEve/openplaud-voice-transcribe.git
cd openplaud-voice-transcribe
```

## 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`. At minimum fill in:

```env
HF_TOKEN=hf_your_token
API_KEY=a_long_random_string_e.g._openssl_rand_hex_32
```

If you are on a China network, also add:

```env
HF_ENDPOINT=https://hf-mirror.com
```

> Generate a strong API key: `openssl rand -hex 32`

Every other env var has a sane default — see [`.env.example`](../.env.example)
for the full list.

## 3. Start the service

```bash
docker compose --env-file .env up -d --build
```

The first boot downloads ~5 GB of model weights into `./models/`. Watch
progress with:

```bash
docker logs -f voice-transcribe
```

You are good when you see `Uvicorn running on http://0.0.0.0:8780`.

Or run the bundled helper:

```bash
./scripts/deploy.sh
```

It checks `.env`, starts the container, and waits for `/healthz`.

## 4. Verify the deployment

```bash
# Health check (always unauthenticated)
curl -sf http://localhost:8780/healthz
# → {"ok":true}

# Any /api/* call needs API_KEY
curl -sS http://localhost:8780/api/voiceprints \
    -H "Authorization: Bearer $API_KEY"
# → [] (empty on first boot)
```

Open <http://localhost:8780/> in a browser for a minimal web UI you can
upload audio to.

## 5. Wire it into OpenPlaud(Maple)

In OpenPlaud(Maple) → Settings → Transcription, set:

- **Private transcription base URL**: `http://<host>:8780`
- **Private transcription API key**: the **exact** `API_KEY` from `.env`

Once saved, the OpenPlaud(Maple) worker will route every recording through
this service. See [`api.en.md`](./api.en.md) for the full contract.

## Upgrades

```bash
cd openplaud-voice-transcribe
git pull
docker compose --env-file .env up -d --build
```

Model weights in `./models/` are cached, rebuild won't redownload them.

## Troubleshooting

### `nvidia-smi` not found inside container
→ NVIDIA Container Toolkit missing or Docker wasn't restarted. Redo step 0.

### `403 Forbidden` downloading pyannote models
→ You didn't accept the gated-model terms, or `HF_TOKEN` is wrong.

### Crashes with `np.NaN was removed`
→ Your `requirements.txt` has been edited and numpy upgraded to 2.x. Keep
the `numpy<2.0` pin.

### Service is up but OpenPlaud(Maple) can't reach it
→ Check that `API_KEY` matches **exactly** on both sides (case/whitespace),
and that OpenPlaud(Maple)'s host can actually reach `:8780` (firewall,
docker networks).

### What do I back up?
→ Just `data/voiceprints/`. Everything else can be re-derived from the
original audio.

See [`security.en.md`](./security.en.md) for deployment-risk details.
