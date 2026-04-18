# Install Guide for AI Agents

[简体中文](./ai-install.zh.md) | **English**

> This document is written for **AI agents / LLMs** asked by a user to
> deploy this service on the user's machine. Humans doing manual
> deployment should read [`quickstart.en.md`](./quickstart.en.md).
>
> Companion doc: once deployed, see [`ai-usage.en.md`](./ai-usage.en.md)
> for how to call the API.

## Your scope

The user will ask you to deploy `openplaud-voice-transcribe` on one of
their machines. You can:
- Run shell commands, read and edit files
- Edit `.env`, `docker-compose.yml`
- Run `docker compose`

You must **NOT**:
- Commit or echo `HF_TOKEN` / `API_KEY` into logs, commits, or chat beyond
  the single hand-off moment
- Skip security hardening (launching without `API_KEY` on a port reachable
  from untrusted networks)
- Run destructive ops like `git reset --hard` or `docker system prune -a`
  "to fix things"
- Fabricate an `HF_TOKEN` — the user must supply it

## Decision tree: inspect the environment first

```
Check 1: is there an NVIDIA GPU?
    $ nvidia-smi
    - works → continue
    - command not found → tell the user "this service requires a GPU", stop
    - GPU present but CUDA unavailable → fix driver first

Check 2: is there enough VRAM? (≥ 12 GB recommended)
    $ nvidia-smi --query-gpu=memory.total --format=csv,noheader
    - < 12 GB → warn about potential OOM, but large-v3 needs ~9 GB so
      continuing is acceptable

Check 3: Docker + NVIDIA Container Toolkit present?
    $ docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
    - GPU info printed → OK
    - "could not select device driver ..." → install nvidia-container-toolkit (below)

Check 4: does the user have an HF_TOKEN?
    - yes → proceed
    - no → pause. Walk the user through:
      1. https://huggingface.co/pyannote/speaker-diarization-3.1  → Agree
      2. https://huggingface.co/pyannote/segmentation-3.0  → Agree
      3. https://huggingface.co/settings/tokens  → create a read token
      Wait for the user to paste the token. **Do not** ask them to paste
      it into git or a public channel — if there's exposure risk, ask for
      it in a terminal or private context.
```

## Installing NVIDIA Container Toolkit (if missing)

**Ubuntu / Debian**:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Other distros: see [NVIDIA's official docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Deployment steps

### 1. Pick a working directory and clone

Default to the user's home directory unless they say otherwise:

```bash
cd ~
git clone https://github.com/MapleEve/openplaud-voice-transcribe.git
cd openplaud-voice-transcribe
```

### 2. Generate and fill `.env`

**Important: `API_KEY` must be a strong random string**. Generate it
yourself — don't ask the user to hand-write one:

```bash
cp .env.example .env
API_KEY_VALUE=$(openssl rand -hex 32)
# confirm with the user, or let them provide their own
```

Then set the two critical fields in place (keep other defaults):

```bash
sed -i.bak "s|^HF_TOKEN=.*|HF_TOKEN=${USER_SUPPLIED_HF_TOKEN}|" .env
sed -i.bak "s|^API_KEY=.*|API_KEY=${API_KEY_VALUE}|" .env
rm .env.bak
```

**Right after the edit**, show the user the `API_KEY` value **one time**
so they can paste the same key into OpenPlaud(Maple)'s
"Settings → Transcription". After that moment, never echo this value
back to logs or chat.

**If the user is on a China network**, also add the HF mirror:

```bash
grep -q '^HF_ENDPOINT=' .env || echo 'HF_ENDPOINT=https://hf-mirror.com' >> .env
```

### 3. Launch

```bash
docker compose --env-file .env up -d --build
```

### 4. Wait for model downloads

First boot downloads ~5 GB of weights from HuggingFace. Poll the logs
periodically (every 30 s or so):

```bash
docker logs --tail 20 voice-transcribe
```

Key signals:
- `Uvicorn running on http://0.0.0.0:8780` → service is up
- `401 Client Error` downloading a model → bad `HF_TOKEN`
- `403 Forbidden` → gated-model terms not accepted, go back to check 4
- `np.NaN was removed` → someone edited `requirements.txt` and let numpy
  2.x in — restore the `numpy<2.0` pin
- Still downloading after 10 min → slow network; add `HF_ENDPOINT` mirror

### 5. Health verification

```bash
curl -sf http://localhost:8780/healthz
# expect: {"ok":true}

# auth actually enforced?
source .env
curl -sS http://localhost:8780/api/voiceprints -H "Authorization: Bearer $API_KEY"
# expect: [] (empty on first boot)

# unauthenticated requests are rejected
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:8780/api/voiceprints
# expect: 401
```

All three pass → deployment done.

## Verify the GPU is actually in use

```bash
docker exec voice-transcribe python -c "import torch; print('cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# expect: cuda= True NVIDIA ...
```

If it prints `cuda= False`, check the compose file's GPU reservation and
that `nvidia-ctk runtime configure` ran.

## Hand-off checklist

After deployment, give the user a crisp one-shot summary:

1. ✅ **Service URL**: `http://<host-ip-or-domain>:8780`
2. ✅ **API_KEY**: the value you generated — share it **once**, tell the
   user to store it safely from here on
3. ✅ Ask them to set the following in OpenPlaud(Maple) → Settings →
   Transcription:
   - Private transcription base URL = the URL above
   - Private transcription API key = the same API_KEY
4. ✅ **Strong reminder**: `:8780` must not be exposed to the public
   Internet directly. Use VPN / reverse proxy with TLS / at least an IP
   allow-list. See [`security.en.md`](./security.en.md).
5. ✅ Backup reminder: `data/voiceprints/` should be backed up regularly

## Upgrade flow

When the user asks you to upgrade:

```bash
cd ~/openplaud-voice-transcribe   # or actual path
git fetch origin
git diff --stat main origin/main   # show the user what will change
git pull
docker compose --env-file .env up -d --build
docker logs --tail 40 voice-transcribe
curl -sf http://localhost:8780/healthz
```

If `git pull` would overwrite uncommitted local changes, **stop and ask
the user**. Never `git reset --hard`.

## Don't do this

- ❌ Echo `HF_TOKEN` or `API_KEY` anywhere beyond the single hand-off
  moment (logs, commits, PR descriptions)
- ❌ `git add .env` (it is gitignored, don't force-add it)
- ❌ Remove pins in `requirements.txt` to "make it start"
- ❌ Delete `./models/` to save disk — that's the weight cache, redownload
  costs 5 GB
- ❌ Run `docker rm -f voice-transcribe` and then expect manually-pip-installed
  packages to survive — `docker compose up --build` rebuilds from
  `requirements.txt`
- ❌ Open a 443 port / reverse proxy / public DNS record without the
  user's explicit go-ahead

## Common follow-ups

- **"OpenPlaud(Maple) can't reach the service"**
  → Check: 1) `API_KEY` matches on both sides exactly (no whitespace/newlines);
  2) host firewall allows `:8780`; 3) OpenPlaud(Maple)'s host can `curl`
  the service URL.
- **"Add HTTPS?"**
  → Recommend a reverse proxy (nginx/caddy/traefik) for TLS termination
  and cert rotation. Don't patch FastAPI to serve certs itself — too much
  maintenance drag.
- **"GPU 0 is taken by something else, can we use GPU 1?"**
  → Edit `CUDA_VISIBLE_DEVICES=1` in `.env`, then restart compose.

## Related docs

- Human-oriented quickstart → [`quickstart.en.md`](./quickstart.en.md)
- API usage for AI agents → [`ai-usage.en.md`](./ai-usage.en.md)
- API contract → [`api.en.md`](./api.en.md)
- Security policy → [`security.en.md`](./security.en.md)
