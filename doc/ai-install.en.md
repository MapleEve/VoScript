# Install Guide for AI Agents

[简体中文](./ai-install.zh.md) | **English**

> This document is written for **AI agents / LLMs** asked by a user to
> deploy this service on the user's machine. Humans doing manual
> deployment should read [`quickstart.en.md`](./quickstart.en.md).
>
> Companion doc: once deployed, see [`ai-usage.en.md`](./ai-usage.en.md)
> for how to call the API.

## Your scope

The user will ask you to deploy `voscript` on one of
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
Check 0: what OS is the user on?
    $ uname -s
    - Linux              → continue with check 1 (GPU branch)
    - Darwin             → macOS branch (see "macOS path" below),
                           skip the docker-gpu checks
    - MINGW*/MSYS*/CYGWIN* → user is likely on Git Bash; the real target
      is almost certainly WSL2. Ask:
        * "Do you want to deploy inside WSL2?" → have them enter WSL2
          and run every subsequent command there.
        * "You want to run on native Windows?" → not supported, push
          toward WSL2.
```

### Linux / WSL2 branch

```
Check 1: NVIDIA GPU present?
    $ nvidia-smi
    - works → continue
    - command not found / CUDA unavailable → tell the user "this path needs
      an NVIDIA GPU + driver"; use a compatible host or fix the driver.
      Do not substitute an unqualified native CPU installation.

Check 2: enough VRAM?
    $ nvidia-smi --query-gpu=memory.total --format=csv,noheader
    - ≥ 12 GB → default large-v3 is fine
    - 8–12 GB → large-v3 still fits (~9 GB in practice); warn about GPU
      contention with other heavy jobs
    - < 8 GB → set WHISPER_MODEL=medium in .env (3–4× faster, acceptable
      quality tradeoff)

Check 3: Docker + NVIDIA Container Toolkit working?
    $ docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
    - GPU info printed → OK
    - "could not select device driver ..." → install nvidia-container-toolkit (below)

Check 3.5: is the user's uid/gid 1000?
    $ id -u ; id -g
    - both 1000 → proceed, container's default APP_UID=1000 is fine
    - something else → either add APP_UID=$(id -u) / APP_GID=$(id -g)
      to .env, or after deploy run sudo chown -R 1000:1000 on the host
      data directory

Check 4: HF_TOKEN and gated-model access both in place?
    HF tokens and gated-model acceptance are **independent** — order
    doesn't matter but both are required.
    - User has a token AND has accepted both pyannote repos' terms → proceed
    - Only one of them → fill in the missing half:
        * missing token: https://huggingface.co/settings/tokens (read scope)
        * missing acceptance:
            - https://huggingface.co/pyannote/speaker-diarization-3.1 → Agree
            - https://huggingface.co/pyannote/segmentation-3.0 → Agree
    - Neither → do both, any order.
    **Do not** have the user paste the token into git, a commit message,
    or a public channel — prefer private / terminal paste.
```

### macOS path (Docker Desktop can't pass through the GPU)

Docker Desktop on macOS **cannot forward a GPU into a container** (neither
CUDA nor Metal). The supported 0.8.5 deployment is the CI-built Linux CUDA
image with Python 3.11, official cu126 wheels, the matching Rust extension,
and authenticated compatibility adaptations. Prefer a compatible Linux / WSL2
GPU service host with the Mac as client. Native CPU installs require a separate
manual port and are not covered by the image compatibility gate; do not deploy
one automatically or promise the earlier native performance estimates.

```
Check A: Python 3.11 available?
    $ python3.11 --version
    - yes → continue
    - no  → brew install python@3.11

Check B: ffmpeg / libsndfile installed?
    $ which ffmpeg && which sndfile-info || brew install ffmpeg libsndfile

Check C: disk + memory
    - ≥ 10 GB free disk (model weights + caches)
    - ≥ 16 GB RAM (Apple Silicon unified memory counts)

Check D: HF acceptance (same as Linux branch)
```

For a separately qualified native CPU port, use:
- `DEVICE=cpu` (pyannote 3.1's MPS coverage is incomplete, don't use it)
- `WHISPER_MODEL=medium` (large-v3 on CPU isn't "slow", it's "unusable")

Explain that these prerequisites alone do not qualify a native deployment.
See the manual-port boundary below before changing the user's environment.

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

## Deployment steps (Linux / WSL2)

### 1. Pick a working directory and clone

Default to the user's home directory unless they say otherwise:

```bash
cd ~
git clone https://github.com/MapleEve/voscript.git
cd voscript
```

### 2. Generate and fill `.env`

**Important: `API_KEY` must be a strong random string**. Generate it
yourself — don't ask the user to hand-write one:

```bash
cp .env.example .env
API_KEY_VALUE=$(openssl rand -hex 32)
# confirm with the user, or let them provide their own
```

Then rewrite `.env` via Python — **not** `sed + string interpolation`.
Characters like `&`, `/`, or newlines in a token break sed, and if either
variable is unset, sed will silently blank the field:

```bash
# bail out early if either variable is missing
: "${USER_SUPPLIED_HF_TOKEN:?need HF_TOKEN from user}"
: "${API_KEY_VALUE:?API_KEY_VALUE is empty}"

# pass values via env; Python doesn't do shell escaping
HF_TOKEN="$USER_SUPPLIED_HF_TOKEN" API_KEY="$API_KEY_VALUE" python3 - <<'PY'
import os, pathlib
env = pathlib.Path(".env")
lines = []
for line in env.read_text().splitlines():
    if line.startswith("HF_TOKEN="):
        line = f"HF_TOKEN={os.environ['HF_TOKEN']}"
    elif line.startswith("API_KEY="):
        line = f"API_KEY={os.environ['API_KEY']}"
    lines.append(line)
env.write_text("\n".join(lines) + "\n")
PY
```

**Right after the edit**, show the user the `API_KEY` value **one time**
so they can paste the same key into BetterAINote's
"Settings → Transcription". After that moment, never echo this value
back to logs or chat.

**If the target has < 12 GB VRAM, or is a CPU / macOS deployment**,
drop the model to medium:

```bash
sed -i.bak "s|^WHISPER_MODEL=.*|WHISPER_MODEL=medium|" .env && rm .env.bak
```

**If the user is on a China network**, also add the HF mirror:

```bash
grep -q '^HF_ENDPOINT=' .env || echo 'HF_ENDPOINT=https://hf-mirror.com' >> .env
```

**If the user's uid is not 1000** (see decision-tree check 3.5), either
pin the container to their uid:

```bash
UID_VAL=$(id -u); GID_VAL=$(id -g)
[ "$UID_VAL" = "1000" ] || {
    grep -q '^APP_UID=' .env && sed -i.bak "s|^APP_UID=.*|APP_UID=$UID_VAL|" .env || echo "APP_UID=$UID_VAL" >> .env
    grep -q '^APP_GID=' .env && sed -i.bak "s|^APP_GID=.*|APP_GID=$GID_VAL|" .env || echo "APP_GID=$GID_VAL" >> .env
    rm -f .env.bak
}
```

…or chown the host data directories to 1000:

```bash
sudo chown -R 1000:1000 ./data ./models 2>/dev/null || true
```

**Low-disk hosts** — lower the upload cap (default 2 GiB), e.g. to
500 MiB:

```bash
grep -q '^MAX_UPLOAD_BYTES=' .env \
  && sed -i.bak "s|^MAX_UPLOAD_BYTES=.*|MAX_UPLOAD_BYTES=524288000|" .env \
  || echo 'MAX_UPLOAD_BYTES=524288000' >> .env
rm -f .env.bak
```

### 3. Launch

Compose defaults to a published image containing the Rust wheel; no local build is needed.

```bash
docker compose up -d
```

### 4. Wait for model downloads

First boot downloads ~5 GB of weights from HuggingFace. Poll the logs
periodically (every 30 s or so):

```bash
docker logs --tail 20 voscript
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

# auth actually enforced? (avoid `source .env` — values could be shell-evaluated)
API_KEY=$(grep -E '^API_KEY=' .env | tail -n1 | cut -d= -f2-)
curl -sS http://localhost:8780/api/voiceprints -H "Authorization: Bearer $API_KEY"
# expect: [] (empty on first boot)

# unauthenticated requests are rejected
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:8780/api/voiceprints
# expect: 401
```

All three pass → deployment done.

## Deployment steps (macOS, native venv)

There is no qualified automatic native installation recipe for this release.
Installing `app/requirements.txt` alone omits the authenticated legacy
compatibility adaptations and required Rust wheel. Do not treat successful
dependency resolution or `/healthz` alone as native compatibility evidence.

If the user explicitly requests a manual port, first follow the
[quickstart's qualification boundary](./quickstart.en.md#03-macos-apple-silicon--intel)
and [Runtime Compatibility Tools](../app/build/README.md). It requires Python
3.11, platform-compatible Torch / torchaudio / torchvision, the separately
installed WhisperX package, authenticated compatibility adaptations, and a
same-source `voscript_core` wheel for the platform and Python ABI. Do not apply
the Docker-only `+cu126` constraints to macOS / CPU-only hosts, or silently set
`RUST_KERNEL_MODE=off` to bypass a missing extension.

Keep existing credentials and runtime data. Qualify real audio I/O, model and
checkpoint loading, alignment, and a fresh transcription on the target platform
before offering a launch or autostart recipe. Until then, use the supported
Linux / WSL2 GPU image path instead.

## Verify the container is not running as root (added in 0.2.0)

```bash
docker exec voscript id
# expected: uid=1000(app) gid=1000(app) groups=1000(app)
# if you see uid=0(root), the image is stale — git pull + rebuild.
```

## Verify the GPU is actually in use (Linux / WSL2 deployments)

```bash
docker exec voscript python -c "import torch; print('cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# expect: cuda= True NVIDIA ...
```

If it prints `cuda= False`, check the compose file's GPU reservation and
that `nvidia-ctk runtime configure` ran.

## Hand-off checklist

After deployment, give the user a crisp one-shot summary:

1. ✅ **Service URL**: `http://<host-ip-or-domain>:8780`
2. ✅ **API_KEY**: have the user read it securely on their own machine and enter
   it in the client; never echo it in chat, logs, or public output. Keep the
   existing key during upgrades rather than generating a replacement.
3. ✅ Ask them to set the following in BetterAINote → Settings →
   Transcription:
   - Private transcription base URL = the URL above
   - Private transcription API key = the same API_KEY
4. ✅ **Strong reminder**: `:8780` must not be exposed to the public
   Internet directly. Use VPN / reverse proxy with TLS / at least an IP
   allow-list. See [`security.en.md`](./security.en.md).
5. ✅ Backup reminder: preserve and back up voiceprints, AS-norm cohort, model caches, and data mounts

## Upgrade flow

When the user asks you to upgrade:

```bash
cd ~/voscript   # or actual path
git fetch origin
git diff --stat main origin/main   # show the user what will change
git pull
docker compose pull
docker compose up -d
docker logs --tail 40 voscript
curl -sf http://localhost:8780/healthz
```

If `git pull` would overwrite uncommitted local changes, **stop and ask
the user**. Never `git reset --hard`.
Back up data first, keep mounts and the existing `API_KEY`, and check whether
`.env` still pins an older `VOSCRIPT_IMAGE`. Never delete voiceprints, models,
or the cohort to perform an upgrade.

## Don't do this

- ❌ Echo `HF_TOKEN` or `API_KEY` in chat, logs, commits, or PR descriptions
- ❌ `git add .env` (it is gitignored, don't force-add it)
- ❌ Remove pins in `requirements.txt` to "make it start"
- ❌ Delete `./models/` to save disk — that's the weight cache, redownload
  costs 5 GB
- ❌ Run `docker rm -f voscript` and then expect manually-pip-installed
  packages to survive — `docker compose up --build` rebuilds from
  `requirements.txt`
- ❌ Open a 443 port / reverse proxy / public DNS record without the
  user's explicit go-ahead

## Common follow-ups

- **"BetterAINote can't reach the service"**
  → Check: 1) `API_KEY` matches on both sides exactly (no whitespace/newlines);
  2) host firewall allows `:8780`; 3) BetterAINote's host can `curl`
  the service URL.
- **"Add HTTPS?"**
  → Recommend a reverse proxy (nginx/caddy/traefik) for TLS termination
  and cert rotation. Don't patch FastAPI to serve certs itself — too much
  maintenance drag.
- **"GPU 0 is taken by something else, can we use GPU 1?"**
  → Follow the [configuration reference](./configuration.en.md) to add a local
  Compose override injecting `CUDA_VISIBLE_DEVICES`; editing `.env` alone does
  not pass that variable into the container.

## Related docs

- Human-oriented quickstart → [`quickstart.en.md`](./quickstart.en.md)
- API usage for AI agents → [`ai-usage.en.md`](./ai-usage.en.md)
- API contract → [`api.en.md`](./api.en.md)
- Security policy → [`security.en.md`](./security.en.md)
