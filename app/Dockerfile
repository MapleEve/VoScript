# Python 3.11 preserves the cp311 Rust/DeepFilterLib wheel ABI.
# Install official CUDA 12.6 wheels without changing the Python/native-wheel ABI.
# Official Docker Hub index digest (Python 3.11.17, Debian bookworm):
# https://hub.docker.com/layers/library/python/3.11.17-slim-bookworm/images/sha256-8a00ecb83ed9d0864dfd102a1bdb2067fda907ceb30bb86d09da9ee558b19590
FROM python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# CUDA 12.6 native-driver floor; do not restrict the operator's visible GPUs.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.6,driver>=560.35.05"
# CTranslate2 dlopens these wheel-provided libraries rather than torch's loader.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Optional mirrors for users behind geographic restrictions.
#   docker build --build-arg APT_MIRROR=https://mirrors.aliyun.com \
#                --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple ./app
ARG APT_MIRROR=""
ARG PIP_INDEX_URL=""
ARG VOSCRIPT_CORE_WHEEL=""

# UID/GID the container runs as. Defaults to 1000 so bind mounts from a
# typical Linux host (where the first non-root user is uid 1000) work
# out of the box. Hosts with a different owner should pass a build arg
# matching `id -u` of the user who owns DATA_DIR.
ARG APP_UID=1000
ARG APP_GID=1000

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian-security|${APT_MIRROR}/debian-security|g; s|http://deb.debian.org/debian|${APT_MIRROR}/debian|g" /etc/apt/sources.list.d/debian.sources; \
    fi

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user. An RCE inside the container only owns the service's
# own uid — it cannot read root-owned files mounted elsewhere on the host.
RUN groupadd -g ${APP_GID} app \
    && useradd -u ${APP_UID} -g ${APP_GID} -m -s /sbin/nologin app \
    && mkdir -p /cache /data \
    && chown -R app:app /cache /data

WORKDIR /app

COPY --chown=app:app requirements.txt requirements.constraints.txt requirements.docker.constraints.txt ./
COPY build/ /build/
ENV PIP_CONSTRAINT=/app/requirements.docker.constraints.txt
RUN set -eu; \
    if [ -n "$PIP_INDEX_URL" ]; then export PIP_INDEX_URL; fi; \
    python -m pip install --no-cache-dir --upgrade pip==26.3 setuptools==84.2.1; \
    python -m pip install --no-cache-dir numpy Pillow
# Torch stays outside business requirements. The published cu126 builds preserve
# CTranslate2's CUDA12/cuDNN9 ABI and meet GHSA-rrmf-rvhw-rf47's 2.13.0 fix floor.
# https://pytorch.org/get-started/previous-versions/#v2130
RUN python -m pip install --no-cache-dir \
    torch==2.13.0+cu126 torchaudio==2.11.0+cu126 torchvision==0.28.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
RUN set -eu; \
    if [ -n "$PIP_INDEX_URL" ]; then export PIP_INDEX_URL; fi; \
    python -m pip download --no-cache-dir --no-deps --only-binary=:all: \
        --require-hashes -r /build/legacy_wheels.txt -d .compat-wheels; \
    python /build/patch_legacy_audio.py --wheel-dir .compat-wheels --verify-wheels-only; \
    python -m pip install --no-cache-dir .compat-wheels/*.whl -r requirements.txt; \
    python -m pip install --no-cache-dir --no-deps whisperx==3.3.1; \
    python /build/patch_legacy_audio.py --wheel-dir .compat-wheels \
        --report /usr/local/share/voscript/legacy-compat.json; \
    rm -rf .compat-wheels /build
COPY --chown=app:app .wheelhouse/ /wheelhouse/
RUN set -eu; \
    if [ -n "$VOSCRIPT_CORE_WHEEL" ]; then \
        pip install --no-cache-dir "/wheelhouse/$VOSCRIPT_CORE_WHEEL"; \
    else \
        set -- /wheelhouse/voscript_core-*.whl; \
        if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then \
            echo "Exactly one CI-built cp311 voscript_core wheel is required." >&2; exit 1; \
        fi; \
        pip install --no-cache-dir "$1"; \
    fi; \
    rm -rf /wheelhouse

COPY --chown=app:app . .

# HuggingFace cache lives under the app user's reach, not /root.
ENV HF_HOME=/cache \
    HUGGINGFACE_HUB_CACHE=/cache \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ETAG_TIMEOUT=3

USER app

EXPOSE 8780

VOLUME ["/data", "/cache"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8780"]
