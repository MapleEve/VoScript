# 快速安装

**简体中文** | [English](./quickstart.en.md)

这篇面向第一次部署的人。走完大约需要 15~30 分钟，其中大部分时间在等模型下载。

## 0. 先决条件

- 一台 Linux 主机，有 NVIDIA GPU（建议 ≥ 12 GB 显存；RTX 3090 / 4090 / A10 以上稳）
- 安装好 Docker 24+
- 安装好 **NVIDIA Container Toolkit**（没装 `docker run --gpus all` 会直接报错）：
  ```bash
  # 以 Ubuntu 为例
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- 一个 HuggingFace account，并且：
  1. 到 <https://huggingface.co/pyannote/speaker-diarization-3.1> 点 **Agree and access repository**
  2. 到 <https://huggingface.co/pyannote/segmentation-3.0> 也点一下同意
  3. 到 <https://huggingface.co/settings/tokens> 生成一个 **read** 权限的 token（以 `hf_` 开头）

> 这两个模型是 gated 的，跳过这一步之后服务会卡在启动时下载。

## 1. 克隆仓库

```bash
git clone https://github.com/MapleEve/openplaud-voice-transcribe.git
cd openplaud-voice-transcribe
```

## 2. 配置 .env

```bash
cp .env.example .env
```

编辑 `.env`，至少填这两项：

```env
HF_TOKEN=hf_你的_token
API_KEY=这里填一串长随机串_例如_openssl_rand_hex_32
```

如果你在中国大陆网络，建议同时加上：

```env
HF_ENDPOINT=https://hf-mirror.com
```

> 生成强随机 API key：`openssl rand -hex 32`

其他环境变量都有合理默认值，详见 [`.env.example`](../.env.example)。

## 3. 启动服务

```bash
docker compose --env-file .env up -d --build
```

第一次跑要下约 **5 GB** 的模型权重到 `./models/`，可以用下面这句跟进度：

```bash
docker logs -f voice-transcribe
```

看到 `Uvicorn running on http://0.0.0.0:8780` 就说明起来了。

也可以直接用仓库自带的脚本一把梭：

```bash
./scripts/deploy.sh
```

脚本会检查 `.env`、启容器、并等 `/healthz` 返回健康。

## 4. 验证部署

```bash
# 健康检查（永远无需鉴权）
curl -sf http://localhost:8780/healthz
# → {"ok":true}

# 需要 API_KEY 才能访问的端点
curl -sS http://localhost:8780/api/voiceprints \
    -H "Authorization: Bearer $API_KEY"
# → [] （首次一定是空数组）
```

浏览器打开 <http://localhost:8780/> 能看到一个简陋的 Web UI，可以直接上传音频测试。

## 5. 对接 OpenPlaud(Maple)

在 OpenPlaud(Maple) 的"设置 → 转录"里配：

- **Private transcription base URL**：`http://你部署的主机:8780`
- **Private transcription API key**：跟 `.env` 里 **完全一样** 的那串 `API_KEY`

配完后 OpenPlaud(Maple) 的 worker 会自动把每条录音提交到这个服务。
具体接口细节参考 [`api.zh.md`](./api.zh.md)。

## 升级

```bash
cd openplaud-voice-transcribe
git pull
docker compose --env-file .env up -d --build
```

模型权重被缓存到 `./models/`，重建镜像不会重新下载。

## 常见问题

### `nvidia-smi` 在容器里找不到
→ NVIDIA Container Toolkit 没装或者 Docker 没重启。回到第 0 步。

### 启动日志里看到 `403 Forbidden` 下载 pyannote 模型
→ 没点同意 gated 模型条款，或者 `HF_TOKEN` 写错了。

### `np.NaN was removed` 崩溃
→ `requirements.txt` 被改坏了、numpy 被升到了 2.x。保持 `numpy<2.0` 的 pin 不要动。

### 服务起来了但 OpenPlaud(Maple) 调不通
→ 检查 `API_KEY` 两端是否一模一样（大小写、空格都不能差），以及 OpenPlaud(Maple) 主机能不能
访问到 `:8780`（防火墙、docker 网络）。

### 我要备份什么
→ 只用备份 `data/voiceprints/`。其他东西丢了都能从原始音频重建。

更多线上风险看 [`security.zh.md`](./security.zh.md)。
