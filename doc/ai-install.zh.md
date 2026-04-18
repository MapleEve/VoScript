# 给 AI 用的安装部署指南

**简体中文** | [English](./ai-install.en.md)

> 这篇文档是写给 **AI agent / LLM** 看的，目标是"用户让你帮他部署这个服务"
> 这个场景。人工部署指引请看 [`quickstart.zh.md`](./quickstart.zh.md)。
>
> 配套阅读：部署完成后怎么用接口，看 [`ai-usage.zh.md`](./ai-usage.zh.md)。

## 你的工作边界

用户会让你在他的一台机器上部署 `openplaud-voice-transcribe`。你能做的：
- 通过 shell 跑命令、读写文件
- 编辑 `.env`、`docker-compose.yml`
- 跑 `docker compose`

你**不能**擅自做的：
- 不能把 `HF_TOKEN` / `API_KEY` 写死进 commit、日志、聊天记录
- 不能跳过安全硬化（不设 `API_KEY` 就启动到公网可达的端口）
- 不能跑 `git reset --hard` / `docker system prune -a` 这类破坏性操作去"修问题"
- 不能自己瞎造 HF_TOKEN，必须让用户提供

## 决策树：先判断环境

```
检查 1：有没有 NVIDIA GPU？
    $ nvidia-smi
    - 能输出 → 继续
    - 找不到命令 → 告诉用户"本服务必须 GPU"，停
    - 有卡但 CUDA 不可用 → 先修驱动

检查 2：显存够不够？（建议 ≥ 12 GB）
    $ nvidia-smi --query-gpu=memory.total --format=csv,noheader
    - < 12 GB → 警告用户可能 OOM，但可以继续（large-v3 最小约 9 GB）

检查 3：有没有 Docker + NVIDIA Container Toolkit？
    $ docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
    - 输出 GPU 信息 → OK
    - 报错 "could not select device driver ..." → 装 nvidia-container-toolkit，看下面

检查 4：用户有没有 HF_TOKEN？
    - 有 → 直接进下一步
    - 没有 → 暂停。指导用户去：
      1. https://huggingface.co/pyannote/speaker-diarization-3.1  点 Agree
      2. https://huggingface.co/pyannote/segmentation-3.0  点 Agree
      3. https://huggingface.co/settings/tokens  创建 read token
      等用户把 token 粘给你。**不要**让用户把 token 贴到 git 或公开聊天里，
      如果有风险提示他走私聊/终端粘贴。
```

## 安装 NVIDIA Container Toolkit（如果没装）

**Ubuntu / Debian**：

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

其他发行版参考 [NVIDIA 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

## 部署步骤

### 1. 选一个工作目录，克隆仓库

默认放在用户家目录下：

```bash
cd ~  # 或用户偏好的位置
git clone https://github.com/MapleEve/openplaud-voice-transcribe.git
cd openplaud-voice-transcribe
```

### 2. 生成并填 `.env`

**关键：API_KEY 必须是强随机串**。你应该主动给用户生成一个，不要让用户手写：

```bash
cp .env.example .env
API_KEY_VALUE=$(openssl rand -hex 32)
# 向用户确认是否用这个值，或让用户自带
```

然后用下面这段（或等价工具）把 `.env` 里两个关键字段填上：

```bash
# 用 sed 只改这两行，保持其他默认值
sed -i.bak "s|^HF_TOKEN=.*|HF_TOKEN=${USER_SUPPLIED_HF_TOKEN}|" .env
sed -i.bak "s|^API_KEY=.*|API_KEY=${API_KEY_VALUE}|" .env
rm .env.bak
```

**在改完之后**立刻向用户展示 `.env` 里的 `API_KEY`（只在这一次露出明文），
让他把同一个 key 配到 OpenPlaud(Maple) 的"设置 → 转录"里。之后不要再把这个值
打印到日志/聊天。

**如果用户在中国大陆网络**，还要加一行镜像：

```bash
grep -q '^HF_ENDPOINT=' .env || echo 'HF_ENDPOINT=https://hf-mirror.com' >> .env
```

### 3. 启动

```bash
docker compose --env-file .env up -d --build
```

### 4. 等模型下载完毕

首次启动会从 HuggingFace 下载约 5 GB 权重。你应该**周期性**（每 30 秒）检查日志：

```bash
docker logs --tail 20 voice-transcribe
```

关键信号：
- 看到 `Uvicorn running on http://0.0.0.0:8780` → 服务起来了
- 看到 `401 Client Error` 下载模型 → `HF_TOKEN` 错了
- 看到 `403 Forbidden` → 没接受 gated 模型条款，回到决策树的"检查 4"
- 看到 `np.NaN was removed` → 有人改了 `requirements.txt`，把 numpy 2.x 放进去了
- 超过 10 分钟还在下载 → 网络慢，建议加 `HF_ENDPOINT` 镜像

### 5. 健康检查

```bash
curl -sf http://localhost:8780/healthz
# 期望：{"ok":true}

# 鉴权有没有生效
source .env
curl -sS http://localhost:8780/api/voiceprints -H "Authorization: Bearer $API_KEY"
# 期望：[]（首次一定是空）

# 确认没 key 会被拒
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:8780/api/voiceprints
# 期望：401
```

三个都符合预期 → 部署完成。

## 验证 GPU 是真的用上了

```bash
docker exec voice-transcribe python -c "import torch; print('cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# 期望：cuda= True NVIDIA ...
```

如果输出 `cuda= False`，检查 compose 里 GPU reservation、`nvidia-ctk runtime configure` 有没有执行。

## 向用户交付的清单

部署结束后，跟用户同步这些东西（一次性、清楚）：

1. ✅ **服务地址**：`http://<主机 IP 或域名>:8780`
2. ✅ **API_KEY**：把 `.env` 里生成的值完整告诉用户**一次**；提示 "后面请自己妥善保管"
3. ✅ 让用户在 OpenPlaud(Maple) "设置 → 转录" 里：
   - Private transcription base URL = 上面的服务地址
   - Private transcription API key = 同一个 API_KEY
4. ✅ **强提醒**：`:8780` **不要**直接暴露到公网，最好挂 VPN / 反代 + TLS /
   至少白名单。详见 [`security.zh.md`](./security.zh.md)。
5. ✅ 备份建议：提醒用户 `data/voiceprints/` 要定期备份

## 升级流程

当用户让你升级服务：

```bash
cd ~/openplaud-voice-transcribe  # 或实际路径
git fetch origin
git diff --stat main origin/main   # 让用户看一下会变什么
git pull
docker compose --env-file .env up -d --build
docker logs --tail 40 voice-transcribe
curl -sf http://localhost:8780/healthz
```

如果 `git pull` 会覆盖用户本地未提交改动，**先停下来问用户**，不要 `git reset --hard`。

## 不要做的事

- ❌ 把 `HF_TOKEN`、`API_KEY` 回显到用户聊天记录之外的任何地方（日志、commit、PR）
- ❌ 把 `.env` `git add`（已经 gitignore，但别手动强加）
- ❌ 为了"启动成功"去掉 `requirements.txt` 里的版本 pin
- ❌ 为了节约磁盘删 `./models/` —— 那是模型权重缓存，删了下次要重下 5 GB
- ❌ 用 `docker rm -f voice-transcribe` 之后期待容器里手动装的包还在——记住
  `docker compose up --build` 之后会重建，一切以 `requirements.txt` 为准
- ❌ 不经用户同意就开一个 443 端口 / 反向代理 / 公网 DNS 记录

## 常见 followup

- **"OpenPlaud(Maple) 连不上这个服务"**
  → 检查：1) 两边 `API_KEY` 完全一致（无空格、无换行）；2) 主机防火墙放行了
  `8780`；3) OpenPlaud(Maple) 主机能不能 `curl` 通。
- **"能不能加个 HTTPS？"**
  → 推荐在前面挂一层 nginx/caddy/traefik 做 TLS 终止 + 证书自动续期。不要
  改 FastAPI 让它自己拿证书——维护成本高。
- **"GPU 0 被别的服务占了，能不能换 GPU 1？"**
  → 改 `.env` 里 `CUDA_VISIBLE_DEVICES=1`，然后重启 compose。

## 相关文档

- 给人看的安装文档 → [`quickstart.zh.md`](./quickstart.zh.md)
- 给 AI 看的接口使用 → [`ai-usage.zh.md`](./ai-usage.zh.md)
- 接口合同 → [`api.zh.md`](./api.zh.md)
- 安全策略 → [`security.zh.md`](./security.zh.md)
