# openplaud-voice-transcribe

**简体中文** | [English](./README.en.md)

自托管的 GPU 转录服务，带持久化说话人声纹。为 [**OpenPlaud(Maple)**](https://github.com/MapleEve/openplaud)
打造的私有后端，也可以独立作为一个 FastAPI 服务使用。

```
音频  ──►  faster-whisper large-v3  （转录）
      ──►  pyannote 3.1             （说话人分离）
      ──►  ECAPA-TDNN               （声纹提取）
      ──►  VoiceprintDB             （与已注册声纹做余弦匹配）
      ──►  带时间戳和已识别说话人姓名的文本
```

## 文档

所有详细文档都在 [`doc/`](./doc/)，默认中文，每一份都有对应英文：

| 主题 | 中文 | English |
| --- | --- | --- |
| 快速安装 | [quickstart.zh.md](./doc/quickstart.zh.md) | [quickstart.en.md](./doc/quickstart.en.md) |
| API 参考 | [api.zh.md](./doc/api.zh.md) | [api.en.md](./doc/api.en.md) |
| **给 AI 的安装部署指南** | [ai-install.zh.md](./doc/ai-install.zh.md) | [ai-install.en.md](./doc/ai-install.en.md) |
| **给 AI 的接口使用指南** | [ai-usage.zh.md](./doc/ai-usage.zh.md) | [ai-usage.en.md](./doc/ai-usage.en.md) |
| 安全策略 | [security.zh.md](./doc/security.zh.md) | [security.en.md](./doc/security.en.md) |
| 更新日志 | [changelog.zh.md](./doc/changelog.zh.md) | [changelog.en.md](./doc/changelog.en.md) |

人第一次部署 → [快速安装](./doc/quickstart.zh.md)；
AI agent 帮用户部署 → [给 AI 的安装部署指南](./doc/ai-install.zh.md)；
AI agent 调用接口 → [给 AI 的接口使用指南](./doc/ai-usage.zh.md)。

## 为什么单独放一个仓库

OpenPlaud(Maple) 是一个单用户控制面板。把 whisper / pyannote 加载进显存、
常驻 GPU、做说话人分离、维护一套声纹库——这些重活都放在一个私有 HTTP API 后面，
这样 OpenPlaud(Maple) 本体就不用在浏览器里跑 GPU 模型，也不用把原始声纹数据暴露出去。

这个仓库就是那个私有 API。OpenPlaud(Maple) 上传音频、轮询任务、把转录结果存到本地
数据库，当用户做声纹登记时会调这里的 voiceprint 接口。

## 功能

- 异步任务流水线（`queued → converting → transcribing → identifying → completed`）
- 中文 + 多语种转录（faster-whisper large-v3）
- 说话人分离（pyannote 3.1）
- 持久化声纹：**一次登记，后续录音自动识别**（余弦相似度 ≥ 0.75 视为命中）
- 稳定的 HTTP 合同，OpenPlaud(Maple) 的
  [`voice-transcribe-provider.ts`](https://github.com/MapleEve/openplaud/blob/main/src/lib/transcription/providers/voice-transcribe-provider.ts)
  和 [`voice-transcribe/client.ts`](https://github.com/MapleEve/openplaud/blob/main/src/lib/voice-transcribe/client.ts)
  可以直接对接
- 所有 `/api/*` 路由支持可选 Bearer / `X-API-Key` 鉴权
- `/` 自带一个轻量 Web UI，方便单独测试

## 30 秒上手

```bash
git clone https://github.com/MapleEve/openplaud-voice-transcribe.git
cd openplaud-voice-transcribe

cp .env.example .env
# 编辑 .env —— 至少要填 HF_TOKEN 和 API_KEY

docker compose up -d --build
curl -sf http://localhost:8780/healthz
```

完整步骤 + 排障清单看 [`doc/quickstart.zh.md`](./doc/quickstart.zh.md)。

## 和 OpenPlaud(Maple) 对接

在 OpenPlaud(Maple) 的"设置 → 转录"里配：

- **Private transcription base URL**：`http://<主机>:8780`
- **Private transcription API key**：跟 `.env` 里的 `API_KEY` 一致

之后 OpenPlaud(Maple) 的 worker 就会把每条录音都丢给这个服务。接口细节见
[`doc/api.zh.md`](./doc/api.zh.md)。

## License

MIT —— 看 [LICENSE](./LICENSE)。
