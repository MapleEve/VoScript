# 更新日志

**简体中文** | [English](./changelog.en.md)

## 0.1.0 — 首次公开发布

- 首次公开发布 [OpenPlaud](https://github.com/MapleEve/openplaud) 的私有转录后端。
- 异步任务流水线：`queued → converting → transcribing → identifying → completed`。
- faster-whisper `large-v3` + pyannote `3.1` + ECAPA-TDNN 声纹提取。
- 持久化声纹库，基于余弦相似度自动匹配。
- 所有 `/api/*` 路由支持可选的 `API_KEY` Bearer 鉴权。
- 可移植的 `docker-compose.yml`（数据/模型路径都通过环境变量配置）。
- 必要的版本 pin，让 `pyannote.audio==3.1.1` 仍可用：
  - `numpy<2`（pyannote 3.1.1 用了 `np.NaN`，numpy 2.x 已移除）。
  - `huggingface_hub<0.24`（保留 pyannote 3.1.1 调用的 `use_auth_token` kwarg）。
