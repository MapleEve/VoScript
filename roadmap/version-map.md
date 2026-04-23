# Version Map

## 总原则

- 1.0 前先收口，不把长期 vision 混进当前版本承诺
- 小版本继续保持细粒度子阶段，方便按边界推进而不是按大词推进
- 每个阶段优先整理接口、状态、持久化与产品语义，再考虑扩大能力面

## 主线概览

| 版本 | 阶段定位 | 关键词 |
| --- | --- | --- |
| v0.7.2 | 当前收口起点 | planning 对齐、兼容性冻结、roadmap 体系补齐 |
| v0.8.0 | 架构边界整理 | pipeline stage slot、provider 插拔、结果协议继续稳定 |
| v0.9.0 | 产品协议稳定 | batch/job/transcript 语义收敛，面向接入方稳定 |
| v0.9.5 | 1.0 冻结预演 | upgrade path、默认运维姿势、回归矩阵 |
| v1.0.0 | 1.0 发布线 | 稳定 API、稳定持久化、稳定 speaker identity 语义 |

## v0.7.2

### 0.7.2-a Planning baseline
- 建立 roadmap 文档树
- 明确 1.0 前主线与 post-1.0 分层
- 把 compatibility 原则从零散表述整理成统一文档

### 0.7.2-b Scope discipline
- 明确 VoScript 长期仍是会议/录音场景应用后端
- 明确不会在 1.0 前扩成通用语音 runtime / model zoo / 训练平台
- README 补充路线说明，但不改当前产品叙事

## v0.8.0

### 0.8.0-a Canonical pipeline shape
- 固化 canonical stage sequence
- 为 ingest/normalize/enhance/vad/asr/diarization/embedding/voiceprint_match/punc/postprocess/artifacts 建立稳定 slot 认知
- 把“可以替换 provider”写成架构约束，而不是零散实现习惯

### 0.8.0-b Provider boundary cleanup
- 明确 provider 层负责模型/后端差异
- 明确领域策略仍留在 voiceprints / transcript 语义层
- 为未来可插拔推理后端预留边界，但不引入 post-1.0 级别大范围能力承诺

### 0.8.0-c Artifact contract hardening
- 明确转录结果、任务状态、持久化物的责任分界
- 为后续 batch/streaming 统一协议打基础

## v0.9.0

### 0.9.0-a Job and transcript semantics
- 稳定 batch 提交、轮询、结果获取的统一语义
- 明确 status lifecycle、失败恢复、幂等/去重预期
- 保持结果对象对接入方可预测

### 0.9.0-b Speaker identity semantics
- 稳定 speaker_label、speaker_name、speaker_id 的边界
- 明确人工纠错、自动匹配、持久化声纹之间的职责关系
- 建立 transcript state machine 的最小闭环认知

### 0.9.0-c Product-facing API consistency
- 面向 BetterAINote 与其它调用方整理 product API 视角
- 保持 HTTP API 是当前最重要的集成面

## v0.9.5

### 0.9.5-a Freeze rehearsal
- 逐项检查 compatibility map
- 对外文档与行为承诺对齐
- 明确哪些能力属于 1.0，哪些明确延后

### 0.9.5-b Upgrade and migration posture
- 为旧部署、旧结果、旧声纹库提供升级说明原则
- 形成 1.0 前的回归检查矩阵

### 0.9.5-c Release candidate discipline
- 以“少惊喜、少破坏、少隐式迁移”为原则清理尾项

## v1.0.0

### 1.0.0-a Stable self-hosted backend
- 稳定自托管部署体验
- 稳定核心 HTTP API
- 稳定转录结果与作业状态读取方式

### 1.0.0-b Stable speaker memory baseline
- 稳定 voiceprints.db 的角色
- 稳定 speaker identity 的产品语义
- 稳定人工修正与自动识别并存时的预期

### 1.0.0-c Stable extension boundary
- 可以继续替换推理后端，但不改变 1.0 对外产品定义
- post-1.0 vision 另行推进，不反向修改 1.0 的定义
