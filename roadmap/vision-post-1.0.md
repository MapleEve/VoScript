# Post-1.0 Vision

本文档只描述 1.0 之后的长期方向，不构成 v0.7 / v0.8 / v0.9 / v1.0 的执行承诺。

## 边界先说清楚

VoScript 的长期方向会继续扩大，但扩大方式不是“什么语音能力都做一点”。

我们的长期方向是：

- 统一语音应用后端
- 更具体地说，是 meeting speech backend

这里的“统一”，不是做一个泛化到所有语音任务的基础设施总包，而是把会议、录音、访谈、归档、复盘这类应用真正需要的后端能力统一起来。

## 我们不做什么

VoScript 的 post-1.0 vision 不是：

- 另一个 FunASR
- 通用训练平台
- model zoo
- 全任务语音中台
- 某个现成产品的复刻版

原因很简单：这些方向会把产品拉向“广度优先的通用语音基础设施”，而 VoScript 的优势来自“围绕真实录音工作流，把后端产品语义做深”。

我们当然会持续吸收行业里的好做法，但不会为了看起来更大，就把 scope 扩成一个无边界平台。

## 长期核心能力

post-1.0 真正要扩的是下面这些能力层：

### 1. 可插拔推理后端

- 同一套产品协议下，可以接多种 ASR / VAD / diarization / enhancement / separation 后端
- 后端可替换，不要求上层产品 API 跟着重写
- 重点是“产品后端统一”，不是“做一个通用声学 runtime”

### 2. batch / streaming 协议统一

- 让离线文件转录与在线流式处理尽量共享一套概念模型
- 对调用方暴露统一的 job / session / transcript 更新语义
- 同一个应用可以在录后整理、实时会议、边录边转之间平滑切换

### 3. speaker memory

- 把持久化声纹从“库里存了几个 embedding”提升成长期可用的 speaker memory
- 重点不是追求抽象名词，而是支持真实产品需求：长期识别、回溯修正、跨会话延续、受控更新
- speaker memory 要服务 transcript 和产品工作流，而不是独立膨胀成另一个系统

### 4. transcript state machine

- 逐字稿不再只是一次性静态结果，而是有状态演化的对象
- 状态机会覆盖：自动识别、人工修正、speaker relabel、增量更新、最终冻结、导出
- 这样才能支撑 batch / streaming 统一，以及更可靠的 product API

### 5. orchestration / product API

- 更关注“应用怎样消费语音能力”，而不是“模型怎样排列组合”
- API 会朝产品编排层发展：任务、会话、结果、修正、导出、身份记忆、权限与集成
- VoScript 的长期竞争力更像应用后端，而不是模型集市

## VoScript 与 sherpa-onnx 的区别

这部分必须明确。

### sherpa-onnx 更像什么

sherpa-onnx 更像通用语音 runtime / 多任务基础平台。

它的思路更接近：

- 提供广泛的语音任务能力边界
- 面向多种设备、部署形态、推理后端
- 更强调 runtime 层、任务层、模型接入层的通用性

从能力边界上，我们完全可以参考这类项目覆盖的任务面，例如：

- ASR
- VAD
- diarization
- enhancement
- separation

### VoScript 更像什么

VoScript 的长期 vision 仍然是会议/录音场景的应用后端，而不是通用声学 runtime。

也就是说，哪怕未来 VoScript 接入更多推理后端、覆盖更多语音处理环节，它的目标仍然是：

- 服务会议、录音、访谈、纪要、回放、归档这类应用场景
- 把 speaker memory、transcript state machine、product API 做成完整产品后端
- 保持“应用语义优先”，而不是把自己定义成任何任务都能跑的基础 runtime

### 参考能力边界，不复制 scope

我们可以参考 sherpa-onnx 这类项目的能力边界，理解一个现代语音系统通常会触达哪些环节；但这不意味着 VoScript 要把自己的 scope 扩成通用语音基础设施。

区别可以概括成一句话：

- sherpa-onnx 更偏“通用语音 runtime / 多任务基础平台”
- VoScript 更偏“会议与录音场景的语音应用后端”

这个区别在 1.0 后也不会消失。

## 为什么 vision 要继续扩大

因为只做“上传音频 -> 返回转录结果”的服务，长期会天花板很低。

但扩大应该沿着正确方向：

- 从单次转录走向长期可维护的 transcript lifecycle
- 从一次匹配走向 speaker memory
- 从一个文件接口走向 batch + streaming 统一协议
- 从模型调用走向 orchestration / product API

而不是沿着错误方向：

- 从一个产品后端变成没有边界的通用语音平台
- 从 meeting speech backend 变成“我们也做一切语音任务”
- 从自己的产品定义滑向对现成产品的模仿

## 对 1.0 前 roadmap 的约束

为了保证当前路线不失焦，post-1.0 vision 对 1.0 前只有两条约束：

1. 当前架构要保留未来插拔与协议统一的空间
2. 当前版本不能因为“未来可能要做”就提前承诺大范围 scope

所以 1.0 前仍然先把基础产品打稳，1.0 后再进入更大的 horizon。
