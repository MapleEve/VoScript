# VoScript 定位研究（克制版）

## 执行摘要

这份研究不把 VoScript 夸大成“通用语音平台”，也不把它提前定义成已经成立的“大一统 meeting speech backend”。

当前更稳妥的判断是：

**VoScript 现在最成立的产品定义，是一个面向会议/录音整理场景的、自托管、身份增强型转录后端。**

这里的“身份增强”指的是：它不仅输出转录文本和 speaker 分离结果，还把持久化 voiceprint、真名映射、会后导出、异步 job、去重与 API 接入组合到了一起。

这不等于它已经验证了一个很大的市场空挡。更准确的说法是：

- 它可能填补了一个**窄但真实**的空挡；
- 这个空挡成立的前提，是“固定团队/组织内部，同一批人反复开会、反复录音、反复整理”；
- 如果脱离这个前提，把它讲成“所有转录用户都需要”的产品，就会落入伪需求。

因此，研究结论必须分三层写：
1. 已被当前代码和产品叙事支撑的事实；
2. 基于竞品对照得到的合理定位；
3. 仍需未来验证的市场推断。

## 一、研究方法与证据边界

### 这份研究看什么
- 当前 VoScript 仓库的 README、CLAUDE、API 文档与代码结构
- 公开可见的基础平台、实时引擎、商业会议产品与中文圈产品定位
- 各对象在“产品抽象层”和“能力深度”上的相对位置

### 这份研究不做什么
- 不把公开文案当作完整产品真相
- 不把愿景直接当作已验证市场
- 不把功能存在等同于需求成立

### 结论口径
- 事实：仓库与公开文档能直接支持
- 定位：在已知竞品中相对成立的产品身位
- 假设：post-1.0 以后可能扩张的方向，必须和 1.0 前主线分开

## 二、产品轴定义

产品轴回答的是：**这类东西本质上在卖什么。**

### P1 基础模型 / 训练平台
- 关键词：训练、微调、评测、模型生态、研究到产业桥接
- 代表：FunASR

### P2 通用推理 runtime / SDK
- 关键词：本地推理、多任务 runtime、跨平台 SDK、端侧/边缘部署
- 代表：sherpa-onnx

### P3 流式增量转录引擎
- 关键词：partial / revised / final、低延迟增量输出、回看修正
- 代表：Whisper-Streaming

### P4 实时 ASR 服务层
- 关键词：实时 server、WebSocket/RTC、API 兼容层、会话输入
- 代表：WhisperLiveKit、Deepgram、AssemblyAI Realtime、Gladia

### P5 通用转录 API / 异步语音后端
- 关键词：文件上传、异步 job、查询、导出、批处理转录
- 代表：Azure Speech、Google Cloud STT、腾讯云语音、VoScript 的一部分

### P6 身份增强型转录后端
- 关键词：speaker diarization 之上，再做长期 voiceprint、真名映射、跨会话识别
- 代表：VoScript 当前最接近这个类别

### P7 会议/录音工作流后端
- 关键词：面向 meeting/recording workflow 的后端能力，包括转录、说话人、导出、资产管理、工作流接入
- 代表：VoScript 的中期方向

### P8 会议 SaaS / 会议纪要产品
- 关键词：终端协作体验、纪要、待办、组织工作台、会议内闭环
- 代表：飞书妙记、腾讯会议 AI、钉钉会议纪要、Otter、Fireflies、Notta、讯飞听见、通义听悟

## 三、能力轴定义

能力轴回答的是：**它到底做到哪一层。**

1. batch 文件转录
2. async job / polling
3. streaming / live session
4. partial / revised / final
5. word-level timestamps / alignment
6. diarization
7. speaker memory / voiceprint library
8. true-name attribution
9. enroll / update / delete / rebuild
10. export / artifacts
11. Web UI / end-user shell
12. HTTP API / SDK / WebSocket
13. self-hosted / on-prem
14. cloud / SaaS delivery
15. transcript state machine
16. workflow integration / orchestration

## 四、定位坐标轴与身位判断

为了避免继续只用表格堆信息，这里用一组更直观的二维轴。

### 坐标轴定义
- **X 轴：从基础能力层 → 产品工作流层**
- **Y 轴：从匿名转录/通用能力 → 身份感知/长期记忆**

### 相对位置判断
- **FunASR**：偏左下。是基础模型/训练平台，能力宽，但不以组织内身份记忆为产品中心。
- **sherpa-onnx**：左下。典型通用 runtime / SDK，离工作流后端和长期身份都较远。
- **Whisper-Streaming**：左下偏右。比 runtime 更接近实时引擎，但仍主要停留在匿名流式能力层。
- **WhisperLiveKit**：中部偏右、略向上。已进入实时服务层，但长期 voiceprint memory 不是它的主公开叙事。
- **Deepgram / AssemblyAI / Gladia**：中右偏上。API 和实时能力强，但长期身份记忆并非第一主轴。
- **飞书妙记 / 腾讯会议 AI / 钉钉会议纪要**：右上区域，但实名更多来自组织账号/参会关系，而不是独立持久化声纹库。
- **讯飞听见 / 通义听悟**：右中上。更接近纪要产品壳，兼具一定转录与整理能力，但长期身份记忆并不突出。
- **Otter / Fireflies / Notta**：右上。典型会议纪要 SaaS，工作流和协作体验强。
- **VoScript 当前**：靠右上，但低于完整会议产品层；更准确是**身份增强型转录后端**。
- **VoScript post-1.0**：应在当前点位基础上继续向右移动、略向上抬升，目标是更强的工作流后端 / meeting speech backend，而不是 SaaS 前台。

### 一句话结论

**VoScript 当前身位 = 靠右上，但仍低于完整会议产品层；更准确是“身份增强型转录后端”。**

### 不该去的身位
- 不该回到左下，变成“基础模型平台”叙事
- 不该停在左中，变成“通用 runtime / SDK”叙事
- 不该只补实时能力后停在中部，变成“普通实时 ASR 服务层”
- 不该一路向最右上外壳漂移，变成“泛会议 SaaS / 纪要产品”
- 不该为了显得更大而扩成“什么语音任务都做”的中台

## 五、竞品矩阵（简版）

| 对象 | 产品轴 | 最强能力 | 不像它的地方 | 对 VoScript 的意义 |
| --- | --- | --- | --- | --- |
| FunASR | P1 | 训练+推理+模型生态 | 不是业务产品壳 | 能力参照，不是产品定位参照 |
| sherpa-onnx | P2 | 本地 runtime / SDK / 多任务 | 不是身份感知工作流后端 | 底层能力边界参照 |
| Whisper-Streaming | P3 | partial / revised / final | 不做长期身份层 | 可借鉴 transcript state machine |
| WhisperLiveKit | P4 | 实时 server / API 兼容 | 不主打长期声纹库 | 可借鉴 streaming 协议层 |
| Deepgram / AssemblyAI / Gladia | P4 | 实时语音 API / WS | 长期身份记忆弱 | 看实时服务市场标准 |
| Azure / Google / 腾讯云语音 | P5 | 通用语音 API | 不是身份增强产品 | 看基础 API 基准线 |
| 飞书 / 腾讯会议 / 钉钉 | P8 | 会议内实名 + 协作闭环 | 非自托管；不是独立后端能力层 | 看会议 SaaS 的右上边界 |
| 讯飞听见 / 通义听悟 | P8 | 纪要产品体验 | 不是长期组织声纹后端 | 看中文圈纪要产品壳 |
| Otter / Fireflies / Notta | P8 | 协作型纪要 SaaS | 自托管弱，身份记忆不以声纹库为主 | 看全球会议产品壳 |
| VoScript | P6 / P7 | 长期声纹库 + 真名级会后逐字稿 + API | 不是训练平台，不是 runtime，不是完整会议 SaaS | 当前最合理定位 |

## 六、VoScript 填补了什么空挡

VoScript 填补的不是“转录”空挡，而是一个更窄的组合空挡：

**固定团队/组织内部，长期声纹库 + 真名级会后逐字稿 + 工作流集成**。

这个空挡之所以成立，是因为很多产品和平台虽能做到以下一部分：
- transcription
- diarization
- streaming
- API
- 会议纪要

但并不把下面这一组合作为一等能力：
- 组织内持续复用的 voiceprint library
- 真名级 speaker attribution
- 自托管部署
- 录后逐字稿沉淀
- 对上层 agent / workflow 的可消费结果

这不是单点能力空挡，而是“组合能力空挡”。

## 七、哪些地方可能是伪需求

### 不是强需求的人群
- 一次性录音/一次性访谈用户
- 只关心“有个转录结果”而不关心“谁说的”用户
- 没有固定成员集合的开放场景
- 不愿维护任何 speaker 库的轻量用户

### 伪需求化的风险
如果把 VoScript 讲成：
- “所有转录用户都需要”
- “所有会议产品都缺它”
- “所有人都需要长期声纹库”

那就是伪需求叙事。

### 真正成立的需求前提
- 同一批人反复开会
- 同一批人反复录音
- 同一批人反复整理记录
- 组织愿意维护 speaker memory
- 组织真的在为“谁说了什么”付出持续成本

## 八、VoScript 与 sherpa-onnx 的差异

一句话：
- **sherpa-onnx 更偏通用语音 runtime / 多任务基础平台**
- **VoScript 更偏会议与录音场景的身份增强型语音后端**

本质区别不在“功能多少”，而在抽象层不同：

### sherpa-onnx 更关心
- 多任务覆盖面
- 本地推理与 SDK 分发
- 跨设备、跨语言、跨平台部署
- runtime 层能力边界

### VoScript 更关心
- 录后整理场景
- speaker memory
- 真名级归因
- result.json / status.json / voiceprints.db 这些稳定产品事实源
- 异步任务、导出、工作流接入

所以：
- sherpa-onnx 是基础层参照物
- VoScript 是应用层产品后端
- 两者不该在同一层上竞争

## 九、1.0 前后边界

### 1.0 前
继续收口 VoScript 自己：
- 目录架构
- contract
- artifact
- speaker 事实源
- backend interface
- scheduling / observability
- compatibility freeze

不要在 1.0 前就把项目写成：
- 通用语音平台
- 实时通话平台
- 训练平台
- 全任务语音中台

### 1.0 后
可以扩大到更完整的 **meeting speech backend**，但扩大方向应该是：
- 可插拔推理后端
- batch / streaming 协议统一
- speaker memory
- transcript state machine
- orchestration / product API

而不是：
- model zoo
- 训练/微调平台
- 通用 runtime SDK 平台
- 什么语音任务都做的中台

## 十、战略建议 / 反模式

### 应该继续做的
1. 继续把 VoScript 收到 1.0
2. 强化 identity-aware 这条主轴
3. 把 HTTP API + async job + artifacts + speaker memory 做成稳定后端能力
4. 1.0 后再考虑 streaming / transcript state machine / meeting speech backend

### 最值得继续加深的能力
- speaker memory
- true-name attribution
- artifact consistency
- workflow integration
- post-1.0 的 batch/streaming 统一协议

### 反模式
- 反模式 1：变成“另一个 FunASR”
- 反模式 2：变成“另一个 sherpa-onnx”
- 反模式 3：变成泛化“什么转录都做”的平台
- 反模式 4：为了追实时而丢掉现有身份增强护城河
- 反模式 5：为了看起来更大而把需求讲成泛需求

## 最终一句话

VoScript 最合理的长期路线，不是成为“最全的语音平台”，而是成为：

**固定团队/组织内部，围绕会议与录音整理场景的、可自托管的身份增强型语音后端。**
