# VoScript Roadmap

本目录只管理 planning 文档，不代表已经承诺交付的实现。

## 1.0 前主线

VoScript 在 1.0 之前的主线很明确：

- 继续把产品收口成稳定的“自托管 GPU 转录服务 + 持久化声纹 + HTTP API”
- 把当前接口、持久化格式、部署体验、可运维性整理到可长期使用的状态
- 让会议录音、访谈录音、日常归档录音这类核心场景先跑稳，而不是一边做产品一边不断扩 scope

1.0 前的 roadmap 关注的是：

- 结果协议与兼容性冻结
- 任务状态与恢复语义稳定
- 声纹库、speaker label、人工纠错语义稳定
- 插件化/阶段化架构为 1.0 做好边界整理
- 面向产品接入的 API、批处理体验、运维能力补齐

## 1.0 后 vision 单独管理

1.0 之后，VoScript 会进入更大的 vision horizon，但这部分不会被错误地下压为当前版本承诺。

因此：

- 1.0 前版本承诺只看各版本文档与 compatibility map
- 1.0 后长期方向单独记录在 [vision-post-1.0.md](./vision-post-1.0.md)
- 任何 post-1.0 方向都不自动等于 v0.8/v0.9/v1.0 的 scope

这样做是为了避免两种常见错误：

1. 把长期愿景写进短期版本，导致 roadmap 失真
2. 因为看见长期愿景，就误以为 VoScript 会扩成通用语音基础设施

## 文档导航

- [version-map.md](./version-map.md): 版本与子阶段总览
- [compatibility-map.md](./compatibility-map.md): 1.0 前必须守住的兼容边界
- [v0.7.2.md](./v0.7.2.md): 当前收口阶段
- [v0.8.0.md](./v0.8.0.md): 管线边界与插件化整理
- [v0.9.0.md](./v0.9.0.md): 产品协议与作业语义稳定
- [v0.9.5.md](./v0.9.5.md): 1.0 前冻结与迁移准备
- [v1.0.0.md](./v1.0.0.md): 1.0 定义
- [vision-post-1.0.md](./vision-post-1.0.md): 1.0 之后的长期方向

## 读取方式

如果你要判断“接下来这个版本到底承诺什么”，优先顺序是：

1. 对应版本文档
2. compatibility map
3. version map
4. post-1.0 vision

也就是说，vision 用来决定长期方向，不用来偷偷扩写当前版本 scope。
