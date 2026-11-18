# 完整配置与调参参考

**简体中文** | [English](./configuration.en.md)

本文是 VoScript v0.8.5 的公开配置索引，覆盖当前代码已经读取并生效的
环境变量、`POST /api/transcribe` 的请求级覆盖语义，以及还没有暴露为稳定
配置项的内部默认值。没有在本文列出的 Whisper / diarization / AS-norm 变量，
不要假定已经可用。

## 配置来源与优先级

| 层级 | 示例 | 优先级 |
| --- | --- | --- |
| API 请求字段 | `denoise_model=deepfilternet`、`snr_threshold=8` | 只影响本次任务，优先于服务端 env |
| 容器环境变量 | `.env` 经 `docker-compose.yml` 注入容器 | 服务级默认值 |
| 代码默认值 | `app/config.py` | 未设置 env 时使用；空值和非法值的处理因配置项而异 |

`POST /api/transcribe` 当前只暴露 `language`、`min_speakers`、`max_speakers`、
`denoise_model`、`snr_threshold` 和 `no_repeat_ngram_size`。其它 pipeline
参数即使内部存在默认值，也还不是公开 API 参数。

## 服务基础配置

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `API_KEY` | 空 | 设置后，除 `/`、`/healthz`、`/docs`、`/redoc`、`/openapi.json`、`/static/*` 外，其它接口都需要 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`。 |
| `ALLOW_NO_AUTH` | `0` | 仅在 `API_KEY` 为空时使用。设为 `1` 只是确认无鉴权运行并抑制启动警告，不会开启任何额外保护。 |
| `CORS_ALLOW_ORIGINS` | `*` | 逗号分隔的 CORS 允许源。公网部署建议收窄。 |
| `HOST_PORT` | `8780` | compose 发布到宿主机的端口；不是应用内部 env。 |
| `VOSCRIPT_IMAGE` | `ghcr.io/mapleeve/voscript:v0.8.5` | Compose 使用的发布镜像，可覆盖为兼容的已发布 tag 或 digest；普通安装无需本地构建。 |
| `MAX_UPLOAD_BYTES` | `2147483648` | 单次上传最大字节数，超过后返回 `413` 并清理半截文件。 |
| `DATA_DIR` | `/data` | 容器内数据根目录；转写结果、上传文件、声纹库都在这里。compose 默认把宿主 `./data` 挂载到 `/data`。 |
| `MODEL_CACHE_DIR` | `./models` | compose 用的宿主模型缓存目录，挂载到容器 `/cache` 和只读 `/models`。 |
| `APP_UID` / `APP_GID` | `1000` / `1000` | 容器运行用户。宿主 `DATA_DIR` 和 `MODEL_CACHE_DIR` 必须让这个 uid/gid 可写。 |
| `DEVICE` | `cuda` | pipeline 推理设备。CPU/macOS/无 NVIDIA 环境设为 `cpu`。 |
| `CUDA_VISIBLE_DEVICES` | 未设置 | 可选 NVIDIA 可见卡限制。默认不注入该变量，compose 会请求 Docker 暴露的所有可用 GPU。只有需要把容器限制到某些卡时，才通过 `docker-compose.override.yml` 或显式 operator env 注入；容器内 `cuda:0` 是可见集合的第 0 张，不一定等于宿主物理 GPU0。CPU-only 请设置 `DEVICE=cpu`。 |
| `FFMPEG_TIMEOUT_SEC` | `1800` | ffmpeg 转码超时秒数，超时返回 `504`。 |
| `JOBS_MAX_CACHE` | `200` | 终态 job 的内存历史缓存上限；活跃任务不因该上限被淘汰。被淘汰的终态任务仍可从磁盘 `status.json` / `result.json` 查询。 |
| `TRANSCRIPTION_MAX_ACTIVE_JOBS` | `200` | 未完成转写任务的接纳上限，包含排队与执行中的任务；达到上限时新任务返回 `503`。设为 `0` 关闭该限制。 |
| `TRANSCRIPTION_MAX_IN_FLIGHT_JOBS` | `4` | 按唯一音频 hash 计数的进行中任务上限，不表示可同时执行 4 个 GPU 任务；达到上限时新任务返回 `503`。设为 `0` 关闭该限制。 |
| `TRANSCRIPTION_MIN_FREE_DISK_BYTES` | `1073741824` | 新任务接纳要求数据磁盘至少剩余 1 GiB；不足时返回 `503` 并清理本次上传。设为 `0` 关闭该检查；不是磁盘空间预留。 |
| `MODEL_IDLE_TIMEOUT_SEC` | `180` | GPU 模型空闲卸载超时，默认 180 秒（3 分钟）。设为 `0` 可关闭空闲卸载并保持模型常驻。开启后，只有串行 GPU 运行时空闲达到该秒数才释放已加载模型；下一次 reload 时 ASR、diarization 和 embedding 会在各自 lazy load 时分别选择当前可见 CUDA 中空闲显存最多的设备。 |
| `RUST_KERNEL_MODE` | `required` | 所选 Rust-backed 声纹计分、结果后处理和 artifact manifest helper 路径必须成功导入并执行，否则 fail closed。`off` 仅供显式回退到 Python 实现，不再是默认值。 |

`MODELS_DIR` 和 `LANGUAGE` 在配置模块里有定义，但当前主 HTTP 转写路径
没有把它们作为稳定公开调参入口使用：Whisper 本地 checkpoint 查找仍使用
`/models/faster-whisper-<WHISPER_MODEL>`，语言默认请通过请求字段 `language`
控制或留空自动检测。

空闲卸载是缓解显存占用的功能，不是吞吐优化。卸载 daemon 与转写任务共用同一个
GPU 串行 semaphore，并且会在拿到 semaphore 后重新读取 idle 时间戳；如果等待期间
有新任务排队或刚完成，不会基于等待前的旧判断卸载。CUDA cache 释放是 best-effort，
CPU-only 环境会安全跳过。

Docker Compose 默认用 `count: all` 请求所有可用 NVIDIA GPU，且默认不设置
`CUDA_VISIBLE_DEVICES`，所以容器能看到 Docker 暴露的全部 GPU。`DEVICE=cuda`
表示让每个模型在自身 lazy load 时按当前空闲显存自动选可见卡；`DEVICE=cuda:0`
或其它带索引的值表示固定到容器内对应可见索引，不会自动切到别的卡。

如需限制可见卡，建议在本地创建不提交的 `docker-compose.override.yml`：

```yaml
services:
  voscript:
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
```

然后在本地 `.env` 或启动命令里显式设置 `CUDA_VISIBLE_DEVICES=1,3`。
设置后容器内 `cuda:0` 会映射到可见集合的第 0 张卡，而不是宿主物理 GPU0。

## Hugging Face 与模型缓存

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `HF_TOKEN` | 空 | 访问 pyannote / WeSpeaker gated 模型所需 token。需要先在 Hugging Face 接受相关模型条款。 |
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face Hub 入口；受限网络可改为可信镜像。 |
| `HF_HUB_DISABLE_XET` | `1` | 默认绕开 hf-xet/CAS 下载路径；只有确认环境支持时才改为 `0`。 |
| `HF_HUB_ETAG_TIMEOUT` | `3` | Hub 元数据请求超时秒数，网络慢时更快回退到本地缓存。 |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` / `TORCH_HOME` / `XDG_CACHE_HOME` | `/cache` 相关路径 | Dockerfile 内置缓存路径。通常通过 `MODEL_CACHE_DIR` 挂载宿主缓存，不需要逐项覆盖。 |

faster-whisper 会优先查找 `/models/faster-whisper-<WHISPER_MODEL>`；不存在时再按
`WHISPER_MODEL` 名称走模型加载路径。pyannote 和 WeSpeaker 会先尝试完整的本地
Hugging Face snapshot，缓存不完整时再走 Hub。

## Whisper / ASR

| 配置 | 默认值 | 已支持情况 |
| --- | --- | --- |
| `WHISPER_MODEL` | `large-v3` | 服务级 env，支持 `tiny`、`base`、`small`、`medium`、`large-v3` 等 faster-whisper 模型名。 |
| `DEVICE` | `cuda` | 服务级 env；`cuda` / `cuda:<index>` 使用 `float16`，`cpu` 使用 `int8`，compute type 目前不可单独配置。 |
| API `language` | 自动检测 | 请求级字段；留空时会自动检测，并使用面向普通话的初始提示。 |
| API `no_repeat_ngram_size` | `0` | 请求级字段；`>=3` 时传给 faster-whisper 抑制 n-gram 重复，非整数返回 `422`。 |

当前内部 ASR 默认值：`beam_size=5`、`vad_filter=True`、
`vad_parameters.min_silence_duration_ms=500`、`condition_on_previous_text=False`。
这些值还没有对应 env 或 API 字段；不要写 `WHISPER_BEAM_SIZE`、
`WHISPER_COMPUTE_TYPE`、`WHISPER_VAD_*` 之类未实现配置。

## 降噪

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `DENOISE_MODEL` | `none` | 服务端默认降噪后端：`none`、`deepfilternet`、`noisereduce`。未知值会记录警告并跳过降噪。 |
| `DENOISE_SNR_THRESHOLD` | `10.0` | DeepFilterNet 的 SNR 门限 dB。选择 `deepfilternet` 时，估算 SNR 大于等于该值会跳过，避免处理干净录音；`noisereduce` 不受 SNR gate 控制，但仍受音频时长保护。 |
| `DENOISE_MAX_AUDIO_DURATION_SEC` | `7200` | 音频时长保护，不是执行超时。音频超过该秒数时跳过可选降噪，保留未降噪音频继续转写；时长元数据不可读或无效时也跳过，记录 `reason=audio_duration_unavailable`。设为 `0` 仅取消已知时长上限，不绕过未知时长保护。 |
| API `denoise_model` | 省略 | 省略表示继承 `DENOISE_MODEL`；显式传 `none` 表示只对本次任务关闭降噪。 |
| API `snr_threshold` | 省略 | 省略表示继承 `DENOISE_SNR_THRESHOLD`；显式传值只覆盖本次任务的 DeepFilterNet SNR gate。 |

默认面向干净会议录音，因此 `DENOISE_MODEL=none`。只有噪声环境才建议按任务
或服务级启用 `deepfilternet` / `noisereduce`。如需“干净录音自动跳过”，请选择
`deepfilternet`；两种降噪后端都受 `DENOISE_MAX_AUDIO_DURATION_SEC` 保护。

## Diarization 与 alignment

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| API `min_speakers` / `max_speakers` | `0` | 请求级说话人数约束；`0` 表示自动，不传入 pyannote。 |
| `PYANNOTE_MIN_DURATION_OFF` | `0.5` | pyannote `_binarize.min_duration_off`，用于合并短暂停顿、减少过度切分。若当前 pyannote 对象不支持该属性，服务会记录警告并继续运行。 |
| `WHISPERX_ALIGN_DISABLED_LANGUAGES` | 空 | 逗号分隔语言列表；命中且没有模型覆盖时跳过 forced alignment。只建议作为临时降级开关。 |
| `WHISPERX_ALIGN_DEVICE` | `cpu` | WhisperX forced alignment 的运行设备。默认 CPU，用于隔离 wav2vec2 alignment 与 GPU ASR / speaker embedding 运行时；确认 CUDA alignment 稳定后可设为 `pipeline` / `asr` / `cuda` / `cuda:0`。 |
| `WHISPERX_ALIGN_MODEL_MAP` | 空 | 逗号分隔 `lang=model` 覆盖，例如 `zh=org/model`。 |
| `WHISPERX_ALIGN_MODEL_DIR` | 空 | 可选 alignment 模型目录；仅在当前 WhisperX 版本支持该参数时透传。 |
| `WHISPERX_ALIGN_CACHE_ONLY` | `0` | 为 `1` 时，alignment 模型仅从本地缓存加载，禁止该加载路径联网；缓存不完整时以脱敏失败元数据降级，而不是下载模型。该设置不代表 ASR、diarization 和 embedding 全部离线。 |
| `WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC` | `7200` | 音频时长保护，不是执行超时。超过该秒数时在整段音频加载前跳过 forced alignment，保留转写结果；时长元数据不可读或无效时也记录 `alignment.status=skipped, reason=audio_duration_unavailable`。设为 `0` 仅取消已知时长上限，不绕过未知时长保护。 |

alignment 是可选元数据。成功时结果顶层可能包含 `alignment.status=succeeded`
和 `segments[].words`；被显式禁用或加载失败时任务仍会完成，`words` 可能缺失，
`alignment` 会以脱敏方式记录 `skipped` 或 `failed`。客户端必须把这两个字段视为
可选。

## Embedding

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EMBEDDING_DIM` | `256` | 声纹向量维度，用于声纹库和 AS-norm cohort 形状校验。不要把不同维度的既有声纹库混用。 |
| `MIN_EMBED_DURATION` | `1.5` | 短于该时长的 diarization turn 不参与 speaker embedding。 |
| `MAX_EMBED_DURATION` | `10.0` | 长于该时长的 turn 会截断到该窗口后再提取 embedding。 |
| `EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC` | `1800` | 全音频预载的时长保护，不是执行超时。已知音频时长不超过该秒数时可预载一次；更长或时长未知时按选中的 turn 分段读取，仍继续提取 embedding。设为 `0` 取消已知时长音频的预载上限。 |

先按元数据筛选并排序候选 turn，再为每个说话人 cluster 提取最多 10 个最长可用
片段的平均 embedding；不是先把全部 turn 解码到内存再截取 10 个。预载失败时也会
回退分段读取。太短、太碎或噪声很大的 turn 会降低登记与识别质量。

## 声纹与 AS-norm

| 项 | 默认值 | 说明 |
| --- | --- | --- |
| `VOICEPRINT_THRESHOLD` | `0.75` | raw cosine 模式基础阈值。实际阈值会按样本数和 `sample_spread` 自适应。 |
| raw 单样本放松 | `0.05` | 单样本说话人默认有效阈值约 `0.70`。内部默认，未暴露为 env。 |
| raw spread 放松 | `3.0 * sample_spread`，上限 `0.10` | 多样本但样本差异更大的说话人会适度放宽。内部默认。 |
| raw 绝对下限 | `0.60` | raw cosine 永不接受低于该值的自动命名。内部默认。 |
| AS-norm 激活条件 | `10` 条 cohort embedding | cohort 小于 10 时，`ASNormScorer.score()` 回退 raw cosine。内部默认。 |
| AS-norm base | `0.5` | cohort 充足后使用的 z-score 类操作点，不是 raw cosine。内部默认。 |
| AS-norm top-1/top-2 margin | `0.05` | normalized 第一名与第二名太近时保持未命名。内部默认。 |
| AS-norm cohort `top_n` | `200` | 统计 impostor 分布时使用的最近 cohort 数量，上限为 cohort 实际大小。内部默认。 |

`similarity` 的口径取决于 cohort 状态：

- cohort < 10 或 AS-norm 未初始化：`similarity` 是 raw cosine，通常落在 `[-1, 1]`。
- cohort >= 10：`similarity` 是 AS-norm normalized score，可大于 `1` 或为负。
- `speaker_id != null` 才表示通过当前模式下的有效阈值；不要把 `similarity`
  当成百分比。

cohort 生命周期：

- 启动时若存在 `data/transcriptions/asnorm_cohort.npy`，直接加载。
- 否则扫描持久化转写结果和 `emb_*.npy` 构建并保存 cohort。
- 每次 enroll / update 后，后台 `cohort-rebuild` 线程每 60 秒检查一次，在最近一次
  enroll 至少过去 30 秒后自动重建。
- v0.8.4 的后台自动重建会保护更大的已加载或已持久化 cohort：清空转写结果、
  只有少量 embedding，或源数量少于现有 cohort 时，不会自动缩小 cohort。
- `POST /api/voiceprints/rebuild-cohort` 是显式手动重建，仍按当前可用 embedding
  立即生成新 cohort。

## 结果契约

完成态转写结果的稳定锚点：

- `status`：持久化结果为 `completed`；任务状态接口还可能返回
  `queued`、`converting`、`denoising`、`transcribing`、`identifying`、`failed`。
- `segments[].speaker_label`：pyannote 原始 cluster 标签，是登记声纹和后续纠错的稳定 key。
- `segments[].speaker_name`：展示名；匹配失败时回退为 `speaker_label`，多个 cluster
  命中同名声纹时会自动加序号区分。
- `segments[].speaker_id`：匹配成功时为声纹 ID，否则为 `null`。
- `segments[].similarity`：说话人级匹配分数；raw cosine 或 AS-norm z-score，取决于 cohort。
- `segments[].words`：可选词级 alignment。
- 顶层 `alignment`：可选 forced-alignment 元数据，字段内容会脱敏。
- 同名展示名不会合并不同的 speaker cluster；`speaker_label` 始终独立保留。
- 顶层 `params`：记录本次任务实际使用的请求级与服务级处理参数，便于离线解释结果。
- 顶层 `artifacts`：可选 artifact manifest，只列出稳定 / 可选 / 实验 artifact 的
  文件名、角色、类别、媒体类型和 `speaker_label`；不暴露本地路径、主机、token 或
  调试信息。
- `speaker_map`：diarization cluster 到声纹匹配结果的映射；人工改单段说话人不会回写它。
- `unique_speakers`：按当前 segment 展示名去重后的列表。

新增字段按可选字段原则扩展；客户端应忽略不认识的字段，并容忍 `words` /
`alignment` / `artifacts` / `warning` 缺失。

## v0.8.4 验证口径

v0.8.4 已用 internal live validation 覆盖：可选 Rust kernel 基础能力、已选择的
声纹计分、结果后处理、artifact/status helper contract、Docker packaging smoke
以及 public release scan gate。公开文档只记录行为类别，不发布真实任务名、
样本名、job id、speaker id、主机、日志或路径。

## v0.7.6 验证口径

v0.7.6 已用 internal live validation 覆盖：GPU cleanup 期间 `/healthz` 仍保持可用、
WhisperX forced alignment 运行时隔离与模型复用、短单段 stock outro 幻觉过滤，以及
embedding 单次读取规范化 WAV 后按 diarization turn 切片的路径。公开文档只记录行为
类别，不发布真实任务名、样本名、job id、speaker id、主机、日志或路径。

## v0.7.4 验证口径

v0.7.4 已用内部 live validation 验证：清空持久化转写结果后，只要既有声纹库和已加载 /
已持久化的 AS-norm cohort 仍在，后台自动重建不会把较大的 cohort 缩小为空或小样本
cohort。新声纹的 enroll、cohort rebuild、probe 命中和 cleanup 入口也覆盖过；但当前
公开验证没有可信的 >=10 cohort 证据，因此只能说明声纹 API、cohort 刷新入口和
raw-cosine fallback 可用，不能声称 probe 已完整走 AS-norm scoring path。完整 AS-norm
验证需要 cohort >=10。公开文档只记录行为结论，不发布真实任务名、样本名、job id、
speaker id、主机或路径。

## 手动 Heavy CI

Heavy CI 在 PR 的 `opened`、`reopened`、`ready_for_review` 事件和 `main` push
时自动运行；后续 PR 更新必须显式手动触发。同仓 candidate 自验证时，选择
candidate 分支作为 workflow ref，并将输入 `ref` 设为其完整 40 位 HEAD SHA。
workflow SHA、run SHA 和输入 SHA 必须一致。Actions 原生 checks
（`resolve-source`、`rust-wheel`、`docker-packaging`）归属该 HEAD；candidate
不获得 `checks: write`，不产生可信 target proof，也不执行 target-check reporter。
分支更新后必须针对新 SHA 重新手动触发。

只有从 `main` 手动触发时，才允许测试其他精确 target SHA、生成可信 target proof
和 report checks。candidate run 不能通过发布 verifier；正式发布仍要求已合入
`main` 的 PR 源码，以及该合并 SHA 的全部 9 项 required checks，不能沿用合并前
candidate 的绿灯。

## 相关文档

- [快速上手](./quickstart.zh.md)
- [API 参考](./api.zh.md)
- [声纹调参参考](./voiceprint-tuning.zh.md)
- [更新日志](./changelog.zh.md)
