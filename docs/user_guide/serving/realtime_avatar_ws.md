# OmniRT Realtime Avatar WebSocket

OmniRT Native Realtime Avatar WebSocket is the long-term protocol for model-agnostic digital-human streaming. It keeps the efficient `AUDI` / `VIDX` binary framing from the FlashTalk-compatible path, but uses an OmniRT session control plane with `session_id`, `trace_id`, structured errors, and metrics.

## Endpoint

```text
WS /v1/avatar/realtime
GET /v1/audio2video/models
WS /v1/audio2video/flashtalk
WS /v1/audio2video/wav2lip
```

`/v1/audio2video/flashtalk` and `/v1/audio2video/wav2lip` are the public
FlashTalk-compatible streaming paths for OpenTalking. `/v1/avatar/flashtalk`
and `/v1/avatar/wav2lip` remain compatibility aliases. `/v1/avatar/realtime`
is the model-agnostic control-plane protocol.

## Session create

```json
{
  "type": "session.create",
  "model": "soulx-flashtalk-14b",
  "backend": "auto",
  "inputs": {
    "image_b64": "<base64 png/jpeg>",
    "prompt": "A person is talking naturally."
  },
  "config": {
    "preset": "realtime",
    "seed": 9999,
    "wav2lip_postprocess_mode": false,
    "mouth_metadata": {
      "source_image_hash": "<sha256>",
      "animation": {
        "mouth_center": [0.5, 0.56],
        "mouth_rx": 0.06,
        "mouth_ry": 0.02,
        "outer_lip": [[0.45, 0.55], [0.5, 0.53], [0.55, 0.55]]
      }
    }
  }
}
```

Response:

```json
{
  "type": "session.created",
  "session_id": "avt_...",
  "trace_id": "trace_...",
  "audio": {
    "format": "pcm_s16le",
    "sample_rate": 16000,
    "channels": 1,
    "chunk_samples": 17920
  },
  "video": {
    "encoding": "jpeg-seq",
    "wire_magic": "VIDX",
    "fps": 25,
    "width": 416,
    "height": 704
  }
}
```

## Wav2Lip postprocess mode

Wav2Lip sessions accept `wav2lip_postprocess_mode` and optional
`mouth_metadata` in session config. When disabled, OmniRT keeps native Wav2Lip
output behavior. When enabled, the Wav2Lip runtime can use the supplied mouth
polygon to blend the generated mouth region back into the reference frame with
lower-lip coverage, feathering, and color matching.

The service default is off. It can be enabled process-wide with:

```bash
OMNIRT_WAV2LIP_POSTPROCESS_MODE=1 omnirt serve ...
```

The enhanced path exposes separate knobs for lower-lip coverage and jaw motion
transfer:

```bash
OMNIRT_WAV2LIP_LOWER_LIP_DYNAMIC_EXPAND=0.25
OMNIRT_WAV2LIP_ENABLE_JAW_MOTION_BLEND=1
OMNIRT_WAV2LIP_JAW_BLEND_ALPHA=0.22
OMNIRT_WAV2LIP_JAW_MASK_EXPAND_X=0.25
OMNIRT_WAV2LIP_JAW_MASK_EXPAND_Y=0.55
```

Jaw motion blending is disabled by default so enhanced mouth blending and jaw
motion can be A/B tested independently.

OpenTalking-compatible clients may also send the same fields in the `init`
message to `/v1/audio2video/wav2lip`.

## Audio and video chunks

Send audio:

```text
b"AUDI" + pcm_s16le
```

The server sends a metrics event, then a video binary payload:

```json
{"type": "metrics", "chunk_index": 1, "infer_ms": 0, "encode_ms": 0}
```

```text
b"VIDX" + uint32(frame_count) + repeated(uint32(jpeg_len) + jpeg_bytes)
```

## QuickTalk 性能诊断日志

benchmark / debug 时，在运行 QuickTalk runtime 的服务进程中设置 `OMNIRT_PERF_LOG=1`，再按原有方式启动服务。未设置或设为 `0` 时关闭详细性能日志，不逐 chunk 输出；开启后快速 chunk 和空帧预热 chunk 也会记录，不受原先 200 ms 慢 chunk 门槛限制。

`quicktalk_ws_chunk` 覆盖 `/v1/audio2video/quicktalk`、其 `/v1/avatar/quicktalk` 别名，以及 `/v1/avatar/realtime` 中的 QuickTalk session。每次 VIDX 的发送 await 成功返回后输出一行，采用空格分隔的英文 `key=value` 格式：

| 指标 | 中文含义与计量范围 |
|---|---|
| `session_id` | 会话标识 |
| `chunk_index` | 分块序号，直接使用 `service.push_audio_chunk()` 返回值，从 1 开始 |
| `inter_chunk_gap_ms` | 分块间隔，上一次 VIDX 的 `send_bytes` await 成功返回，到当前 AUDI 的 `websocket.receive()` 返回时立即采样的时间差；首个 chunk 为 `null` |
| `lock_wait_ms` | 锁等待耗时，从准备获取全局 `avatar_runtime_lock` 到实际持有锁 |
| `infer_ms` | 推理耗时，直接使用服务返回的 `metrics["infer_ms"]` |
| `payload_bytes` | 视频载荷字节数，等于 `len(video_payload)`，包含 VIDX 头及帧长度字段 |
| `ws_send_ms` | ASGI/WebSocket send await 耗时，仅计量 `await websocket.send_bytes(video_payload)` |
| `server_total_ms` | 服务端总耗时，从准备处理音频 chunk 到二进制发送 await 返回；包含锁等待、线程调度、推理和发送，原生路由还包含已有 metrics JSON 的发送 |

所有 `*_ms` 均以毫秒计。`ws_send_ms` 是 ASGI/WebSocket send await duration，可能反映 socket 背压，但不代表远端应用已完整收到 payload，更不是公网完整传输耗时。`server_total_ms` 不包含等待接收客户端音频的时间，也不包含 `inter_chunk_gap_ms`。

`inter_chunk_gap_ms` 使用 `time.perf_counter()` 计时，接收端点是 ASGI 应用收到完整 message 的时刻，不是网卡收到数据包的时刻。如果 AUDI 已在 ASGI 层排队，下一次 `receive()` 可能立即返回，间隔接近 0。该间隔可能包含期间的客户端发送节奏、网络、下游消费，以及服务端日志输出和事件循环调度；不能仅凭这个值区分根因，也不能将其当作纯网络 RTT。当前 chunk 的锁等待、推理和发送均不在这个间隔内。

间隔仅在 QuickTalk 且 `OMNIRT_PERF_LOG=1` 时跟踪，由每个 WebSocket loop 为当前 session 独立维护。`init` / `session.create`、`close` / `session.close` 和原生路由的 `session.cancel` 会重置记录；断开连接后局部状态随 loop 退出清理。其他协议消息不更新上一次 VIDX 的发送完成时间。

以下两行数值仅为示意；首个 chunk 无上一次成功发送记录，第二个 chunk 的分块间隔为 250 ms：

```text
quicktalk_ws_chunk session_id=example-session chunk_index=1 inter_chunk_gap_ms=null lock_wait_ms=1.000 infer_ms=80.0 payload_bytes=192000 ws_send_ms=8.000 server_total_ms=90.000
quicktalk_ws_chunk session_id=example-session chunk_index=2 inter_chunk_gap_ms=250.000 lock_wait_ms=0.060 infer_ms=82.0 payload_bytes=198400 ws_send_ms=8.000 server_total_ms=90.560
```

同一开关还启用 runtime 的 `quicktalk_render_chunk` 日志：`feature_ms`（特征提取）、`generate_ms`（视频帧生成）、`encode_ms`（JPEG 编码）、`total_ms`（渲染总耗时）及 `frames`（输出帧数）。保留的 `session` 对应服务层 `session_id`；`chunk` 是渲染时已完成的分块数，正常音频 chunk 对应服务层 `chunk_index - 1`。初始化预热也可能产生 render 日志，但不对应 WebSocket 发送。

这些日志仅在服务端输出，不添加客户端消息或 metrics 字段，VIDX 二进制内容、原有消息顺序和 session 行为保持不变。代理转发入口不会在本进程执行推理，需要在实际承载 QuickTalk runtime 的服务中启用开关。

## Control messages

```json
{"type": "session.cancel"}
{"type": "session.close"}
{"type": "ping"}
```

## Runtime 模式

v1 endpoint 保持 wire contract 稳定，不同部署可以选择不同 runtime：

| 模式 | 选择方式 | 说明 |
|---|---|---|
| `fake` | 默认，或 `OMNIRT_REALTIME_AVATAR_RUNTIME=fake` | 为协议测试和 CPU-stub demo 输出确定性 JPEG chunk |
| `proxy` | `OMNIRT_REALTIME_AVATAR_RUNTIME=proxy` + `OMNIRT_AVATAR_FLASHTALK_WS_URL` | 把 FlashTalk-compatible 路由转发到已有 WebSocket 服务 |
| `resident` | `OMNIRT_REALTIME_AVATAR_RUNTIME=resident` | 通过 OmniRT resident `soulx-flashtalk-14b` 执行路径渲染 chunk |

`GET /v1/audio2video/models` 会返回 `fallback_runtime`、`proxy` 或 `resident_runtime`，客户端可以据此区分协议测试模式和真实模型后端。
