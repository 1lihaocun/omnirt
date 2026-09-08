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
    "seed": 9999
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

## QuickTalk performance diagnostics

For benchmarks or debugging, set `OMNIRT_PERF_LOG=1` in the process hosting the QuickTalk runtime before starting the service as usual. Leaving it unset or setting it to `0` disables detailed per-chunk performance logs. When enabled, fast chunks and empty priming chunks are logged too, without the previous 200 ms slow-chunk threshold.

`quicktalk_ws_chunk` covers `/v1/audio2video/quicktalk`, its `/v1/avatar/quicktalk` alias, and QuickTalk sessions on `/v1/avatar/realtime`. It emits one line after each VIDX send await returns successfully, using space-separated English `key=value` fields:

| Metric | Meaning and measurement boundary |
|---|---|
| `session_id` | Session identifier |
| `chunk_index` | Chunk number returned by `service.push_audio_chunk()`, starting at 1 |
| `inter_chunk_gap_ms` | Time from the previous successful VIDX `send_bytes` await return to the timestamp captured immediately after `websocket.receive()` returns the current AUDI message; `null` for the first chunk |
| `lock_wait_ms` | Time from attempting to acquire the global `avatar_runtime_lock` until holding it |
| `infer_ms` | Inference duration reused directly from the service's `metrics["infer_ms"]` |
| `payload_bytes` | `len(video_payload)`, including the VIDX header and frame-length fields |
| `ws_send_ms` | ASGI/WebSocket send await duration, measuring only `await websocket.send_bytes(video_payload)` |
| `server_total_ms` | Time from preparing to process the audio chunk until the binary send await returns; includes lock wait, thread scheduling, inference, and sending, plus the existing metrics JSON send on the native route |

All `*_ms` fields use milliseconds. `ws_send_ms` is an ASGI/WebSocket send await duration. It may reflect socket backpressure, but does not mean the remote application has received the complete payload and does not measure full public-network transfer time. `server_total_ms` excludes waiting to receive client audio and excludes `inter_chunk_gap_ms`.

`inter_chunk_gap_ms` uses `time.perf_counter()`. Its receive boundary is when the ASGI application receives a complete message, not when a packet arrives at the network interface. If AUDI is already queued in ASGI, the next `receive()` may return immediately and the gap may be near zero. The gap can include client pacing, network delays, downstream consumption, server log output, and event-loop scheduling during that interval. This value alone cannot distinguish those causes and is not a pure network RTT measurement. The current chunk's lock wait, inference, and send are outside this interval.

The gap is tracked only for QuickTalk with `OMNIRT_PERF_LOG=1`, using local state in each WebSocket loop for its current session. `init` / `session.create`, `close` / `session.close`, and native `session.cancel` reset this state; disconnecting discards the local state when the loop exits. Other protocol messages do not update the previous VIDX send completion time.

The values below are illustrative. The first chunk has no previous successful send, and the second chunk has a 250 ms gap:

```text
quicktalk_ws_chunk session_id=example-session chunk_index=1 inter_chunk_gap_ms=null lock_wait_ms=1.000 infer_ms=80.0 payload_bytes=192000 ws_send_ms=8.000 server_total_ms=90.000
quicktalk_ws_chunk session_id=example-session chunk_index=2 inter_chunk_gap_ms=250.000 lock_wait_ms=0.060 infer_ms=82.0 payload_bytes=198400 ws_send_ms=8.000 server_total_ms=90.560
```

The same switch enables runtime `quicktalk_render_chunk` logs with `feature_ms` (feature extraction), `generate_ms` (video-frame generation), `encode_ms` (JPEG encoding), `total_ms` (total rendering duration), and `frames` (output frame count). The retained `session` field matches the server's `session_id`; `chunk` is the count of completed chunks at render time, corresponding to `chunk_index - 1` for normal audio chunks. Initialization warmup may also emit render logs without a WebSocket send.

These logs stay on the server. They add no client messages or metrics fields and preserve VIDX bytes, message order, and session behavior. Proxy forwarding does not run inference locally; enable the switch in the service actually hosting the QuickTalk runtime.

## Control messages

```json
{"type": "session.cancel"}
{"type": "session.close"}
{"type": "ping"}
```

## Runtime modes

The v1 endpoint keeps the wire contract stable while the backing runtime can be selected per deployment:

| Mode | Selection | Notes |
|---|---|---|
| `fake` | default, or `OMNIRT_REALTIME_AVATAR_RUNTIME=fake` | deterministic JPEG chunks for protocol tests and CPU-stub demos |
| `proxy` | `OMNIRT_REALTIME_AVATAR_RUNTIME=proxy` plus `OMNIRT_AVATAR_FLASHTALK_WS_URL` | forwards the FlashTalk-compatible route to an existing WebSocket service |
| `resident` | `OMNIRT_REALTIME_AVATAR_RUNTIME=resident` | renders chunks through OmniRT's resident `soulx-flashtalk-14b` execution path |

`GET /v1/audio2video/models` reports the active reason as `fallback_runtime`, `proxy`, or `resident_runtime` so clients can distinguish protocol-only test mode from a model-backed deployment.
