from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi import FastAPI, WebSocketDisconnect  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from omnirt.server.realtime_avatar import (  # noqa: E402
    MAGIC_AUDIO,
    RealtimeAvatarService,
    encode_jpeg_sequence,
)
from omnirt.server.routes import avatar  # noqa: E402


_PERF_KEYS = {
    "session_id",
    "chunk_index",
    "lock_wait_ms",
    "infer_ms",
    "payload_bytes",
    "ws_send_ms",
    "server_total_ms",
}
_VIDEO_PAYLOADS = [
    encode_jpeg_sequence([]),
    encode_jpeg_sequence([b"first-jpeg"]),
    encode_jpeg_sequence([b"second-jpeg", b"third-jpeg"]),
]


class _RecordingService(RealtimeAvatarService):
    def __init__(self) -> None:
        runtime = SimpleNamespace(
            render_chunk=lambda session, pcm: _VIDEO_PAYLOADS[session.chunk_index],
        )
        super().__init__(runtime=runtime)
        self.returned: list[tuple[bytes, dict[str, object]]] = []

    def push_audio_chunk(self, session_id, payload):
        video, metrics = super().push_audio_chunk(session_id, payload)
        self.returned.append((video, dict(metrics)))
        return video, metrics


def _app(service: RealtimeAvatarService) -> FastAPI:
    app = FastAPI()
    app.state.realtime_avatar_service = service
    app.state.default_backend = "cpu-stub"
    app.state.default_request_config = {"chunk_samples": 4, "width": 32, "height": 32}
    app.include_router(avatar.router)
    return app


def _start_session(ws, *, native: bool, model: str) -> None:
    image_b64 = base64.b64encode(b"reference-image").decode("ascii")
    if native:
        ws.send_json(
            {
                "type": "session.create",
                "model": model,
                "inputs": {"image_b64": image_b64},
            }
        )
        assert ws.receive_json()["type"] == "session.created"
    else:
        ws.send_json({"type": "init", "ref_image": image_b64})
        assert ws.receive_json()["type"] == "init_ok"


def _perf_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("quicktalk_ws_chunk ")]


def _perf_fields(line: str) -> dict[str, str]:
    assert line.isascii()
    prefix, *items = line.split(" ")
    assert prefix == "quicktalk_ws_chunk"
    fields = dict(item.split("=", 1) for item in items)
    assert set(fields) == _PERF_KEYS
    return fields


@pytest.mark.parametrize("native", [False, True], ids=["compatible", "native"])
@pytest.mark.parametrize("perf_log", [None, "0", "1"], ids=["unset", "disabled", "enabled"])
def test_quicktalk_perf_logging_preserves_every_chunk_on_wire(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    native: bool,
    perf_log: str | None,
) -> None:
    if perf_log is None:
        monkeypatch.delenv("OMNIRT_PERF_LOG", raising=False)
    else:
        monkeypatch.setenv("OMNIRT_PERF_LOG", perf_log)
    # All server durations are zero: enabled logging must include fast chunks too.
    monkeypatch.setattr(avatar, "time", SimpleNamespace(perf_counter=lambda: 10.0))
    service = _RecordingService()
    app = _app(service)
    path = "/v1/avatar/realtime" if native else "/v1/audio2video/quicktalk"

    with TestClient(app).websocket_connect(path) as ws:
        _start_session(ws, native=native, model="quicktalk")
        session_id = next(iter(service._sessions))
        for expected in _VIDEO_PAYLOADS:
            ws.send_bytes(MAGIC_AUDIO + b"\0\0" * 4)
            if native:
                # The existing metrics message precedes the binary, with its exact fields.
                metrics = ws.receive_json()
                assert metrics == service.returned[-1][1]
                assert set(metrics) == {"type", "chunk_index", "infer_ms", "encode_ms"}
            message = ws.receive()
            assert message == {"type": "websocket.send", "bytes": expected}

        ws.send_json({"type": "session.close" if native else "close"})
        closed = ws.receive_json()
        assert closed["type"] == ("session.closed" if native else "close_ok")
        assert service._sessions == {}

    assert [video for video, _ in service.returned] == _VIDEO_PAYLOADS
    assert [metrics["chunk_index"] for _, metrics in service.returned] == [1, 2, 3]
    lines = _perf_lines(capsys.readouterr().out)
    if perf_log != "1":
        assert lines == []
        return
    assert len(lines) == len(_VIDEO_PAYLOADS)
    for line, (video, metrics) in zip(lines, service.returned):
        fields = _perf_fields(line)
        assert fields == {
            "session_id": session_id,
            "chunk_index": str(metrics["chunk_index"]),
            "infer_ms": str(metrics["infer_ms"]),
            "payload_bytes": str(len(video)),
            "lock_wait_ms": "0.000",
            "ws_send_ms": "0.000",
            "server_total_ms": "0.000",
        }


@pytest.mark.parametrize("native", [False, True], ids=["compatible", "native"])
def test_perf_logging_does_not_enable_chunk_logs_for_other_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    native: bool,
) -> None:
    monkeypatch.setenv("OMNIRT_PERF_LOG", "1")
    service = _RecordingService()
    path = "/v1/avatar/realtime" if native else "/v1/audio2video/wav2lip"

    with TestClient(_app(service)).websocket_connect(path) as ws:
        _start_session(ws, native=native, model="wav2lip")
        ws.send_bytes(MAGIC_AUDIO + b"\0\0" * 4)
        if native:
            assert ws.receive_json() == service.returned[-1][1]
        assert ws.receive_bytes() == _VIDEO_PAYLOADS[0]

    assert _perf_lines(capsys.readouterr().out) == []


def test_perf_timings_measure_contended_lock_and_awaited_send_separately(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run() -> None:
        clock = SimpleNamespace(now=10.0)
        monkeypatch.setattr(avatar, "time", SimpleNamespace(perf_counter=lambda: clock.now))
        lock_attempted = asyncio.Event()
        send_entered = asyncio.Event()
        send_finished = asyncio.Event()

        class ObservedLock(asyncio.Lock):
            async def acquire(self):
                if self.locked():
                    lock_attempted.set()
                return await super().acquire()

        lock = ObservedLock()
        await lock.acquire()
        video = _VIDEO_PAYLOADS[1]
        metrics = {"type": "metrics", "chunk_index": 27, "infer_ms": 6.125, "encode_ms": 0}
        original_metrics = dict(metrics)
        service_result = (video, metrics)
        sent: list[bytes] = []

        def push_audio_chunk(session_id, payload):
            assert lock.locked()
            assert session_id == "contended-session"
            assert payload == b"audio"
            clock.now = 10.0725
            return service_result

        async def send_bytes(payload):
            assert not lock.locked()
            async with lock:
                sent.append(payload)
            send_entered.set()
            await send_finished.wait()

        websocket = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(avatar_runtime_lock=lock)),
            send_bytes=send_bytes,
        )
        performance = avatar._AudioChunkPerformance()
        clock.now = 10.005
        push = asyncio.create_task(
            avatar._push_audio_chunk_async(
                websocket,
                SimpleNamespace(push_audio_chunk=push_audio_chunk),
                "contended-session",
                b"audio",
                performance=performance,
            )
        )
        await asyncio.wait_for(lock_attempted.wait(), timeout=5)
        assert not push.done()
        clock.now = 10.055
        lock.release()
        result = await asyncio.wait_for(push, timeout=5)
        assert result is service_result
        assert performance.lock_wait_ms == pytest.approx(50.0)
        assert not lock.locked()

        send = asyncio.create_task(
            avatar._send_audio_chunk_async(websocket, "contended-session", *result, performance)
        )
        await asyncio.wait_for(send_entered.wait(), timeout=5)
        assert not send.done()
        clock.now = 10.0835
        send_finished.set()
        await asyncio.wait_for(send, timeout=5)
        assert sent == [video]
        assert metrics == original_metrics

    asyncio.run(run())
    lines = _perf_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert _perf_fields(lines[0]) == {
        "session_id": "contended-session",
        "chunk_index": "27",
        "lock_wait_ms": "50.000",
        "infer_ms": "6.125",
        "payload_bytes": str(len(_VIDEO_PAYLOADS[1])),
        "ws_send_ms": "11.000",
        "server_total_ms": "83.500",
    }


@pytest.mark.parametrize("native", [False, True], ids=["compatible", "native"])
def test_bad_audio_keeps_error_response_and_session_usable_with_perf_logging(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    native: bool,
) -> None:
    monkeypatch.setenv("OMNIRT_PERF_LOG", "1")
    service = _RecordingService()
    path = "/v1/avatar/realtime" if native else "/v1/audio2video/quicktalk"

    with TestClient(_app(service)).websocket_connect(path) as ws:
        _start_session(ws, native=native, model="quicktalk")
        ws.send_bytes(b"INVALID" + b"\0\0" * 4)
        error = ws.receive_json()
        assert error == {
            "type": "error",
            "code": "bad_audio_magic",
            "message": "Binary audio payload must start with AUDI magic.",
        }
        ws.send_bytes(MAGIC_AUDIO + b"\0\0" * 4)
        if native:
            assert ws.receive_json() == service.returned[-1][1]
        assert ws.receive_bytes() == _VIDEO_PAYLOADS[0]

    assert len(service.returned) == 1
    assert len(_perf_lines(capsys.readouterr().out)) == 1


def test_send_disconnect_propagates_without_logging_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def send_bytes(payload):
        assert payload == _VIDEO_PAYLOADS[0]
        raise WebSocketDisconnect(code=1001)

    with pytest.raises(WebSocketDisconnect) as error:
        asyncio.run(
            avatar._send_audio_chunk_async(
                SimpleNamespace(send_bytes=send_bytes),
                "disconnected-session",
                _VIDEO_PAYLOADS[0],
                {"chunk_index": 1, "infer_ms": 2.5},
                avatar._AudioChunkPerformance(),
            )
        )

    assert error.value.code == 1001
    assert _perf_lines(capsys.readouterr().out) == []
