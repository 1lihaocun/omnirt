from __future__ import annotations

import asyncio
import base64
import json
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
    "inter_chunk_gap_ms",
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
            "inter_chunk_gap_ms": "null" if metrics["chunk_index"] == 1 else "0.000",
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
        performance = avatar._AudioChunkPerformance(inter_chunk_gap_ms=125.0)
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
        assert await asyncio.wait_for(send, timeout=5) == 10.0835
        assert sent == [video]
        assert metrics == original_metrics

    asyncio.run(run())
    lines = _perf_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert _perf_fields(lines[0]) == {
        "session_id": "contended-session",
        "chunk_index": "27",
        "inter_chunk_gap_ms": "125.000",
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


def test_send_without_perf_logging_returns_no_timestamp(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[bytes] = []

    async def send_bytes(payload):
        sent.append(payload)

    finished_at = asyncio.run(
        avatar._send_audio_chunk_async(
            SimpleNamespace(send_bytes=send_bytes), "quiet-session", _VIDEO_PAYLOADS[0], {}, None
        )
    )

    assert finished_at is None
    assert sent == [_VIDEO_PAYLOADS[0]]
    assert _perf_lines(capsys.readouterr().out) == []


class _ClockedWebSocket:
    """Drive the real route loop at explicit receipt times, without wall-clock sleeps."""

    def __init__(self, app: FastAPI, clock: SimpleNamespace, *, native: bool) -> None:
        self.app = app
        self.clock = clock
        self.native = native
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.outgoing: asyncio.Queue = asyncio.Queue()

    async def accept(self):
        pass

    async def receive(self):
        received_at, message = await self.incoming.get()
        self.clock.now = received_at
        if isinstance(message, WebSocketDisconnect):
            raise message
        return message

    async def send_json(self, payload):
        self.outgoing.put_nowait(("json", dict(payload)))

    async def send_bytes(self, payload):
        assert not self.app.state.avatar_runtime_lock.locked()
        self.clock.now += 0.010
        self.outgoing.put_nowait(("bytes", payload))

    def run(self):
        if self.native:
            return asyncio.create_task(avatar.native_realtime_avatar(self))
        return asyncio.create_task(avatar._flashtalk_compatible_loop(self, model="quicktalk"))

    async def response(self):
        return await asyncio.wait_for(self.outgoing.get(), timeout=5)

    async def control(self, received_at: float, payload: dict[str, object]):
        self.incoming.put_nowait((received_at, {"text": json.dumps(payload)}))
        kind, response = await self.response()
        assert kind == "json"
        return response

    async def start_session(self, received_at: float) -> str:
        service = self.app.state.realtime_avatar_service
        before = set(service._sessions)
        image_b64 = base64.b64encode(b"reference-image").decode("ascii")
        payload = (
            {
                "type": "session.create",
                "model": "quicktalk",
                "inputs": {"image_b64": image_b64},
            }
            if self.native
            else {"type": "init", "ref_image": image_b64}
        )
        response = await self.control(received_at, payload)
        assert response["type"] == ("session.created" if self.native else "init_ok")
        created = set(service._sessions) - before
        assert len(created) == 1
        return created.pop()

    async def push_audio(self, received_at: float) -> bytes:
        self.incoming.put_nowait((received_at, {"bytes": MAGIC_AUDIO + b"\0\0" * 4}))
        if self.native:
            kind, metrics = await self.response()
            assert kind == "json"
            assert metrics == self.app.state.realtime_avatar_service.returned[-1][1]
            assert set(metrics) == {"type", "chunk_index", "infer_ms", "encode_ms"}
        kind, payload = await self.response()
        assert kind == "bytes"
        return payload

    def disconnect(self, received_at: float, *, raises: bool = False) -> None:
        message = WebSocketDisconnect(1001) if raises else {"type": "websocket.disconnect"}
        self.incoming.put_nowait((received_at, message))


def _clocked_app(clock: SimpleNamespace) -> FastAPI:
    service = _RecordingService()
    render_chunk = service.runtime.render_chunk

    def timed_render(session, pcm):
        clock.now += 0.020
        return render_chunk(session, pcm)

    service.runtime.render_chunk = timed_render
    return _app(service)


@pytest.mark.parametrize("native", [False, True], ids=["compatible", "native"])
def test_inter_chunk_gap_is_connection_local_and_excludes_current_processing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    native: bool,
) -> None:
    async def run() -> tuple[str, str]:
        clock = SimpleNamespace(now=0.0)
        monkeypatch.setenv("OMNIRT_PERF_LOG", "1")
        monkeypatch.setattr(avatar, "time", SimpleNamespace(perf_counter=lambda: clock.now))
        app = _clocked_app(clock)
        lock_attempted = asyncio.Event()

        class ObservedLock(asyncio.Lock):
            async def acquire(self):
                if self.locked():
                    lock_attempted.set()
                return await super().acquire()

        lock = ObservedLock()
        app.state.avatar_runtime_lock = lock
        first = _ClockedWebSocket(app, clock, native=native)
        second = _ClockedWebSocket(app, clock, native=native)
        first_loop, second_loop = first.run(), second.run()
        first_id = await first.start_session(9.0)
        second_id = await second.start_session(9.1)
        assert first_id != second_id

        assert await first.push_audio(10.0) == _VIDEO_PAYLOADS[0]  # Send finishes at 10.030.
        assert await second.push_audio(10.1) == _VIDEO_PAYLOADS[0]  # Send finishes at 10.130.
        await lock.acquire()
        next_first = asyncio.create_task(first.push_audio(10.25))
        await asyncio.wait_for(lock_attempted.wait(), timeout=5)
        clock.now = 10.29
        lock.release()
        assert await asyncio.wait_for(next_first, timeout=5) == _VIDEO_PAYLOADS[1]
        assert await second.push_audio(10.5) == _VIDEO_PAYLOADS[1]
        first.disconnect(10.6)
        await asyncio.wait_for(first_loop, timeout=5)
        second.disconnect(10.7)
        await asyncio.wait_for(second_loop, timeout=5)
        assert app.state.realtime_avatar_service._sessions == {}
        return first_id, second_id

    first_id, second_id = asyncio.run(run())
    fields = [_perf_fields(line) for line in _perf_lines(capsys.readouterr().out)]
    assert [item["session_id"] for item in fields] == [first_id, second_id, first_id, second_id]
    assert [item["chunk_index"] for item in fields] == ["1", "1", "2", "2"]
    assert [item["inter_chunk_gap_ms"] for item in fields] == ["null", "null", "220.000", "370.000"]
    # The second chunk waits 40 ms, renders for 20 ms and sends for 10 ms;
    # none of those durations may be added to its 220 ms inter-chunk gap.
    assert fields[2]["lock_wait_ms"] == "40.000"
    assert fields[2]["ws_send_ms"] == "10.000"
    assert fields[2]["server_total_ms"] == "70.000"


@pytest.mark.parametrize("native", [False, True], ids=["compatible", "native"])
@pytest.mark.parametrize("raises_disconnect", [False, True], ids=["disconnect", "disconnect-exception"])
def test_inter_chunk_gap_resets_when_sessions_are_recreated_or_reconnected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    native: bool,
    raises_disconnect: bool,
) -> None:
    async def run() -> list[str]:
        clock = SimpleNamespace(now=0.0)
        monkeypatch.setenv("OMNIRT_PERF_LOG", "1")
        monkeypatch.setattr(avatar, "time", SimpleNamespace(perf_counter=lambda: clock.now))
        app = _clocked_app(clock)
        service = app.state.realtime_avatar_service
        websocket = _ClockedWebSocket(app, clock, native=native)
        loop = websocket.run()
        first_id = await websocket.start_session(1.0)
        assert await websocket.push_audio(1.1) == _VIDEO_PAYLOADS[0]
        assert await websocket.push_audio(1.2) == _VIDEO_PAYLOADS[1]
        session_ids = [first_id, first_id]

        reinitialized_id = await websocket.start_session(1.3)
        assert first_id not in service._sessions
        assert await websocket.push_audio(1.4) == _VIDEO_PAYLOADS[0]
        session_ids.append(reinitialized_id)
        closed = await websocket.control(1.5, {"type": "session.close" if native else "close"})
        assert closed["type"] == ("session.closed" if native else "close_ok")
        assert service._sessions == {}
        reopened_id = await websocket.start_session(1.6)
        assert await websocket.push_audio(1.7) == _VIDEO_PAYLOADS[0]
        session_ids.append(reopened_id)

        if native:
            cancelled = await websocket.control(1.8, {"type": "session.cancel"})
            assert cancelled == {"type": "session.cancelled", "session_id": reopened_id}
            websocket.incoming.put_nowait((1.9, {"bytes": MAGIC_AUDIO + b"\0\0" * 4}))
            kind, error = await websocket.response()
            assert kind == "json"
            assert error["type"] == "error"
            assert error["code"] == "session_cancelled"
            after_cancel_id = await websocket.start_session(2.0)
            assert await websocket.push_audio(2.1) == _VIDEO_PAYLOADS[0]
            session_ids.append(after_cancel_id)

        websocket.disconnect(2.2, raises=raises_disconnect)
        await asyncio.wait_for(loop, timeout=5)
        assert service._sessions == {}
        reconnected = _ClockedWebSocket(app, clock, native=native)
        reconnected_loop = reconnected.run()
        reconnected_id = await reconnected.start_session(2.3)
        assert await reconnected.push_audio(2.4) == _VIDEO_PAYLOADS[0]
        session_ids.append(reconnected_id)
        reconnected.disconnect(2.5)
        await asyncio.wait_for(reconnected_loop, timeout=5)
        assert service._sessions == {}
        return session_ids

    session_ids = asyncio.run(run())
    fields = [_perf_fields(line) for line in _perf_lines(capsys.readouterr().out)]
    assert [item["session_id"] for item in fields] == session_ids
    assert [item["inter_chunk_gap_ms"] for item in fields] == [
        "null", "70.000", *(["null"] * (len(session_ids) - 2))
    ]
