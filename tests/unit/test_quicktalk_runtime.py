from __future__ import annotations

import itertools
import threading
from types import MethodType

import numpy as np
import pytest


def test_quicktalk_auto_device_prefers_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    monkeypatch.setattr(
        quicktalk_runtime,
        "_is_accelerator_available",
        lambda kind: kind == "npu",
    )

    assert quicktalk_runtime.resolve_quicktalk_device("auto") == "npu:0"


def test_quicktalk_auto_device_prefers_cuda_when_npu_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    monkeypatch.setattr(
        quicktalk_runtime,
        "_is_accelerator_available",
        lambda kind: kind == "cuda",
    )

    assert quicktalk_runtime.resolve_quicktalk_device("auto") == "cuda:0"


def test_quicktalk_streaming_pcm_features_skip_compressed_audio_cache(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.models.quicktalk.runtime_worker import RealtimeV3SessionState, RealtimeV3Worker

    class FakeQuickTalkV2:
        face_cache_dir = tmp_path
        sync_offset = 0

        def extract_representations_pcm(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
            return np.zeros((1, 10, 1024), dtype=np.float32)

        def build_rep_chunks(self, repst: np.ndarray, n_frames: int, fps: float) -> list[np.ndarray]:
            return [np.full((10, 1024), index, dtype=np.float32) for index in range(n_frames)]

    monkeypatch.setenv("OMNIRT_QUICKTALK_STREAMING_LOOKAHEAD_CHUNKS", "0")
    worker = object.__new__(RealtimeV3Worker)
    worker.v2 = FakeQuickTalkV2()
    worker.fps = 25.0
    worker._hubert_cache_identity = MethodType(
        lambda self: {"path": "fake-hubert", "files": [], "device_type": "cpu"},
        worker,
    )

    pcm = np.zeros(2560, dtype=np.int16)
    reps, _elapsed = worker.prepare_streaming_pcm_features(
        pcm,
        16_000,
        state=RealtimeV3SessionState(),
    )

    assert len(reps) == 4
    assert list(tmp_path.glob("audio_pcm_*.npz")) == []


@pytest.fixture
def quicktalk_render_case(monkeypatch: pytest.MonkeyPatch):
    from omnirt.models.quicktalk.runtime import QuickTalkRealtimeRuntime
    from omnirt.server.realtime_avatar import RealtimeAvatarSession

    state = object()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeWorker:
        def make_state(self):
            return state

        def prepare_streaming_pcm_features(self, pcm, sample_rate, *, state):
            assert pcm.tolist() == [1, 2, 3, 4]
            assert sample_rate == 16_000
            return ["rep"], 0.0

        def generate_frames_from_reps(self, reps, *, state):
            assert reps == ["rep"]
            return [frame]

    worker = FakeWorker()
    runtime = object.__new__(QuickTalkRealtimeRuntime)
    runtime._worker_lock = threading.RLock()
    runtime._states = {}
    monkeypatch.setattr(runtime, "_worker_for", lambda session: worker)
    monkeypatch.setattr(runtime, "_encode_jpeg_bgr", lambda frame: b"\xff\xd8\xff\xd9")
    session = RealtimeAvatarSession(
        session_id="perf-session",
        trace_id="perf-trace",
        model="quicktalk",
        backend="cpu",
        prompt="",
    )
    pcm = np.asarray([1, 2, 3, 4], dtype=np.int16).tobytes()
    # One four-byte JPEG in the existing little-endian VIDX binary framing.
    payload = b"VIDX\x01\x00\x00\x00\x04\x00\x00\x00\xff\xd8\xff\xd9"
    return runtime, session, pcm, payload, state


@pytest.mark.parametrize("perf_log", [None, "0", "true"])
@pytest.mark.parametrize("chunk_index,tick_seconds", [(0, 0.001), (3, 0.05)])
def test_quicktalk_perf_log_disabled_is_quiet_for_first_and_slow_chunks(
    quicktalk_render_case,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    perf_log: str | None,
    chunk_index: int,
    tick_seconds: float,
) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    runtime, session, pcm, expected_payload, state = quicktalk_render_case
    if perf_log is None:
        monkeypatch.delenv("OMNIRT_PERF_LOG", raising=False)
    else:
        monkeypatch.setenv("OMNIRT_PERF_LOG", perf_log)
    ticks = itertools.count(step=tick_seconds)
    monkeypatch.setattr(quicktalk_runtime.time, "perf_counter", lambda: next(ticks))
    session.chunk_index = chunk_index

    assert runtime.render_chunk(session, pcm) == expected_payload
    assert session.chunk_index == chunk_index
    assert runtime._states[session.session_id] is state
    assert capsys.readouterr().out == ""


def test_quicktalk_perf_log_records_every_fast_chunk_without_changing_vidx_payload(
    quicktalk_render_case,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    runtime, session, pcm, expected_payload, state = quicktalk_render_case
    ticks = itertools.count(step=0.001)
    monkeypatch.setattr(quicktalk_runtime.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setenv("OMNIRT_PERF_LOG", "0")
    baseline_payload = runtime.render_chunk(session, pcm)
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("OMNIRT_PERF_LOG", "1")

    for chunk_index in (1, 2):
        session.chunk_index = chunk_index
        assert runtime.render_chunk(session, pcm) == baseline_payload == expected_payload
        assert session.chunk_index == chunk_index
        assert runtime._states[session.session_id] is state

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    for chunk_index, line in zip((1, 2), lines):
        assert line.isascii()
        assert line == (
            "quicktalk_render_chunk "
            f"session=perf-session chunk={chunk_index} "
            "samples=4 reps=1 frames=1 "
            "feature_ms=1.0 "
            "generate_ms=1.0 "
            "encode_ms=1.0 "
            "total_ms=7.0"
        )
