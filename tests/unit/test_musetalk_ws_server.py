from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from model_backends.musetalk import musetalk_ws_server as server


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in os.environ:
        if key.startswith("OMNIRT_MUSETALK_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("OMNIRT_MUSETALK_DEVICE", "cpu")
    monkeypatch.setattr(server, "_RUNTIME", None)


def _files(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


@pytest.fixture
def model_tree(tmp_path, monkeypatch):
    models = tmp_path / "models"
    whisper = tmp_path / "hf-whisper-tiny"
    _files(models, "sd-vae-ft-mse/diffusion_pytorch_model.bin", "dwpose/dw-ll_ucoco_384.pth",
           "face-parse-bisenet/79999_iter.pth", "face-parse-bisenet/resnet18-5c106cde.pth")
    monkeypatch.setenv("OMNIRT_MUSETALK_MODELS_DIR", str(models))
    monkeypatch.setenv("OMNIRT_MUSETALK_WHISPER_DIR", str(whisper))

    def create(version):
        if version == "v15":
            _files(models, "musetalkV15/musetalk.json", "musetalkV15/unet.pth")
            _files(whisper, "config.json", "preprocessor_config.json", "model.safetensors")
        else:
            _files(models, "musetalk/musetalk.json", "musetalk/pytorch_model.bin", "whisper/tiny.pt")
        return models, whisper

    return create


def _module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen = []

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, latents, timesteps, *, encoder_hidden_states):
        self.seen.append(encoder_hidden_states.clone())
        return SimpleNamespace(sample=encoder_hidden_states[:, :1, :1])


@pytest.fixture
def runtime_factory(tmp_path, monkeypatch, model_tree):
    repo = tmp_path / "MuseTalk"
    repo.mkdir()
    monkeypatch.setattr(server, "_inject_musetalk_repo", lambda: repo)
    monkeypatch.setattr(server, "_patch_torch_load_weights_only", Mock())
    legacy_patch = Mock()
    monkeypatch.setattr(server, "_patch_openai_whisper_torch_load", legacy_patch)
    for name in ("musetalk", "musetalk.models", "musetalk.utils", "musetalk.whisper"):
        _module(monkeypatch, name)

    def create(version=None):
        if version is not None:
            monkeypatch.setenv("OMNIRT_MUSETALK_VERSION", version)
        selected = version or "v1"
        models, whisper_dir = model_tree(selected)
        unet_model = _Model()
        unet = Mock(return_value=SimpleNamespace(model=unet_model))
        vae = Mock(return_value=SimpleNamespace(
            vae=_Model(),
            get_latents_for_unet=Mock(return_value=torch.zeros(1, 8, 2, 2)),
            decode_latents=lambda latents: [
                np.full((16, 16, 3), int(value.item()), dtype=np.uint8) for value in latents[:, 0, 0]
            ],
        ))
        _module(monkeypatch, "musetalk.models.unet", UNet=unet, PositionalEncoding=Mock(return_value=torch.nn.Identity()))
        _module(monkeypatch, "musetalk.models.vae", VAE=vae)
        parser = Mock()
        _module(monkeypatch, "musetalk.utils.face_parsing", FaceParsing=parser)
        audio = Mock()
        whisper = _Model()
        whisper_loader = Mock(return_value=whisper)
        if selected == "v1":
            _module(monkeypatch, "musetalk.whisper.audio2feature", Audio2Feature=audio)
            monkeypatch.setitem(sys.modules, "musetalk.utils.audio_processor", None)
            monkeypatch.setitem(sys.modules, "transformers", None)
        else:
            _module(monkeypatch, "musetalk.utils.audio_processor", AudioProcessor=audio)
            _module(monkeypatch, "transformers", WhisperModel=SimpleNamespace(from_pretrained=whisper_loader))
            monkeypatch.setitem(sys.modules, "musetalk.whisper.audio2feature", None)
        runtime = server.MuseTalkRuntime()
        return runtime, SimpleNamespace(
            models=models, whisper_dir=whisper_dir, unet=unet, vae=vae, parser=parser,
            audio=audio, whisper=whisper, whisper_loader=whisper_loader, legacy_patch=legacy_patch,
        )

    return create


@pytest.mark.parametrize("version", [None, "v1", "v15"])
def test_versioned_load_and_model_links(runtime_factory, monkeypatch, caplog, version):
    monkeypatch.setenv("OMNIRT_MUSETALK_BBOX_SHIFT", "17")
    with caplog.at_level(logging.INFO):
        runtime, calls = runtime_factory(version)
    selected = version or "v1"
    assert runtime.version == selected
    assert f"MuseTalk version={selected}" in caplog.text
    assert runtime.bbox_shift == (0 if selected == "v15" else 17)
    model_dir = "musetalkV15" if selected == "v15" else "musetalk"
    weight_name = "unet.pth" if selected == "v15" else "pytorch_model.bin"
    calls.unet.assert_called_once_with(
        unet_config=str(calls.models / model_dir / "musetalk.json"),
        model_path=str(calls.models / model_dir / weight_name), device=torch.device("cpu"),
    )
    calls.vae.assert_called_once_with(model_path=str(calls.models / "sd-vae-ft-mse"))
    assert (runtime.repo / "models" / model_dir).resolve() == calls.models / model_dir
    assert (runtime.repo / "models" / "face-parse-bisent").resolve() == calls.models / "face-parse-bisenet"
    if selected == "v15":
        calls.audio.assert_called_once_with(feature_extractor_path=str(calls.whisper_dir))
        calls.whisper_loader.assert_called_once_with(str(calls.whisper_dir))
        calls.parser.assert_called_once_with(left_cheek_width=90, right_cheek_width=90)
        calls.legacy_patch.assert_not_called()
        assert calls.whisper.dtype == runtime.unet.model.dtype == torch.float16
        assert not calls.whisper.training
        assert not calls.whisper.weight.requires_grad
        assert runtime.extra_margin == 10
        assert runtime.parsing_mode == "jaw"
        assert not (runtime.repo / "models" / "whisper").exists()
        assert (runtime.repo / "models" / "whisper-hf").resolve() == calls.whisper_dir
    else:
        calls.audio.assert_called_once_with(model_path=str(calls.models / "whisper" / "tiny.pt"))
        calls.parser.assert_called_once_with()
        calls.legacy_patch.assert_called_once_with()
        assert runtime.whisper is None
        assert (runtime.repo / "models" / "whisper").resolve() == calls.models / "whisper"


def test_default_hf_whisper_directory(monkeypatch):
    monkeypatch.delenv("OMNIRT_MUSETALK_WHISPER_DIR", raising=False)
    assert server._whisper_dir() == Path("/models/whisper-hf")


def test_invalid_version_fails_before_loading(monkeypatch):
    monkeypatch.setenv("OMNIRT_MUSETALK_VERSION", "v2")
    with pytest.raises(RuntimeError, match="OMNIRT_MUSETALK_VERSION=.*expected v1 or v15"):
        server.MuseTalkRuntime()


@pytest.mark.parametrize("weight", ["model.safetensors", "pytorch_model.bin"])
def test_v15_layout_accepts_hf_weights_without_legacy_files(model_tree, monkeypatch, weight):
    models, whisper = model_tree("v15")
    (whisper / "model.safetensors").unlink()
    _files(whisper, weight)
    monkeypatch.setenv("OMNIRT_MUSETALK_VERSION", "v15")
    server._check_model_layout(models)


@pytest.mark.parametrize(("root", "missing", "error"), [
    ("models", "musetalkV15/unet.pth", "MuseTalk v15 UNet weights"),
    ("models", "musetalkV15/musetalk.json", "MuseTalk v15 UNet config"),
    ("whisper", "config.json", "MuseTalk v15 HF Whisper config"),
    ("whisper", "preprocessor_config.json", "MuseTalk v15 HF Whisper feature extractor config"),
    ("whisper", "model.safetensors", "MuseTalk v15 HF Whisper weights"),
])
def test_v15_missing_weights_are_explicit(model_tree, monkeypatch, root, missing, error):
    models, whisper = model_tree("v15")
    model_tree("v1")  # Existing v1 files must not hide missing v15 files.
    path = (models if root == "models" else whisper) / missing
    path.unlink()
    monkeypatch.setenv("OMNIRT_MUSETALK_VERSION", "v15")
    with pytest.raises(RuntimeError, match=error) as exc:
        server._check_model_layout(models)
    assert str(path.parent if missing == "model.safetensors" else path) in str(exc.value)


def test_v15_missing_whisper_directory(model_tree, monkeypatch, tmp_path):
    models, _ = model_tree("v15")
    monkeypatch.setenv("OMNIRT_MUSETALK_VERSION", "v15")
    monkeypatch.setenv("OMNIRT_MUSETALK_WHISPER_DIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="MuseTalk v15 HF Whisper directory.*OMNIRT_MUSETALK_WHISPER_DIR"):
        server._check_model_layout(models)


@pytest.mark.parametrize("missing", ["musetalk/pytorch_model.bin", "musetalk/musetalk.json", "whisper/tiny.pt"])
def test_default_v1_still_requires_legacy_files(model_tree, missing):
    models, _ = model_tree("v1")
    model_tree("v15")
    path = models / missing
    path.unlink()
    with pytest.raises(RuntimeError) as exc:
        server._check_model_layout(models)
    assert str(path) in str(exc.value)


def _landmarks(monkeypatch, frame, box=(4, 4, 24, 24)):
    detect = Mock(return_value=([box], [frame]))
    _module(monkeypatch, "musetalk.utils.preprocessing", coord_placeholder=(0, 0, 0, 0), get_landmark_and_bbox=detect)
    return detect


@pytest.mark.parametrize(("margin", "expected_y2"), [(None, 34), (0, 24), (7, 31), (100, 40)])
def test_v15_session_uses_official_crop_and_blending(runtime_factory, monkeypatch, margin, expected_y2):
    if margin is not None:
        monkeypatch.setenv("OMNIRT_MUSETALK_EXTRA_MARGIN", str(margin))
    monkeypatch.setenv("OMNIRT_MUSETALK_BBOX_SHIFT", "ignored-by-v15")
    runtime, calls = runtime_factory("v15")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    detect = _landmarks(monkeypatch, frame)
    blend = Mock(return_value=(np.zeros((40, 40), dtype=np.uint8), (0, 0, 40, 40)))
    _module(monkeypatch, "musetalk.utils.blending", get_image_prepare_material=blend)
    state = runtime.prepare_session(frame)
    assert detect.call_args.args[1] == 0
    assert state.face_box == (4, 4, 24, expected_y2)
    assert blend.call_args.args[1] == state.face_box
    assert blend.call_args.kwargs == {"fp": runtime.face_parser, "mode": "jaw"}
    assert calls.vae.return_value.get_latents_for_unet.call_args.args[0].shape == (256, 256, 3)
    assert len(state.latent_cycle) == len(state.mask_cycle) == 1


def test_v15_configurable_parsing_parameters(runtime_factory, monkeypatch):
    monkeypatch.setenv("OMNIRT_MUSETALK_PARSING_MODE", "neck")
    monkeypatch.setenv("OMNIRT_MUSETALK_LEFT_CHEEK_WIDTH", "85")
    monkeypatch.setenv("OMNIRT_MUSETALK_RIGHT_CHEEK_WIDTH", "95")
    runtime, calls = runtime_factory("v15")
    calls.parser.assert_called_once_with(left_cheek_width=85, right_cheek_width=95)
    blend = Mock(return_value=(None, None))
    _module(monkeypatch, "musetalk.utils.blending", get_image_prepare_material=blend)
    runtime._prepare_blend_material(np.zeros((40, 40, 3)), (4, 4, 24, 24))
    assert blend.call_args.kwargs["mode"] == "neck"


def test_v1_session_retains_raw_mask_crop_and_shift(runtime_factory, monkeypatch):
    monkeypatch.setenv("OMNIRT_MUSETALK_BBOX_SHIFT", "9")
    monkeypatch.setenv("OMNIRT_MUSETALK_EXTRA_MARGIN", "invalid-v15-only-setting")
    runtime, _ = runtime_factory()
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    detect = _landmarks(monkeypatch, frame)
    # Only the legacy APIs exist. v1 must retain expand=1.2 and default raw parsing.
    segment = Mock(side_effect=lambda image, **kwargs: Image.new("L", image.size, 255))
    crop = Mock(return_value=((0, 0, 40, 40), 20))
    _module(monkeypatch, "musetalk.utils.blending", face_seg=segment, get_crop_box=crop)
    state = runtime.prepare_session(frame)
    assert detect.call_args.args[1] == 9
    assert state.face_box == (4, 4, 24, 24)
    crop.assert_called_once_with((4, 4, 24, 24), 1.2)
    assert segment.call_args.kwargs == {"fp": runtime.face_parser}
    assert state.mask_cycle[0].shape == (40, 40)
    assert state.mask_cycle[0][:16].sum() == 0
    assert state.mask_cycle[0][20:24].sum() > 0


@pytest.mark.parametrize("version", ["v1", "v15"])
def test_missing_preprocessing_keeps_fallback_v1_only(runtime_factory, monkeypatch, version):
    runtime, _ = runtime_factory(version)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    detect = _landmarks(monkeypatch, frame)
    detect.side_effect = ModuleNotFoundError("No module named 'mmcv'", name="mmcv")
    fallback = Mock(return_value=(4, 4, 24, 24))
    monkeypatch.setattr(server, "_detect_face_box_fallback", fallback)
    monkeypatch.setattr(runtime, "_prepare_blend_material", Mock(return_value=(np.zeros((40, 40)), (0, 0, 40, 40))))
    if version == "v15":
        with pytest.raises(RuntimeError, match="MuseTalk v15 requires upstream landmark preprocessing.*mmcv"):
            runtime.prepare_session(frame)
        fallback.assert_not_called()
    else:
        assert runtime.prepare_session(frame).face_box == (4, 4, 24, 24)
        fallback.assert_called_once()


@pytest.mark.parametrize("version", ["v1", "v15"])
@pytest.mark.parametrize("shortfall", [0, 2])
def test_render_preserves_context_chunk_length_and_tensor_features(runtime_factory, monkeypatch, version, shortfall):
    runtime, _ = runtime_factory(version)
    writes = []
    _module(monkeypatch, "soundfile", write=lambda path, audio, rate: writes.append((audio.copy(), rate)))
    api_calls = []

    def get_audio_feature(path, *, weight_dtype):
        assert Path(path).is_file()
        assert not torch.is_grad_enabled()
        assert weight_dtype == torch.float16
        return [torch.ones(1, 80, 3000)], len(writes[-1][0])

    def get_whisper_chunk(features, device, dtype, whisper, length, **kwargs):
        assert not torch.is_grad_enabled()
        assert device == runtime.device and dtype == runtime.unet.model.dtype
        assert whisper is runtime.whisper
        api_calls.append(kwargs)
        count = length * 25 // 16000 - shortfall
        return torch.arange(count, dtype=dtype).view(-1, 1, 1).expand(-1, 50, 384)

    if version == "v15":
        runtime.audio_processor = SimpleNamespace(get_audio_feature=get_audio_feature, get_whisper_chunk=get_whisper_chunk)
    else:
        runtime.audio_processor = SimpleNamespace(
            audio2feat=lambda path: len(writes[-1][0]),
            feature2chunks=lambda length, *, fps: [
                np.full((50, 384), index, dtype=np.float32) for index in range(length * fps // 16000 - shortfall)
            ],
        )

    def datagen(chunks, latents, *, batch_size, device):
        # Mirrors the upstream stack/cat contract; no numpy conversion of HF tensors.
        assert all(isinstance(chunk, torch.Tensor) for chunk in chunks)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            yield torch.stack(batch), torch.cat([latents[0]] * len(batch))

    _module(monkeypatch, "musetalk.utils.utils", datagen=datagen)
    _module(monkeypatch, "musetalk.utils.blending", get_image_blending=lambda frame, face, *args: face)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    state = server.MuseTalkSessionState(frame, (0, 0, 16, 16), [torch.zeros(1, 8, 2, 2)], [frame], [frame[:, :, 0]], [(0, 0, 16, 16)])
    pcm = np.full(16000, 8192, dtype=np.int16)
    first = runtime.render_chunk(state, pcm, slice_len=25, fps=25)
    second = runtime.render_chunk(state, -pcm, slice_len=25, fps=25)
    assert len(first) == len(second) == 25
    assert [int(frame[0, 0, 0]) for frame in first] == [min(i, 24 - shortfall) for i in range(25)]
    assert [int(frame[0, 0, 0]) for frame in second] == [min(i, 49 - shortfall) for i in range(25, 50)]
    assert state.frame_cursor == 50
    np.testing.assert_array_equal(state.audio_context, -pcm)
    assert [len(audio) for audio, _ in writes] == [16000, 32000]
    np.testing.assert_array_equal(writes[1][0], np.concatenate([pcm, -pcm]).astype(np.float32) / 32768)
    if version == "v15":
        assert api_calls == [{"fps": 25, "audio_padding_length_left": 2, "audio_padding_length_right": 2}] * 2


@pytest.mark.parametrize("version", ["v1", "v15"])
def test_preload_and_ws_protocol_stay_compatible(runtime_factory, monkeypatch, version):
    runtime, _ = runtime_factory(version)
    constructor = Mock(return_value=runtime)
    monkeypatch.setattr(server, "MuseTalkRuntime", constructor)
    monkeypatch.setenv("OMNIRT_MUSETALK_PRELOAD", "1")
    server._preload_runtime()
    frame = np.full((16, 24, 3), 128, dtype=np.uint8)
    state = object()
    monkeypatch.setattr(runtime, "prepare_session", Mock(return_value=state))
    monkeypatch.setattr(runtime, "render_chunk", Mock(return_value=[frame] * 25))
    ok, png = cv2.imencode(".png", frame)
    assert ok
    messages = [json.dumps({"type": "init", "ref_image": base64.b64encode(png).decode()}),
                b"AUDI" + bytes(32000), b"AUDI" + bytes(32000), json.dumps({"type": "close"})]

    class WebSocket:
        def __init__(self):
            self.sent = []

        async def __aiter__(self):
            for message in messages:
                yield message

        async def send(self, message):
            self.sent.append(message)

    ws = WebSocket()
    asyncio.run(server._handler(ws))
    constructor.assert_called_once_with()
    runtime.prepare_session.assert_called_once()
    assert runtime.render_chunk.call_count == 2
    assert runtime.render_chunk.call_args.args[0] is state
    assert runtime.render_chunk.call_args.kwargs == {"slice_len": 25, "fps": 25}
    assert json.loads(ws.sent[0]) == {"type": "init_ok", "frame_num": 33, "motion_frames_num": 8,
                                      "slice_len": 25, "fps": 25, "height": 16, "width": 24}
    assert json.loads(ws.sent[3]) == {"type": "close_ok"}
    for message in ws.sent[1:3]:
        assert message[:4] == b"VIDX"
        assert struct.unpack_from("<I", message, 4)[0] == 25
        offset = 8
        for _ in range(25):
            size = struct.unpack_from("<I", message, offset)[0]
            jpeg = message[offset + 4:offset + 4 + size]
            assert jpeg[:2] == b"\xff\xd8"
            assert cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR).shape == frame.shape
            offset += 4 + size
        assert offset == len(message)


@pytest.mark.parametrize("version", ["v1", "v15"])
def test_startup_logs_version_without_preload(monkeypatch, caplog, version):
    monkeypatch.setenv("OMNIRT_MUSETALK_VERSION", version)

    async def run_server(host, port):
        pass

    monkeypatch.setattr(server, "_run_server", run_server)
    with caplog.at_level(logging.INFO):
        assert server.main([]) == 0
    assert f"MuseTalk version={version}" in caplog.text


@pytest.fixture
def silence_runtime(runtime_factory, monkeypatch):
    def create(version="v15"):
        runtime, _ = runtime_factory(version)
        writes = []
        _module(monkeypatch, "soundfile", write=lambda path, audio, rate: writes.append(audio.copy()))
        # Deliberately nonzero even for zero PCM: model-generated silence may still
        # have an open mouth. The real energy gate must correct that output.
        runtime.audio_processor = SimpleNamespace(
            get_audio_feature=lambda *args, **kwargs: (None, len(writes[-1])),
            get_whisper_chunk=lambda features, device, dtype, whisper, length, **kwargs:
                torch.full((length * 25 // 16000, 50, 384), 100.0, dtype=dtype),
            audio2feat=lambda path: len(writes[-1]),
            feature2chunks=lambda length, **kwargs:
                [np.full((50, 384), 100.0, dtype=np.float32)] * (length * 25 // 16000),
        )
        runtime.pe = lambda features: features + 7  # Observe zeros BEFORE PE.
        runtime.vae.decode_latents = lambda latents: [
            np.full((256, 256, 3), int(value.item()), dtype=np.uint8) for value in latents[:, 0, 0]
        ]

        def datagen(chunks, latents, *, batch_size, device):
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                yield torch.stack(batch), torch.cat([latents[i % len(latents)] for i in range(start, start + len(batch))])

        _module(monkeypatch, "musetalk.utils.utils", datagen=datagen)
        _module(monkeypatch, "musetalk.utils.blending", get_image_blending=lambda frame, face, *args: face)
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        state = server.MuseTalkSessionState(
            frame, (0, 0, 256, 256), [torch.zeros(1, 8, 2, 2)], [frame],
            [frame[:, :, 0]], [(0, 0, 256, 256)],
        )
        return runtime, state, writes
    return create


def test_v15_silent_tail_converges_with_speech_context_and_reuses_session_cache(silence_runtime):
    runtime, state, writes = silence_runtime()
    speech = np.full(16000, 8192, dtype=np.int16)
    first = runtime.render_chunk(state, speech, slice_len=25, fps=25)
    assert not getattr(state, "closed_prediction_cache", {})
    silence = runtime.render_chunk(state, np.zeros(16000, dtype=np.int16), slice_len=25, fps=25)
    # Context remains real speech plus zeros; gate energy comes from current PCM.
    np.testing.assert_array_equal(writes[-1][:16000], speech / 32768.0)
    assert not writes[-1][16000:].any()
    assert first[-1][176, 128, 0] == 107
    assert silence[-1][176, 128, 0] < 20
    assert silence[-1][0, 0, 0] == 107  # Keep non-mouth pixels from the normal prediction.
    assert len(first) == len(silence) == 25
    assert state.frame_cursor == 50
    reference = state.closed_prediction_cache[0]
    runtime.render_chunk(state, np.zeros(16000, dtype=np.int16), slice_len=25, fps=25)
    resumed = runtime.render_chunk(state, speech, slice_len=25, fps=25)
    assert state.closed_prediction_cache[0] is reference
    assert resumed[-1][176, 128, 0] == 107
    zero_reference_calls = [batch for batch in runtime.unet.model.seen if torch.all(batch == 7)]
    assert len(zero_reference_calls) == 1
    assert zero_reference_calls[0].shape == (1, 50, 384)
    assert state.frame_cursor == 100


def test_v15_mixed_speech_and_silence_only_gates_silent_frames(silence_runtime):
    runtime, state, _ = silence_runtime()
    pcm = np.concatenate([np.full(17 * 640, 8192, dtype=np.int16), np.zeros(8 * 640, dtype=np.int16)])
    frames = runtime.render_chunk(state, pcm, slice_len=25, fps=25)
    assert all(frame[176, 128, 0] == 107 for frame in frames[:17])
    assert all(frame[176, 128, 0] < 20 for frame in frames[17:])
    np.testing.assert_array_equal(server._compute_per_frame_energy(pcm / 32768.0, 25), [1] * 17 + [0] * 8)


@pytest.mark.parametrize("version", ["v1", "v15"])
def test_silence_gate_disabled_preserves_original_prediction(silence_runtime, monkeypatch, version):
    monkeypatch.setenv("OMNIRT_MUSETALK_SILENCE_GATE", "0" if version == "v15" else "invalid-v15-only-setting")
    runtime, state, _ = silence_runtime(version)
    frames = runtime.render_chunk(state, np.zeros(16000, dtype=np.int16), slice_len=25, fps=25)
    assert runtime.silence_gate == 0
    assert not state.closed_prediction_cache
    assert all(np.all(frame == 107) for frame in frames)


def test_closed_reference_cache_is_session_local(silence_runtime):
    runtime, first, _ = silence_runtime()
    second = server.MuseTalkSessionState(
        first.base_frame, first.face_box, first.latent_cycle, first.frame_cycle,
        first.mask_cycle, first.mask_coords_cycle,
    )
    for state in (first, second):
        runtime.render_chunk(state, np.zeros(16000, dtype=np.int16), slice_len=25, fps=25)
    assert first.closed_prediction_cache is not second.closed_prediction_cache
    assert first.closed_prediction_cache[0] is not second.closed_prediction_cache[0]
    assert sum(bool(torch.all(batch == 7)) for batch in runtime.unet.model.seen) == 2


def test_closed_mouth_blend_scales_with_gate_strength():
    prediction = np.full((256, 256, 3), 200, dtype=np.uint8)
    closed = np.full_like(prediction, 20)
    full = server._blend_prediction_toward_closed_mouth(prediction, closed, 1.0)
    half = server._blend_prediction_toward_closed_mouth(prediction, closed, 0.5)
    assert full[176, 128, 0] < half[176, 128, 0] < prediction[176, 128, 0]
    assert full[0, 0, 0] == half[0, 0, 0] == 200
    assert server._blend_prediction_toward_closed_mouth(prediction, closed, 0.0) is prediction


def test_ws_normal_audi_emits_corrected_silence_then_resumes_speech(silence_runtime, monkeypatch):
    runtime, state, _ = silence_runtime()
    monkeypatch.setattr(server, "_RUNTIME", runtime)
    runtime.prepare_session = Mock(return_value=state)
    ok, png = cv2.imencode(".png", state.base_frame)
    assert ok
    speech = np.full(16000, 8192, dtype=np.int16).tobytes()
    messages = [json.dumps({"type": "init", "ref_image": base64.b64encode(png).decode()}),
                b"AUDI" + speech, b"AUDI" + bytes(32000), b"AUDI" + speech,
                json.dumps({"type": "close"}), b"AUDI" + speech]

    class WebSocket:
        def __init__(self):
            self.sent = []

        async def __aiter__(self):
            for message in messages:
                yield message

        async def send(self, message):
            self.sent.append(message)

    ws = WebSocket()
    asyncio.run(server._handler(ws))
    assert "capabilities" not in json.loads(ws.sent[0])
    last_frames = []
    for payload in ws.sent[1:4]:
        assert payload[:4] == b"VIDX" and struct.unpack_from("<I", payload, 4)[0] == 25
        offset = 8
        for _ in range(25):
            size = struct.unpack_from("<I", payload, offset)[0]
            jpeg = payload[offset + 4:offset + 4 + size]
            offset += 4 + size
        assert offset == len(payload)
        last_frames.append(cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR))
    assert last_frames[0][176, 128, 0] > 100
    assert last_frames[1][176, 128, 0] < 20
    assert last_frames[2][176, 128, 0] > 100
    runtime.prepare_session.assert_called_once()
    assert state.frame_cursor == 75
    assert json.loads(ws.sent[4]) == {"type": "close_ok"}
    assert json.loads(ws.sent[5])["type"] == "error"
