#!/usr/bin/env python3
"""
MuseTalk WebSocket server (FlashTalk protocol compatible)

Same wire protocol as ``model_backends/wav2lip/wav2lip_ws_server.py`` / SoulX FlashTalk so
OpenTalking can use the same OmniRT audio2video client path for MuseTalk.

Inference is intentionally OmniRT-side. This service imports the upstream MuseTalk source tree,
loads MuseTalk v1 (default) or v1.5 weights, and exposes a FlashTalk-compatible WebSocket. OpenTalking stays
as orchestration/client code only.

Environment (high level):
  OMNIRT_MUSETALK_REPO              MuseTalk source checkout (default: <OMNIRT_HOME>/model-repos/MuseTalk)
  OMNIRT_MUSETALK_MODELS_DIR        Weight tree root (default: <omnirt>/models) — also sets
                                    MuseTalk model paths during loading.
  OMNIRT_MUSETALK_VERSION           v1 | v15 (default v1)
  OMNIRT_MUSETALK_WHISPER_DIR       v15 HF whisper-tiny directory (default /models/whisper-hf)
  OMNIRT_MUSETALK_EXTRA_MARGIN      v15 lower face crop margin (default 10); bbox_shift is always 0
  OMNIRT_MUSETALK_PARSING_MODE      v15 face parsing mode (default jaw)
  OMNIRT_MUSETALK_SILENCE_GATE      v15 normalized frame-energy threshold (default 0.04; 0 disables)
  OMNIRT_MUSETALK_LEFT_CHEEK_WIDTH / RIGHT_CHEEK_WIDTH  v15 parsing widths (default 90 each)
  OMNIRT_MUSETALK_DEVICE            auto | npu | npu:0 | cuda | cpu (default auto)
  OMNIRT_MUSETALK_NPU_INDEX         used when DEVICE=npu (default 0)
  OMNIRT_MUSETALK_HOST / PORT       bind (defaults 0.0.0.0:8766)
  OMNIRT_MUSETALK_PRELOAD           1/true: load weights at startup (default 1)
  OMNIRT_MUSETALK_DEFAULT_REF_IMAGE optional default ref_image if init omits it
  OMNIRT_MUSETALK_FRAME_NUM / MOTION_FRAMES_NUM / FPS  protocol chunking (defaults match wav2lip)
  OMNIRT_MUSETALK_JPEG_QUALITY      1-100 (default 85)
  OMNIRT_MUSETALK_MAX_LONG_EDGE / MIN_LONG_EDGE  ref_image resize (same semantics as wav2lip)

Dependencies: ``requirements-musetalk-ascend.txt`` (NPU) or ``requirements-musetalk-gpu.txt`` (CUDA).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import json
import logging
import os
import shutil
import tempfile
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch

LOG = logging.getLogger("omnirt.musetalk_ws")

MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"

DEFAULT_FRAME_NUM = int(os.environ.get("OMNIRT_MUSETALK_FRAME_NUM", "33"))
DEFAULT_MOTION_FRAMES_NUM = int(os.environ.get("OMNIRT_MUSETALK_MOTION_FRAMES_NUM", "8"))
DEFAULT_FPS = int(os.environ.get("OMNIRT_MUSETALK_FPS", "25"))
SAMPLE_RATE = 16000


def _omnirt_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_models_dir() -> Path:
    return _omnirt_root() / "models"


def _default_musetalk_repo() -> Path:
    home = os.environ.get("OMNIRT_HOME", "").strip()
    omnirt_home = Path(home).expanduser().resolve() if home else _omnirt_root() / ".omnirt"
    return omnirt_home / "model-repos" / "MuseTalk"


def _musetalk_repo() -> Path:
    raw = os.environ.get("OMNIRT_MUSETALK_REPO", "").strip()
    repo = Path(raw).expanduser().resolve() if raw else _default_musetalk_repo()
    if not (repo / "musetalk").is_dir():
        raise RuntimeError(
            f"MuseTalk source checkout not found: {repo}. "
            "Run `omnirt runtime install musetalk --device cuda` or set OMNIRT_MUSETALK_REPO "
            "to a MuseTalk checkout containing the musetalk/ package."
        )
    return repo


def _inject_musetalk_repo() -> Path:
    repo = _musetalk_repo()
    s = str(repo)
    if s not in sys.path:
        sys.path.insert(0, s)
    return repo


def _models_dir() -> Path:
    return Path(
        os.environ.get("OMNIRT_MUSETALK_MODELS_DIR", str(_default_models_dir()))
    ).expanduser().resolve()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise RuntimeError(f"Missing {label}: {path}")
    return path


def _musetalk_version() -> str:
    version = os.environ.get("OMNIRT_MUSETALK_VERSION", "v1").strip().lower()
    if version not in {"v1", "v15"}:
        raise RuntimeError(f"Invalid OMNIRT_MUSETALK_VERSION={version!r}; expected v1 or v15")
    return version


def _whisper_dir() -> Path:
    return Path(os.environ.get("OMNIRT_MUSETALK_WHISPER_DIR", "/models/whisper-hf")).expanduser().resolve()


def _unet_paths(models_dir: Path, version: str) -> tuple[Path, Path]:
    if version == "v15":
        return models_dir / "musetalkV15" / "musetalk.json", models_dir / "musetalkV15" / "unet.pth"
    return models_dir / "musetalk" / "musetalk.json", models_dir / "musetalk" / "pytorch_model.bin"


def _check_model_layout(
    models_dir: Path, version: str | None = None, whisper_dir: Path | None = None,
) -> None:
    version = _musetalk_version() if version is None else version
    unet_config, unet_weights = _unet_paths(models_dir, version)
    _require_file(unet_weights, f"MuseTalk {version} UNet weights")
    _require_file(unet_config, f"MuseTalk {version} UNet config")
    _require_dir(models_dir / "sd-vae-ft-mse", "MuseTalk VAE directory")
    _require_file(
        models_dir / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin",
        "MuseTalk VAE weights",
    )
    if version == "v15":
        whisper_dir = _whisper_dir() if whisper_dir is None else whisper_dir
        _require_dir(whisper_dir, "MuseTalk v15 HF Whisper directory (OMNIRT_MUSETALK_WHISPER_DIR)")
        _require_file(whisper_dir / "config.json", "MuseTalk v15 HF Whisper config")
        _require_file(whisper_dir / "preprocessor_config.json", "MuseTalk v15 HF Whisper feature extractor config")
        if not any((whisper_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise RuntimeError(
                f"Missing MuseTalk v15 HF Whisper weights: {whisper_dir} "
                "(expected model.safetensors or pytorch_model.bin; whisper/tiny.pt is v1 only)"
            )
    else:
        _require_file(models_dir / "whisper" / "tiny.pt", "Whisper tiny checkpoint")
    _require_file(models_dir / "dwpose" / "dw-ll_ucoco_384.pth", "DWPose checkpoint")
    _require_file(models_dir / "face-parse-bisenet" / "79999_iter.pth", "face parsing checkpoint")
    _ensure_face_parse_resnet(models_dir)


def _ensure_face_parse_resnet(models_dir: Path) -> Path:
    target = models_dir / "face-parse-bisenet" / "resnet18-5c106cde.pth"
    if target.is_file():
        return target

    torchvision_cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / target.name
    if torchvision_cache.is_file():
        shutil.copy2(torchvision_cache, target)
        return target

    url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading face-parse ResNet18 weights: %s -> %s", url, target)
    urlretrieve(url, target)
    return _require_file(target, "face parsing ResNet18 checkpoint")


def _prepare_repo_model_links(
    repo: Path, models_dir: Path, version: str | None = None, whisper_dir: Path | None = None,
) -> None:
    """Make upstream MuseTalk hard-coded ./models paths resolve to OmniRT's model root.

    Several upstream helpers keep default paths such as ./models/dwpose/... and
    ./models/face-parse-bisent/.... Running with cwd=repo and placing symlinks
    avoids vendoring or mutating upstream source files.
    """
    repo_models = repo / "models"
    repo_models.mkdir(parents=True, exist_ok=True)
    version = _musetalk_version() if version is None else version
    links = {
        "sd-vae-ft-mse": models_dir / "sd-vae-ft-mse",
        "dwpose": models_dir / "dwpose",
        # Upstream misspells this directory as "bisent"; OpenTalking/OmniRT
        # model layout uses the clearer "bisenet".
        "face-parse-bisent": models_dir / "face-parse-bisenet",
    }
    if version == "v15":
        links["musetalkV15"] = models_dir / "musetalkV15"
        links["whisper-hf"] = _whisper_dir() if whisper_dir is None else whisper_dir
    else:
        links["musetalk"] = models_dir / "musetalk"
        links["whisper"] = models_dir / "whisper"
    for name, target in links.items():
        _require_dir(target, f"MuseTalk model directory {name}")
        link = repo_models / name
        if link.exists() or link.is_symlink():
            if link.resolve() != target.resolve():
                LOG.warning("MuseTalk repo model path already exists; keeping it: %s", link)
            continue
        link.symlink_to(target, target_is_directory=True)


def _temporary_cwd(path: Path):
    class _Cwd:
        def __enter__(self) -> None:
            self.previous = Path.cwd()
            os.chdir(path)

        def __exit__(self, exc_type, exc, tb) -> None:
            os.chdir(self.previous)

    return _Cwd()


def _detect_face_box_fallback(repo: Path, image_path: Path) -> tuple[int, int, int, int]:
    with _temporary_cwd(repo):
        from musetalk.utils.face_detection import FaceAlignment, LandmarksType

        s3fd_path = _models_dir() / "wav2lip" / "s3fd.pth"
        if s3fd_path.is_file():
            os.environ.setdefault("FACE_ALIGNMENT_DETECTOR_PATH", str(s3fd_path))
        detector = FaceAlignment(LandmarksType._2D, flip_input=False, device=_inference_device_str())
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read ref_image for fallback face detection: {image_path}")
        boxes = detector.get_detections_for_batch(np.asarray([frame]))
        box = boxes[0] if boxes else None
        if box is None:
            raise RuntimeError("MuseTalk fallback face detector could not find a face in ref_image")
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"MuseTalk fallback face box is invalid: {(x1, y1, x2, y2)}")
        return x1, y1, x2, y2


def _decode_init_ref_image(msg: dict) -> tuple[np.ndarray | None, str | None]:
    ref_b64 = (msg.get("ref_image") or "").strip()
    image_data: bytes | None = None
    if ref_b64:
        try:
            image_data = base64.b64decode(ref_b64)
        except Exception:
            return None, "Invalid base64 ref_image"
    else:
        default_path = os.environ.get("OMNIRT_MUSETALK_DEFAULT_REF_IMAGE", "").strip()
        if not default_path:
            return None, "Missing ref_image (or set OMNIRT_MUSETALK_DEFAULT_REF_IMAGE)"
        path = Path(default_path).expanduser()
        if not path.is_file():
            return None, f"OMNIRT_MUSETALK_DEFAULT_REF_IMAGE not found: {path}"
        image_data = path.read_bytes()
        LOG.info("init: using OMNIRT_MUSETALK_DEFAULT_REF_IMAGE=%s", path)

    buf = np.frombuffer(image_data, dtype=np.uint8)
    base_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if base_frame is None:
        return None, "Could not decode ref_image"
    return base_frame, None


def _max_long_edge_limit() -> int:
    raw = os.environ.get("OMNIRT_MUSETALK_MAX_LONG_EDGE", "768").strip()
    if raw in {"", "0", "none", "off"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning("Invalid OMNIRT_MUSETALK_MAX_LONG_EDGE=%r, using 768", raw)
        return 768


def _min_long_edge_limit() -> int:
    raw = os.environ.get("OMNIRT_MUSETALK_MIN_LONG_EDGE", "0").strip()
    if raw in {"", "0", "none", "off"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning("Invalid OMNIRT_MUSETALK_MIN_LONG_EDGE=%r, ignoring", raw)
        return 0


def _downscale_bgr_max_long_edge(bgr: np.ndarray, max_long_edge: int) -> np.ndarray:
    if max_long_edge <= 0:
        return bgr
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return bgr
    scale = max_long_edge / float(long_edge)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out)


def _upscale_bgr_min_long_edge(bgr: np.ndarray, min_long_edge: int) -> np.ndarray:
    if min_long_edge <= 0:
        return bgr
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge >= min_long_edge:
        return bgr
    scale = min_long_edge / float(long_edge)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(out)


def _slice_params() -> tuple[int, int, int]:
    frame_num = DEFAULT_FRAME_NUM
    motion = DEFAULT_MOTION_FRAMES_NUM
    slice_len = frame_num - motion
    if slice_len <= 0:
        raise ValueError("Need frame_num > motion_frames_num")
    return frame_num, motion, slice_len


def _audio_chunk_bytes(slice_len: int, fps: int) -> int:
    samples = slice_len * SAMPLE_RATE // fps
    return samples * 2


def _encode_video_message(jpeg_parts: list[bytes]) -> bytes:
    buf = bytearray()
    buf.extend(MAGIC_VIDEO)
    buf.extend(struct.pack("<I", len(jpeg_parts)))
    for jp in jpeg_parts:
        buf.extend(struct.pack("<I", len(jp)))
        buf.extend(jp)
    return bytes(buf)


def _try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401

        return True
    except ImportError:
        return False


def _inference_device_str() -> str:
    raw = os.environ.get("OMNIRT_MUSETALK_DEVICE", "auto").strip().lower()
    if raw in {"", "auto"}:
        if _try_import_torch_npu() and getattr(torch, "npu", None) is not None:
            try:
                if torch.npu.is_available():  # type: ignore[union-attr]
                    idx = (os.environ.get("OMNIRT_MUSETALK_NPU_INDEX", "0") or "0").strip()
                    return f"npu:{idx}"
            except Exception:
                pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if raw == "npu":
        idx = (os.environ.get("OMNIRT_MUSETALK_NPU_INDEX", "0") or "0").strip()
        return f"npu:{idx}"
    return raw


@dataclass
class MuseTalkSessionState:
    base_frame: np.ndarray
    face_box: tuple[int, int, int, int]
    latent_cycle: list[torch.Tensor]
    frame_cycle: list[np.ndarray]
    mask_cycle: list[np.ndarray]
    mask_coords_cycle: list[tuple[int, int, int, int]]
    frame_cursor: int = 0
    audio_context: np.ndarray | None = None
    closed_prediction_cache: dict[int, np.ndarray] = field(default_factory=dict)


def _compute_per_frame_energy(pcm_f32: np.ndarray, frame_count: int) -> np.ndarray:
    """Match OpenTalking's local MuseTalk energy gate, using only the current PCM."""
    if frame_count <= 0:
        return np.zeros((0,), dtype=np.float32)
    x = np.asarray(pcm_f32, dtype=np.float32).reshape(-1)
    energies = np.zeros((frame_count,), dtype=np.float32)
    if x.size == 0:
        return energies
    boundaries = np.linspace(0, x.shape[0], num=frame_count + 1, dtype=np.int32)
    for i in range(frame_count):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        if end <= start:
            end = min(x.shape[0], start + 1)
        window = x[start:end]
        if window.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float32)))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(window))))) if window.size > 1 else 0.0
        energies[i] = rms * (1.0 + 0.25 * zcr)
    peak = float(np.max(energies))
    if peak > 1e-6:
        floor = max(0.01, min(0.06, peak * 0.18))
        energies = np.clip((energies - floor) / max(1e-6, peak - floor), 0.0, 1.0)
        energies = energies * energies * (3.0 - 2.0 * energies)
    else:
        energies.fill(0.0)
    return energies.astype(np.float32, copy=False)


def _blend_prediction_toward_closed_mouth(
    prediction: np.ndarray, closed_prediction: np.ndarray, amount: float,
) -> np.ndarray:
    """Use the local adapter's feathered mouth ellipse before official face blending."""
    if amount <= 1e-4:
        return prediction
    size = prediction.shape[0]
    yy, xx = np.ogrid[:size, :size]
    ellipse = (
        ((xx - size // 2) / max(1, int(size * 0.20))) ** 2
        + ((yy - int(size * 0.69)) / max(1, int(size * 0.12))) ** 2
    ) <= 1.0
    mask = ellipse.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), size * 0.045)
    mask = (mask * float(np.clip(amount, 0.0, 1.0)))[:, :, np.newaxis]
    blended = prediction.astype(np.float32) * (1.0 - mask) + closed_prediction.astype(np.float32) * mask
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _patch_openai_whisper_torch_load() -> None:
    """openai-whisper 对 torch>=1.13 使用 ``torch.load(..., weights_only=True)``；官方 ``tiny.pt`` 是含
    ``dims`` 等字段的旧 pickle，在 PyTorch 2.4+ 上会 ``UnpicklingError: Unsupported operand …``。
    仅在 ``whisper.load_model`` 调用期间强制 ``weights_only=False``。"""
    try:
        import whisper
    except ImportError:
        return
    if getattr(whisper, "_omnirt_weights_only_patch", False):
        return

    _orig_load_model = whisper.load_model

    def _load_model_wrapped(*args: Any, **kwargs: Any) -> Any:
        _orig_torch_load = torch.load

        def _torch_load_wrapped(*a: Any, **kw: Any) -> Any:
            kw2 = dict(kw)
            kw2["weights_only"] = False
            return _orig_torch_load(*a, **kw2)

        torch.load = _torch_load_wrapped  # type: ignore[assignment]
        try:
            return _orig_load_model(*args, **kwargs)
        finally:
            torch.load = _orig_torch_load  # type: ignore[assignment]

    whisper.load_model = _load_model_wrapped  # type: ignore[assignment]
    whisper._omnirt_weights_only_patch = True  # type: ignore[attr-defined]
    LOG.info("Patched openai-whisper load_model to use torch.load(weights_only=False) for legacy checkpoints")


def _patch_torch_load_weights_only() -> None:
    if getattr(torch, "_omnirt_musetalk_weights_only_patch", False):
        return

    _orig_torch_load = torch.load

    def _torch_load_wrapped(*args: Any, **kwargs: Any) -> Any:
        kw = dict(kwargs)
        kw.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kw)

    torch.load = _torch_load_wrapped  # type: ignore[assignment]
    torch._omnirt_musetalk_weights_only_patch = True  # type: ignore[attr-defined]
    LOG.info("Patched torch.load to default weights_only=False for legacy MuseTalk checkpoints")


class MuseTalkRuntime:
    def __init__(self) -> None:
        self.version = _musetalk_version()
        LOG.info("MuseTalk version=%s", self.version)
        self.repo = _inject_musetalk_repo()
        self.models_dir = _models_dir()
        self.whisper_dir = _whisper_dir() if self.version == "v15" else None
        _check_model_layout(self.models_dir, self.version, self.whisper_dir)
        _prepare_repo_model_links(self.repo, self.models_dir, self.version, self.whisper_dir)
        self.device = torch.device(_inference_device_str())
        self.batch_size = max(1, int(os.environ.get("OMNIRT_MUSETALK_BATCH_SIZE", "4")))
        self.bbox_shift = 0 if self.version == "v15" else int(os.environ.get("OMNIRT_MUSETALK_BBOX_SHIFT", "0"))
        if self.version == "v15":
            self.extra_margin = int(os.environ.get("OMNIRT_MUSETALK_EXTRA_MARGIN", "10"))
            self.parsing_mode = os.environ.get("OMNIRT_MUSETALK_PARSING_MODE", "jaw")
        self.silence_gate = 0.0
        if self.version == "v15":
            self.silence_gate = float(os.environ.get("OMNIRT_MUSETALK_SILENCE_GATE", "0.04"))
            if not 0.0 <= self.silence_gate <= 1.0:
                raise ValueError("OMNIRT_MUSETALK_SILENCE_GATE must be between 0 and 1")
        self.audio_context_samples = max(
            0,
            int(os.environ.get("OMNIRT_MUSETALK_AUDIO_CONTEXT_SAMPLES", str(SAMPLE_RATE))),
        )
        self.vae: Any = None
        self.unet: Any = None
        self.pe: Any = None
        self.audio_processor: Any = None
        self.whisper: Any = None
        self.face_parser: Any = None
        self._load()

    def _load(self) -> None:
        os.environ.setdefault("FFMPEG_PATH", str(self.repo / "ffmpeg-6.1-amd64-static"))
        _patch_torch_load_weights_only()
        if self.version == "v1":
            _patch_openai_whisper_torch_load()
        with _temporary_cwd(self.repo):
            from musetalk.models.unet import PositionalEncoding, UNet
            from musetalk.models.vae import VAE
            from musetalk.utils.face_parsing import FaceParsing
            if self.version == "v1":
                from musetalk.whisper.audio2feature import Audio2Feature

                self.audio_processor = Audio2Feature(model_path=str(self.models_dir / "whisper" / "tiny.pt"))
            self.vae = VAE(model_path=str(self.models_dir / "sd-vae-ft-mse"))
            unet_config, unet_weights = _unet_paths(self.models_dir, self.version)
            self.unet = UNet(
                unet_config=str(unet_config),
                model_path=str(unet_weights),
                device=self.device,
            )
            self.pe = PositionalEncoding(d_model=384)
            if self.version == "v15":
                self.face_parser = FaceParsing(
                    left_cheek_width=int(os.environ.get("OMNIRT_MUSETALK_LEFT_CHEEK_WIDTH", "90")),
                    right_cheek_width=int(os.environ.get("OMNIRT_MUSETALK_RIGHT_CHEEK_WIDTH", "90")),
                )
            else:
                self.face_parser = FaceParsing()

        self.pe = self.pe.to(self.device).half()
        self.vae.vae = self.vae.vae.to(self.device).half()
        self.unet.model = self.unet.model.to(self.device).half()
        self.unet.device = self.device
        self.unet.model.eval()
        self.vae.vae.eval()
        if self.version == "v15":
            # Official realtime_inference.py at 0a89dec45a0192b824e3cf4daf96c239440c5ed8.
            from musetalk.utils.audio_processor import AudioProcessor
            from transformers import WhisperModel

            self.audio_processor = AudioProcessor(feature_extractor_path=str(self.whisper_dir))
            self.whisper = WhisperModel.from_pretrained(str(self.whisper_dir))
            self.whisper = self.whisper.to(device=self.device, dtype=self.unet.model.dtype).eval()
            self.whisper.requires_grad_(False)
        LOG.info("Loaded MuseTalk runtime repo=%s models=%s device=%s", self.repo, self.models_dir, self.device)

    def prepare_session(self, base_frame: np.ndarray) -> MuseTalkSessionState:
        with _temporary_cwd(self.repo):
            with tempfile.TemporaryDirectory(prefix="omnirt-musetalk-ref-") as tmp:
                image_path = Path(tmp) / "00000000.png"
                cv2.imwrite(str(image_path), base_frame)
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise RuntimeError("MuseTalk could not reload temporary ref_image")
                try:
                    from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox

                    coords, frames = get_landmark_and_bbox([str(image_path)], self.bbox_shift)
                    face_box = coords[0]
                    if face_box == coord_placeholder:
                        raise RuntimeError("MuseTalk could not detect a usable face box in ref_image")
                    x1, y1, x2, y2 = [int(v) for v in face_box]
                    working_frame = frames[0].copy()
                except ModuleNotFoundError as exc:
                    if self.version == "v15":
                        raise RuntimeError(
                            f"MuseTalk v15 requires upstream landmark preprocessing; missing dependency: {exc.name}"
                        ) from exc
                    LOG.warning(
                        "MuseTalk preprocessing fallback: optional dependency missing (%s), "
                        "using SFD-only face box detection",
                        exc.name or str(exc),
                    )
                    x1, y1, x2, y2 = _detect_face_box_fallback(self.repo, image_path)
                    working_frame = frame.copy()
            if self.version == "v15":
                y2 = min(y2 + self.extra_margin, working_frame.shape[0])
            crop_frame = working_frame[y1:y2, x1:x2]
            if crop_frame.size == 0:
                raise RuntimeError(f"MuseTalk face crop is empty: {(x1, y1, x2, y2)}")
            resized_crop = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latent = self.vae.get_latents_for_unet(resized_crop).to(self.device)
            mask, crop_box = self._prepare_blend_material(working_frame, (x1, y1, x2, y2))

        return MuseTalkSessionState(
            base_frame=working_frame.copy(),
            face_box=(x1, y1, x2, y2),
            latent_cycle=[latent],
            frame_cycle=[working_frame.copy()],
            mask_cycle=[mask],
            mask_coords_cycle=[tuple(int(v) for v in crop_box)],
        )

    def _prepare_blend_material(
        self,
        image: np.ndarray,
        face_box: tuple[int, int, int, int],
        *,
        upper_boundary_ratio: float = 0.5,
        expand: float = 1.2,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        if self.version == "v15":
            with _temporary_cwd(self.repo):
                from musetalk.utils.blending import get_image_prepare_material

                # Use upstream's v1.5 mask, including its default expand=1.5.
                return get_image_prepare_material(image, face_box, fp=self.face_parser, mode=self.parsing_mode)

        from PIL import Image

        with _temporary_cwd(self.repo):
            from musetalk.utils.blending import face_seg, get_crop_box

            body = Image.fromarray(image[:, :, ::-1])
            x1, y1, x2, y2 = face_box
            crop_box, _ = get_crop_box(face_box, expand)
            x_s, y_s, x_e, y_e = crop_box

            face_large = body.crop(crop_box)
            ori_shape = face_large.size
            mask_image = face_seg(face_large, fp=self.face_parser)
            if mask_image is None:
                raise RuntimeError("MuseTalk face parsing failed for ref_image")
            mask_small = mask_image.crop((x1 - x_s, y1 - y_s, x2 - x_s, y2 - y_s))
            mask_image = Image.new("L", ori_shape, 0)
            mask_image.paste(mask_small, (x1 - x_s, y1 - y_s, x2 - x_s, y2 - y_s))

            width, height = mask_image.size
            top_boundary = int(height * upper_boundary_ratio)
            modified_mask_image = Image.new("L", ori_shape, 0)
            modified_mask_image.paste(
                mask_image.crop((0, top_boundary, width, height)),
                (0, top_boundary),
            )
            blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
            mask_array = cv2.GaussianBlur(
                np.array(modified_mask_image),
                (blur_kernel_size, blur_kernel_size),
                0,
            )
            return mask_array, tuple(int(v) for v in crop_box)

    def _closed_mouth_prediction(
        self, state: MuseTalkSessionState, feature: torch.Tensor, latent_index: int,
    ) -> np.ndarray:
        """Cache a zero-feature prediction per avatar latent, as in the local adapter.

        Zero PCM still produces Whisper embeddings. The closure reference instead
        uses zero features BEFORE positional encoding, with the same face latent.
        """
        if latent_index not in state.closed_prediction_cache:
            dtype = self.unet.model.dtype
            silent_features = self.pe(torch.zeros_like(feature).unsqueeze(0).to(device=self.device, dtype=dtype))
            latent = state.latent_cycle[latent_index].to(device=self.device, dtype=dtype)
            prediction = self.unet.model(
                latent, torch.tensor([0], device=self.device), encoder_hidden_states=silent_features,
            ).sample
            state.closed_prediction_cache[latent_index] = self.vae.decode_latents(prediction)[0].copy()
        return state.closed_prediction_cache[latent_index]

    def render_chunk(
        self,
        state: MuseTalkSessionState,
        pcm_int16: np.ndarray,
        *,
        slice_len: int,
        fps: int,
    ) -> list[np.ndarray]:
        gate_energy = (
            _compute_per_frame_energy(pcm_int16.astype(np.float32) / 32768.0, slice_len)
            if self.version == "v15" and self.silence_gate > 0.0 else None
        )
        if state.audio_context is not None and state.audio_context.size:
            audio_for_features = np.concatenate([state.audio_context, pcm_int16])
            context_samples = int(state.audio_context.shape[0])
        else:
            audio_for_features = pcm_int16
            context_samples = 0
        state.audio_context = audio_for_features[-self.audio_context_samples :].copy()

        with tempfile.NamedTemporaryFile(prefix="omnirt-musetalk-audio-", suffix=".wav") as wav:
            import soundfile as sf

            sf.write(wav.name, audio_for_features.astype(np.float32) / 32768.0, SAMPLE_RATE)
            if self.version == "v15":
                with torch.no_grad():
                    features, audio_length = self.audio_processor.get_audio_feature(
                        wav.name, weight_dtype=self.unet.model.dtype,
                    )
                    chunks = list(self.audio_processor.get_whisper_chunk(
                        features,
                        self.device,
                        self.unet.model.dtype,
                        self.whisper,
                        audio_length,
                        fps=fps,
                        audio_padding_length_left=2,
                        audio_padding_length_right=2,
                    ))
            else:
                features = self.audio_processor.audio2feat(wav.name)

        context_frames = int(round(context_samples * fps / SAMPLE_RATE))
        start_frame = max(0, context_frames)
        if self.version == "v1":
            chunks = self.audio_processor.feature2chunks(features, fps=fps)
        chunks = chunks[start_frame : start_frame + slice_len]
        if not chunks:
            raise RuntimeError("MuseTalk audio feature extraction produced zero chunks")
        while len(chunks) < slice_len:
            chunks.append(chunks[-1])
        if self.version == "v1":
            chunks = [torch.from_numpy(np.asarray(chunk)).float() for chunk in chunks]

        frames: list[np.ndarray] = []
        inference_index = 0
        timesteps = torch.tensor([0], device=self.device)
        with _temporary_cwd(self.repo):
            from musetalk.utils.blending import get_image_blending
            from musetalk.utils.utils import datagen

            gen = datagen(chunks, state.latent_cycle, batch_size=self.batch_size, device=str(self.device))
            with torch.no_grad():
                for whisper_batch, latent_batch in gen:
                    audio_feature_batch = whisper_batch.to(device=self.device, dtype=self.unet.model.dtype)
                    audio_feature_batch = self.pe(audio_feature_batch)
                    latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                    pred_latents = self.unet.model(
                        latent_batch,
                        timesteps,
                        encoder_hidden_states=audio_feature_batch,
                    ).sample
                    recon = self.vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        feature_index = inference_index
                        inference_index += 1
                        if gate_energy is not None and gate_energy[feature_index] <= self.silence_gate:
                            # datagen starts its latent cycle at zero for each chunk.
                            closed = self._closed_mouth_prediction(
                                state, chunks[feature_index], feature_index % len(state.latent_cycle),
                            )
                            amount = 1.0 - float(gate_energy[feature_index]) / self.silence_gate
                            res_frame = _blend_prediction_toward_closed_mouth(res_frame, closed, amount)
                        idx = state.frame_cursor % len(state.frame_cycle)
                        x1, y1, x2, y2 = state.face_box
                        try:
                            face = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                        except Exception:
                            continue
                        out = get_image_blending(
                            state.frame_cycle[idx].copy(),
                            face,
                            state.face_box,
                            state.mask_cycle[idx],
                            state.mask_coords_cycle[idx],
                        )
                        frames.append(out)
                        state.frame_cursor += 1
                        if len(frames) >= slice_len:
                            return frames
        if not frames:
            raise RuntimeError("MuseTalk produced zero frames for this chunk")
        while len(frames) < slice_len:
            frames.append(frames[-1].copy())
        return frames[:slice_len]


_RUNTIME: MuseTalkRuntime | None = None


def _get_runtime() -> MuseTalkRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = MuseTalkRuntime()
    return _RUNTIME


def _preload_runtime() -> None:
    raw = os.environ.get("OMNIRT_MUSETALK_PRELOAD", "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return
    LOG.info("OMNIRT_MUSETALK_PRELOAD: loading MuseTalk runtime at startup...")
    _ = _get_runtime()
    LOG.info("OMNIRT_MUSETALK_PRELOAD: startup load complete.")


async def _handler(websocket) -> None:
    frame_num, motion_frames_num, slice_len = _slice_params()
    fps = int(DEFAULT_FPS)
    expected_pcm = _audio_chunk_bytes(slice_len, fps)
    jpeg_q = int(os.environ.get("OMNIRT_MUSETALK_JPEG_QUALITY", "85"))
    jpeg_q = min(100, max(1, jpeg_q))

    session_active = False
    base_frame: np.ndarray | None = None
    state: MuseTalkSessionState | None = None
    height = width = 0
    chunk_idx = 0

    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "init":
                    chunk_idx = 0
                    base_frame, err = _decode_init_ref_image(msg)
                    if err is not None or base_frame is None:
                        await websocket.send(
                            json.dumps({"type": "error", "message": err or "ref_image failed"})
                        )
                        continue

                    h0, w0 = base_frame.shape[:2]
                    min_le = _min_long_edge_limit()
                    if min_le > 0 and max(h0, w0) < min_le:
                        base_frame = _upscale_bgr_min_long_edge(base_frame, min_le)
                    max_le = _max_long_edge_limit()
                    if max_le > 0:
                        base_frame = _downscale_bgr_max_long_edge(base_frame, max_le)

                    height, width = base_frame.shape[:2]
                    loop = asyncio.get_running_loop()
                    try:
                        runtime = _get_runtime()
                        state = await loop.run_in_executor(
                            None,
                            functools.partial(runtime.prepare_session, base_frame),
                        )
                    except Exception as exc:
                        LOG.exception("init session prepare failed: %s", exc)
                        await websocket.send(
                            json.dumps({"type": "error", "message": f"init failed: {exc}"})
                        )
                        continue

                    session_active = True

                    await websocket.send(
                        json.dumps(
                            {
                                "type": "init_ok",
                                "frame_num": frame_num,
                                "motion_frames_num": motion_frames_num,
                                "slice_len": slice_len,
                                "fps": fps,
                                "height": int(height),
                                "width": int(width),
                            }
                        )
                    )
                    LOG.info(
                        "init_ok %dx%d slice_len=%d chunk_pcm_bytes=%d | device=%s",
                        width,
                        height,
                        slice_len,
                        expected_pcm,
                        _inference_device_str(),
                    )

                elif msg_type == "close":
                    session_active = False
                    base_frame = None
                    state = None
                    chunk_idx = 0
                    await websocket.send(json.dumps({"type": "close_ok"}))

                else:
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"Unknown type {msg_type}"})
                    )

            elif isinstance(message, (bytes, bytearray)):
                raw = bytes(message)
                if not session_active or base_frame is None or state is None:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "No active session. Send init first.",
                            }
                        )
                    )
                    continue
                if len(raw) < 4 or raw[:4] != MAGIC_AUDIO:
                    await websocket.send(
                        json.dumps({"type": "error", "message": "Expected AUDI magic"})
                    )
                    continue

                pcm = raw[4:]
                if len(pcm) != expected_pcm:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": (
                                    f"Expected {expected_pcm} bytes PCM, got {len(pcm)} "
                                    f"(slice_len={slice_len}, fps={fps})"
                                ),
                            }
                        )
                    )
                    continue

                pcm_i16 = np.frombuffer(pcm, dtype=np.int16)
                t_start = time.perf_counter()
                loop = asyncio.get_running_loop()
                runtime = _get_runtime()

                def _run() -> list[np.ndarray]:
                    return runtime.render_chunk(
                        state,
                        pcm_i16,
                        slice_len=slice_len,
                        fps=fps,
                    )

                try:
                    frames_bgr = await loop.run_in_executor(None, _run)
                except Exception as exc:
                    LOG.exception("MuseTalk chunk failed: %s", exc)
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"generate failed: {exc}"})
                    )
                    continue

                jpeg_parts: list[bytes] = []
                for fb in frames_bgr:
                    ok, enc = cv2.imencode(
                        ".jpg",
                        fb,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q],
                    )
                    if not ok:
                        raise RuntimeError("JPEG encode failed")
                    jpeg_parts.append(enc.tobytes())

                vmsg = _encode_video_message(jpeg_parts)
                await websocket.send(vmsg)
                t_done = time.perf_counter()
                LOG.info(
                    "MuseTalk chunk-%d: %df total=%.3fs jpeg_q=%d",
                    chunk_idx,
                    len(frames_bgr),
                    t_done - t_start,
                    jpeg_q,
                )
                chunk_idx += 1

    except Exception as e:
        LOG.exception("handler error: %s", e)
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


async def _run_server(host: str, port: int) -> None:
    try:
        from websockets.asyncio.server import serve
    except ImportError as e:
        raise RuntimeError("pip install websockets") from e

    async with serve(_handler, host, port, max_size=50 * 1024 * 1024):
        LOG.info("MuseTalk FlashTalk-compatible WS at ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MuseTalk WebSocket server (FlashTalk protocol)")
    p.add_argument("--host", default=os.environ.get("OMNIRT_MUSETALK_HOST", "0.0.0.0"))
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("OMNIRT_MUSETALK_PORT", "8766")),
    )
    p.add_argument("--ckpt_dir", default="", help="Ignored (use OMNIRT_MUSETALK_MODELS_DIR)")
    p.add_argument("--wav2vec_dir", default="", help="Ignored")
    p.add_argument("--cpu_offload", action="store_true", help="Ignored")
    p.add_argument("--t5_quant", default=None, help="Ignored")
    p.add_argument("--t5_quant_dir", default=None, help="Ignored")
    p.add_argument("--wan_quant", default=None, help="Ignored")
    p.add_argument("--wan_quant_include", default=None, help="Ignored")
    p.add_argument("--wan_quant_exclude", default=None, help="Ignored")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args, _unknown = build_arg_parser().parse_known_args(argv)
    try:
        LOG.info("MuseTalk version=%s", _musetalk_version())
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    fn, mn, sl = _slice_params()
    LOG.info(
        "Protocol: frame_num=%d motion=%d slice_len=%d fps=%d chunk_samples=%d",
        fn,
        mn,
        sl,
        DEFAULT_FPS,
        sl * SAMPLE_RATE // DEFAULT_FPS,
    )
    inf = _inference_device_str()
    if inf.startswith("npu"):
        if not _try_import_torch_npu():
            LOG.error(
                "Inference device is %s but torch_npu could not be imported; "
                "install the CANN + torch_npu build that matches your PyTorch.",
                inf,
            )
            return 1
        try:
            if not torch.npu.is_available():  # type: ignore[union-attr]
                LOG.error(
                    "Inference device is %s but torch.npu.is_available() is False.",
                    inf,
                )
                return 1
        except Exception as exc:
            LOG.error("NPU availability check failed: %s", exc)
            return 1
    LOG.info("MuseTalk inference device=%s", inf)
    try:
        _preload_runtime()
    except Exception as exc:
        LOG.error("Startup load failed: %s", exc)
        return 1
    asyncio.run(_run_server(args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
