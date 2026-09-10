# MuseTalk WebSocket（FlashTalk 协议）

与 SoulX FlashTalk / OmniRT Wav2Lip **同一套 WebSocket 协议**（`AUDI` / `VIDX`、`init` / `init_ok`），OpenTalking 可配置为 **remote flashtalk** 连到本服务。

推理与 MuseTalk 源码加载都在 **OmniRT** 侧完成；OpenTalking 只负责选择模型、编排会话并连接本服务。

本目录只保留必要文件：

| 文件 | 说明 |
|------|------|
| `musetalk_ws_server.py` | WebSocket 服务入口 |
| `requirements-musetalk-ascend.txt` | **昇腾 NPU** 依赖（torch-npu，与 FlashTalk/Wav2Lip 钉版本一致） |
| `requirements-musetalk-gpu.txt` | **NVIDIA GPU / CUDA** 依赖 |
| `README.md` | 本文档 |

启动脚本在仓库根目录：**`scripts/start_musetalk_ws.sh`**（不在本目录重复放一份）。

**昇腾 / GPU 部署与用户指南**（与 [`wav2lip_ws.md`](../../docs/user_guide/serving/wav2lip_ws.md) 同级）：[`../../docs/user_guide/serving/musetalk_ws.md`](../../docs/user_guide/serving/musetalk_ws.md)。

---

## 环境安装

推荐使用 OmniRT runtime 管理 MuseTalk 源码与 Python 环境：

```bash
cd /path/to/omnirt
omnirt runtime install musetalk --device cuda
```

该命令会把官方 MuseTalk 仓克隆到 `${OMNIRT_HOME}/model-repos/MuseTalk`，并创建 `${OMNIRT_HOME}/runtimes/musetalk/<device>/venv`。不需要在 OpenTalking 的 quickstart env 中指定 MuseTalk repo。

### 昇腾（当前主要适配）

1. 安装 CANN / 驱动，配置 `PIP_EXTRA_INDEX_URL` 指向华为 **torch / torch-npu** 与 CANN 匹配的 wheel 源。
2. 创建 venv 后安装：

```bash
cd /path/to/omnirt
pip install -r model_backends/musetalk/requirements-musetalk-ascend.txt
```

3. 运行前 `source` CANN 环境（或由 `start_musetalk_ws.sh` 自动尝试 `set_env.sh`）。

若已有 **OmniRT FlashTalk Ascend** 的 venv（例如 `.omnirt/runtimes/flashtalk/ascend/venv`），其中通常已含 torch/torch_npu/diffusers，可 **`pip install`** 本文件中仍缺的包（如 `openai-whisper`、`torchaudio`、`pydantic-settings`），无需重装整套 torch。

### NVIDIA GPU

```bash
pip install -r model_backends/musetalk/requirements-musetalk-gpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

CUDA runtime 当前跟随 MuseTalk 官方 1.5 依赖组合：Python 3.10、PyTorch 2.0.1 + cu118、diffusers 0.30.2、transformers 4.39.2、accelerate 0.28.0。不要在这个 runtime 里直接升级到新的 PyTorch / diffusers / transformers；较新版本可能导入 `torch.xpu` 或 `torch.float8_e4m3fn`，与 torch 2.0.1 不兼容。

---

## 权重目录（`OMNIRT_MUSETALK_MODELS_DIR`，默认 `<omnirt>/models`）

`OMNIRT_MUSETALK_VERSION` 默认 `v1`，沿用现有权重和 `Audio2Feature`。设置为 `v15` 时使用独立 UNet 权重，以及 `AudioProcessor` + `transformers.WhisperModel`；不加载旧版 `Audio2Feature` 或 `whisper/tiny.pt`。

v1.5 适配参考官方 commit [`0a89dec45a0192b824e3cf4daf96c239440c5ed8`](https://github.com/TMElyralab/MuseTalk/tree/0a89dec45a0192b824e3cf4daf96c239440c5ed8)，重点为 `scripts/realtime_inference.py`、`musetalk/utils/audio_processor.py`、`musetalk/utils/utils.py`。`OMNIRT_MUSETALK_REPO` 应指向此提交的源码 checkout。

| 相对路径 | 说明 |
|----------|------|
| `musetalk/pytorch_model.bin`、`musetalk/musetalk.json` | **v1** UNet |
| `musetalkV15/unet.pth`、`musetalkV15/musetalk.json` | **v15** UNet |
| `sd-vae-ft-mse/` | 两版本共用 VAE：`config.json` + `diffusion_pytorch_model.bin`（当前布局检查要求 `.bin`；可额外提供 `.safetensors`） |
| `whisper/tiny.pt` | **v1** 使用 OpenAI `openai-whisper` 官方 tiny 检查点（约 72MB），不要用 HuggingFace `pytorch_model.bin` 改名顶替 |
| `dwpose/dw-ll_ucoco_384.pth` | DWPose |
| `face-parse-bisenet/79999_iter.pth` | BiSeNet；同目录常配 `resnet18-5c106cde.pth`（PyTorch 官方 ResNet18） |

**v15 的 HF Whisper 目录**由 `OMNIRT_MUSETALK_WHISPER_DIR` 单独指定，默认绝对路径 `/models/whisper-hf`，不随 `OMNIRT_MUSETALK_MODELS_DIR` 改变。使用 Hugging Face `openai/whisper-tiny` 模型目录，至少包含：

```text
/models/whisper-hf/
├── config.json
├── preprocessor_config.json
└── model.safetensors          # 或 pytorch_model.bin
```

v15 不要求保留 v1 的 `musetalk/` 和 `whisper/tiny.pt`。共用的 VAE、DWPose、FaceParsing 权重仍需齐备。缺少 v15 UNet、HF Whisper 配置或权重时，服务会报告版本及缺失路径。

官方 `tiny.pt` 可用已安装 `openai-whisper` 的 Python 按包内 URL 下载并校验 SHA256；或从
`https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt`
下载，SHA256 应为文件名中的 `65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9`。

`sd-vae-ft-mse/` 使用 Hugging Face 官方 `stabilityai/sd-vae-ft-mse` Diffusers 格式文件（保留现有布局检查要求的 `.bin`）：

```bash
mkdir -p models/sd-vae-ft-mse
wget -O models/sd-vae-ft-mse/config.json \
  https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json
wget -O models/sd-vae-ft-mse/diffusion_pytorch_model.bin \
  https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin
wget -O models/sd-vae-ft-mse/diffusion_pytorch_model.safetensors \
  https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
```

若直连 Hugging Face 受限，可将域名替换为 `https://hf-mirror.com`。只有 `diffusion_pytorch_model.bin` 时 Diffusers 仍可回退加载，但会提示 unsafe serialization；生产部署建议补齐 `.safetensors`。

Face-parse 可从 HF 镜像等获取与 MuseTalk 脚本一致的 `79999_iter.pth`；**目录名**须为 **`face-parse-bisenet`**（与 OpenTalking 一致）。

---

## 启动

```bash
cd /path/to/omnirt
export OMNIRT_MUSETALK_PYTHON=$OMNIRT_HOME/runtimes/musetalk/cuda/venv/bin/python
export OMNIRT_MUSETALK_REPO=$OMNIRT_HOME/model-repos/MuseTalk
export OMNIRT_MUSETALK_MODELS_DIR=/path/to/omnirt/models # 可选
bash scripts/start_musetalk_ws.sh
```

默认监听 **`0.0.0.0:8766`**（与 Wav2Lip 常用 8765 错开）。后台：`OMNIRT_MUSETALK_BACKGROUND=1 bash scripts/start_musetalk_ws.sh --background`。

OpenTalking：`OPENTALKING_FLASHTALK_MODE=remote`，`OPENTALKING_FLASHTALK_WS_URL=ws://<host>:8766`。

启用 v15 时，在启动前增加：

```bash
export OMNIRT_MUSETALK_VERSION=v15
export OMNIRT_MUSETALK_MODELS_DIR=/models
export OMNIRT_MUSETALK_WHISPER_DIR=/models/whisper-hf
```

启动日志明确输出 `MuseTalk version=v1` 或 `MuseTalk version=v15`。`init/init_ok`、`AUDI`、`VIDX`、默认 25fps、chunk streaming、会话预处理及 JPEG 输出均沿用现有协议。

---

## 常用环境变量

| 变量 | 含义 |
|------|------|
| `OMNIRT_MUSETALK_HOST` / `PORT` | 绑定地址 / 端口 |
| `OMNIRT_MUSETALK_REPO` | MuseTalk 源码 checkout；默认 `${OMNIRT_HOME}/model-repos/MuseTalk` |
| `OMNIRT_MUSETALK_MODELS_DIR` | 权重根目录 |
| `OMNIRT_MUSETALK_VERSION` | `v1`（默认）或 `v15` |
| `OMNIRT_MUSETALK_WHISPER_DIR` | v15 的 HF whisper-tiny 目录，默认 `/models/whisper-hf` |
| `OMNIRT_MUSETALK_BBOX_SHIFT` | v1 使用该值（默认 `0`）；v15 固定 `0` |
| `OMNIRT_MUSETALK_EXTRA_MARGIN` | v15 裁脸下边界增加的像素数，默认 `10`，不超过图像高度 |
| `OMNIRT_MUSETALK_PARSING_MODE` | v15 的官方融合解析模式，默认 `jaw`（也支持 `neck`、`raw`） |
| `OMNIRT_MUSETALK_LEFT_CHEEK_WIDTH` / `OMNIRT_MUSETALK_RIGHT_CHEEK_WIDTH` | v15 的 FaceParsing 参数，默认各 `90` |
| `OMNIRT_MUSETALK_DEVICE` | `auto` / `npu` / `cuda` / `cpu` |
| `OMNIRT_MUSETALK_MAX_LONG_EDGE` | `init` 里 ref 图最长边上限（默认 768；`0` 表示不缩放） |
| `OMNIRT_MUSETALK_JPEG_QUALITY` | 输出 VIDX JPEG 质量 |

---

## 说明与排错

- **Ascend 目录 owner 警告**：toolkit 若 root 安装、普通用户运行，可能警告属主不一致，一般不影响推理。
- **Whisper `tiny.pt` 只有几百字节且为 XML**：下载错误或误用 HF 权重，按上文替换官方 `tiny.pt`。
- **v15 缺少 `mmcv` / `mmpose` 等预处理依赖**：补齐官方 landmark 预处理依赖；v15 会明确报错，v1 保留现有 SFD fallback。
- **嘴型与底图错位**：OpenTalking `composer` 须对 MuseTalk 使用 **infer 框**贴回（若你自行改过分支，请保持与 upstream 一致）。
- **参考图「只有一块脸在动」**：MuseTalk 本身只在人脸区域生成再贴回；远景小脸会更像贴片，可换近景正脸或调整 `OMNIRT_MUSETALK_MAX_LONG_EDGE`。
