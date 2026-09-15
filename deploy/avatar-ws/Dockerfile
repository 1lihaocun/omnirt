# syntax=docker/dockerfile:1.7

# Reuse the two already validated images. Build without network access so the
# CUDA/PyTorch and TensorRT dependencies cannot drift during a source release.
ARG QUICKTALK_BASE_IMAGE=omnirt-quicktalk:cu124
ARG FASTERLIVEPORTRAIT_BASE_IMAGE=shaoguo/faster_liveportrait:v3
FROM ${FASTERLIVEPORTRAIT_BASE_IMAGE} AS flp
FROM ${QUICKTALK_BASE_IMAGE}

COPY --from=flp /root/miniconda3/lib/python3.10/site-packages /opt/flp-deps

# Keep PyTorch/ONNX Runtime from the QuickTalk venv ahead of the older
# FasterLivePortrait packages. Only packages absent from the venv are read
# from /opt/flp-deps.
RUN python -c 'from pathlib import Path; import sysconfig; Path(sysconfig.get_path("purelib"), "fasterliveportrait-compat.pth").write_text("/opt/flp-deps\n")'

COPY --from=flp /opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer.so.8.6.1 /opt/flp-compat/lib/
COPY --from=flp /opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so.8.6.1 /opt/flp-compat/lib/
COPY --from=flp /opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvonnxparser.so.8.6.1 /opt/flp-compat/lib/
COPY --from=flp /opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvparsers.so.8.6.1 /opt/flp-compat/lib/
COPY --from=flp /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.11.10.3.66 /opt/flp-compat/lib/
COPY --from=flp /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.11.10.3.66 /opt/flp-compat/lib/
COPY --from=flp /lib/x86_64-linux-gnu/libcudnn.so.8.5.0 /opt/flp-compat/lib/
RUN cd /opt/flp-compat/lib && \
    ln -s libnvinfer.so.8.6.1 libnvinfer.so.8 && \
    ln -s libnvinfer_plugin.so.8.6.1 libnvinfer_plugin.so.8 && \
    ln -s libnvonnxparser.so.8.6.1 libnvonnxparser.so.8 && \
    ln -s libnvparsers.so.8.6.1 libnvparsers.so.8 && \
    ln -s libcublas.so.11.10.3.66 libcublas.so.11 && \
    ln -s libcublasLt.so.11.10.3.66 libcublasLt.so.11 && \
    ln -s libcudnn.so.8.5.0 libcudnn.so.8

COPY --from=flp /lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.5.0 /opt/flp-compat/lib/
COPY --from=flp /lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.5.0 /opt/flp-compat/lib/
COPY --from=flp /lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.5.0 /opt/flp-compat/lib/
COPY --from=flp /lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.5.0 /opt/flp-compat/lib/
COPY --from=flp /lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.5.0 /opt/flp-compat/lib/
COPY --from=flp /lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.5.0 /opt/flp-compat/lib/
RUN cd /opt/flp-compat/lib && \
    for name in ops_infer cnn_infer adv_infer ops_train cnn_train adv_train; do \
      ln -s libcudnn_${name}.so.8.5.0 libcudnn_${name}.so.8; \
    done

ENV LD_LIBRARY_PATH=/opt/flp-compat/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTHONPATH=/opt/omnirt/src:/opt/FasterLivePortrait
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NO_ALBUMENTATIONS_UPDATE=1

# This named build context is the independently pinned FasterLivePortrait
# checkout. Checkpoints remain outside the image and are mounted read-only.
COPY --from=flp_src /configs /opt/FasterLivePortrait/configs
COPY --from=flp_src /src /opt/FasterLivePortrait/src
COPY --from=flp_src /assets /opt/FasterLivePortrait/assets

COPY pyproject.toml README.md LICENSE /opt/omnirt/
COPY src/ /opt/omnirt/src/
COPY deploy/avatar-ws/start.sh /opt/omnirt/deploy/avatar-ws/start.sh
RUN chmod 755 /opt/omnirt/deploy/avatar-ws/start.sh && \
    pip install --no-deps --no-build-isolation /opt/omnirt && \
    python -c 'import sys, torch, onnxruntime, tensorrt; from omnirt.server.avatar_app import create_avatar_app; assert torch.__version__ == "2.6.0+cu124"; assert "CUDAExecutionProvider" in onnxruntime.get_available_providers(); assert "/opt/flp-deps" in sys.path; print("shared avatar imports: OK")'

ARG OMNIRT_REVISION
ARG FASTERLIVEPORTRAIT_REVISION
ARG QUICKTALK_BASE_IMAGE_ID
ARG FASTERLIVEPORTRAIT_BASE_IMAGE_ID
LABEL org.opencontainers.image.revision="${OMNIRT_REVISION}" \
      com.omnirt.fasterliveportrait.revision="${FASTERLIVEPORTRAIT_REVISION}" \
      com.omnirt.quicktalk.base-image-id="${QUICKTALK_BASE_IMAGE_ID}" \
      com.omnirt.fasterliveportrait.base-image-id="${FASTERLIVEPORTRAIT_BASE_IMAGE_ID}"

ENTRYPOINT ["/opt/omnirt/deploy/avatar-ws/start.sh"]
CMD ["omnirt", "serve-avatar-ws", "--host", "0.0.0.0", "--port", "9000", "--backend", "cuda"]
