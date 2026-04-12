"""Faster-Whisper 语音转文字（中文，GPU，int8 量化）"""

import glob
import os
from pathlib import Path

from loguru import logger

CUDA_FALLBACK_DIRS = [
    # 修改为你的 NVIDIA CUDA 库路径
    # "/path/to/your/venv/lib/python3.12/site-packages/nvidia",
]


def _ensure_cuda_libs() -> None:
    """WSL 环境下 ctranslate2 可能找不到 CUDA 库，提前注入 LD_LIBRARY_PATH。"""
    if os.environ.get("_CUDA_LIBS_INJECTED"):
        return

    extra: list[str] = []
    for base in CUDA_FALLBACK_DIRS:
        if Path(base).exists():
            for lib_dir in glob.glob(f"{base}/*/lib"):
                if Path(lib_dir).exists():
                    extra.append(lib_dir)

    if extra:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra) + (
            f":{existing}" if existing else ""
        )
        logger.debug("注入 CUDA 库路径: {} 条", len(extra))

    os.environ["_CUDA_LIBS_INJECTED"] = "1"


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        _ensure_cuda_libs()

        from faster_whisper import WhisperModel

        logger.info(
            "加载 Whisper 模型: size={}, device={}, compute_type={}",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.success("Whisper 模型加载完成")

    def transcribe(self, audio_path: str) -> dict:
        self._load_model()

        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio}")

        logger.info("开始转录: {}", audio.name)

        segments_iter, info = self._model.transcribe(
            str(audio),
            language="zh",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        segments: list[dict] = []
        full_text_parts: list[str] = []

        for seg in segments_iter:
            segment_dict = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            segments.append(segment_dict)
            full_text_parts.append(seg.text.strip())
            logger.debug("[{:.1f}s - {:.1f}s] {}", seg.start, seg.end, seg.text.strip())

        full_text = "".join(full_text_parts)
        logger.success("转录完成: {} 个片段, 共 {} 字", len(segments), len(full_text))

        return {"text": full_text, "segments": segments}
