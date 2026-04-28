# sumospace/multimodal/audio.py

"""
Audio Processor — Local Whisper ASR
=====================================
Transcribes audio files using OpenAI Whisper running LOCALLY.
This is the open-source model package — NOT the OpenAI API.
Zero API key. Zero internet at inference time.

Install: pip install sumospace[audio]
Models: tiny (~75MB), base (~145MB), small (~460MB), medium (~1.5GB), large (~3GB)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionResult:
    path: str
    text: str
    language: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    model_used: str = ""
    success: bool = True
    error: str = ""


class AudioProcessor:
    """
    Local audio transcription and analysis using OpenAI Whisper.

    Args:
        model:  Whisper model size. Tradeoff: speed vs accuracy.
                "tiny"   — fastest, lowest accuracy (~75MB)
                "base"   — good for English, fast (~145MB)  ← default
                "small"  — better accuracy, still fast (~460MB)
                "medium" — high accuracy, moderate speed (~1.5GB)
                "large"  — best accuracy, slow (~3GB)
        device: "cpu" | "cuda" | "mps" (auto-detected if not set)
        language: Force transcription language (None = auto-detect)
    """

    SUPPORTED_FORMATS = {".mp3", ".mp4", ".wav", ".flac", ".m4a",
                         ".ogg", ".webm", ".aac", ".wma"}

    def __init__(
        self,
        model: str = "base",
        device: str | None = None,
        language: str | None = None,
    ):
        self.model_name = model
        self.language = language
        self._device = device
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import whisper
            import torch
        except ImportError:
            raise ImportError(
                "Whisper not installed. Run: pip install sumospace[audio]\n"
                "Note: openai-whisper is the LOCAL open-source model, not the API."
            )

        device = self._device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._model = whisper.load_model(self.model_name, device=device)
        self._device_used = device

    async def transcribe(self, path: str | Path) -> TranscriptionResult:
        """Transcribe an audio file to text."""
        path = Path(path)
        if not path.exists():
            return TranscriptionResult(
                path=str(path), text="", success=False,
                error=f"File not found: {path}",
            )
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return TranscriptionResult(
                path=str(path), text="", success=False,
                error=f"Unsupported format: {path.suffix}. Supported: {self.SUPPORTED_FORMATS}",
            )

        self._ensure_model()

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    str(path),
                    language=self.language,
                    verbose=False,
                ),
            )
            return TranscriptionResult(
                path=str(path),
                text=result.get("text", "").strip(),
                language=result.get("language", ""),
                segments=result.get("segments", []),
                model_used=self.model_name,
                success=True,
            )
        except Exception as e:
            return TranscriptionResult(
                path=str(path), text="", success=False, error=str(e),
            )

    async def transcribe_many(
        self,
        paths: list[str | Path],
        concurrency: int = 2,
    ) -> list[TranscriptionResult]:
        """Transcribe multiple audio files."""
        sem = asyncio.Semaphore(concurrency)

        async def _transcribe_one(p):
            async with sem:
                return await self.transcribe(p)

        return await asyncio.gather(*[_transcribe_one(p) for p in paths])

    async def get_duration(self, path: str | Path) -> float:
        """Get audio duration in seconds without full transcription."""
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return info.duration
        except ImportError:
            try:
                import wave
                with wave.open(str(path)) as wf:
                    return wf.getnframes() / wf.getframerate()
            except Exception:
                return 0.0

    def to_chunks(
        self,
        result: TranscriptionResult,
        chunk_duration_s: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Split a transcription into time-aligned chunks for ingestion.
        Each chunk corresponds to approximately chunk_duration_s of audio.
        """
        if not result.segments:
            return [{"text": result.text, "start": 0, "end": 0, "source": result.path}]

        chunks = []
        current_text = []
        current_start = 0.0
        current_duration = 0.0

        for seg in result.segments:
            seg_duration = seg.get("end", 0) - seg.get("start", 0)
            if current_duration + seg_duration > chunk_duration_s and current_text:
                chunks.append({
                    "text": " ".join(current_text),
                    "start": current_start,
                    "end": seg.get("start", 0),
                    "source": result.path,
                })
                current_text = []
                current_start = seg.get("start", 0)
                current_duration = 0.0
            current_text.append(seg.get("text", "").strip())
            current_duration += seg_duration

        if current_text:
            chunks.append({
                "text": " ".join(current_text),
                "start": current_start,
                "end": result.segments[-1].get("end", 0),
                "source": result.path,
            })

        return chunks
