# sumospace/multimodal/video.py

"""
Video Processor — Local Video Analysis
========================================
Frame extraction + audio transcription.
All processing is local — no API key.

Install: pip install sumospace[video]
Dependencies: opencv-python-headless, pillow, imageio, ffmpeg-python, openai-whisper
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sumospace.multimodal.audio import AudioProcessor, TranscriptionResult


@dataclass
class VideoFrame:
    frame_number: int
    timestamp_s: float
    path: str           # path to saved frame image
    width: int = 0
    height: int = 0


@dataclass
class VideoAnalysisResult:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    frames: list[VideoFrame] = field(default_factory=list)
    transcript: TranscriptionResult | None = None
    success: bool = True
    error: str = ""

    @property
    def summary(self) -> str:
        lines = [
            f"Video: {self.path}",
            f"Duration: {self.duration_s:.1f}s | FPS: {self.fps:.1f} | "
            f"Resolution: {self.width}x{self.height}",
            f"Frames extracted: {len(self.frames)}",
        ]
        if self.transcript and self.transcript.success:
            lines.append(f"Transcript ({self.transcript.language}): {self.transcript.text[:200]}")
        return "\n".join(lines)


class VideoProcessor:
    """
    Local video frame extraction and optional audio transcription.

    Args:
        audio_model:      Whisper model for transcription ("base", "small", etc.)
        frames_per_second: How many frames to extract per second of video.
        max_frames:        Maximum total frames to extract.
        output_dir:        Where to save extracted frames (temp dir if None).
        transcribe_audio:  Whether to extract + transcribe the audio track.
    """

    SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}

    def __init__(
        self,
        audio_model: str = "base",
        frames_per_second: float = 0.5,    # 1 frame every 2 seconds by default
        max_frames: int = 100,
        output_dir: str | None = None,
        transcribe_audio: bool = True,
    ):
        self.audio_model = audio_model
        self.frames_per_second = frames_per_second
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.transcribe_audio = transcribe_audio
        self._audio_processor = AudioProcessor(model=audio_model)

    async def analyze(self, path: str | Path) -> VideoAnalysisResult:
        """Extract frames and optionally transcribe audio from a video file."""
        path = Path(path)
        if not path.exists():
            return VideoAnalysisResult(
                path=str(path), duration_s=0, fps=0, width=0, height=0,
                success=False, error=f"File not found: {path}",
            )
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return VideoAnalysisResult(
                path=str(path), duration_s=0, fps=0, width=0, height=0,
                success=False, error=f"Unsupported format: {path.suffix}",
            )

        try:
            import cv2
        except ImportError:
            raise ImportError("pip install sumospace[video]")

        loop = asyncio.get_event_loop()

        # Extract frames in executor (blocking IO)
        frames, metadata = await loop.run_in_executor(
            None, lambda: self._extract_frames(path)
        )

        result = VideoAnalysisResult(
            path=str(path),
            duration_s=metadata["duration_s"],
            fps=metadata["fps"],
            width=metadata["width"],
            height=metadata["height"],
            frames=frames,
        )

        # Transcribe audio
        if self.transcribe_audio:
            audio_path = await self._extract_audio(path)
            if audio_path:
                result.transcript = await self._audio_processor.transcribe(audio_path)
                try:
                    Path(audio_path).unlink()
                except Exception:
                    pass

        result.success = True
        return result

    def _extract_frames(
        self, path: Path
    ) -> tuple[list[VideoFrame], dict[str, Any]]:
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = frame_count / fps

        frame_interval = max(1, int(fps / self.frames_per_second))
        output_dir = Path(self.output_dir) if self.output_dir else Path(tempfile.mkdtemp())
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        frame_idx = 0
        extracted = 0

        while extracted < self.max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps
            frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(VideoFrame(
                frame_number=frame_idx,
                timestamp_s=timestamp,
                path=str(frame_path),
                width=width,
                height=height,
            ))
            frame_idx += frame_interval
            extracted += 1

        cap.release()
        return frames, {
            "duration_s": duration_s,
            "fps": fps,
            "width": width,
            "height": height,
        }

    async def _extract_audio(self, video_path: Path) -> str | None:
        """Extract audio track from video to a temp WAV file using ffmpeg."""
        try:
            import ffmpeg
        except ImportError:
            return None

        audio_path = tempfile.mktemp(suffix=".wav")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: (
                    ffmpeg
                    .input(str(video_path))
                    .output(audio_path, acodec="pcm_s16le", ac=1, ar="16000")
                    .overwrite_output()
                    .run(quiet=True)
                ),
            )
            return audio_path
        except Exception:
            return None

    def frames_to_chunks(
        self, result: VideoAnalysisResult
    ) -> list[dict[str, Any]]:
        """
        Convert frames + transcript to ingestion-ready chunks.
        Each chunk pairs a frame timestamp with the transcript text around that time.
        """
        chunks = []

        if result.transcript and result.transcript.segments:
            seg_map = {seg["start"]: seg["text"] for seg in result.transcript.segments}

        for frame in result.frames:
            text_parts = [f"[Video frame at {frame.timestamp_s:.1f}s from {result.path}]"]

            if result.transcript and result.transcript.segments:
                # Find transcript text within ±5s of this frame
                nearby = [
                    seg["text"] for seg in result.transcript.segments
                    if abs(seg.get("start", 0) - frame.timestamp_s) <= 5
                ]
                if nearby:
                    text_parts.append("Transcript nearby: " + " ".join(nearby))

            chunks.append({
                "text": "\n".join(text_parts),
                "frame_path": frame.path,
                "timestamp_s": frame.timestamp_s,
                "source": result.path,
                "type": "video_frame",
            })

        return chunks
