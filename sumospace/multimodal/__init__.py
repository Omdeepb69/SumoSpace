# sumospace/multimodal/__init__.py

"""
Multimodal Subsystem
=====================
All processors are fully local — no API keys required.

Audio:  openai-whisper (local ASR)    → pip install sumospace[audio]
Video:  opencv + whisper              → pip install sumospace[video]

Usage:
    from sumospace.multimodal.audio import AudioProcessor
    from sumospace.multimodal.video import VideoProcessor
"""

from __future__ import annotations


def _require(extra: str, package: str):
    raise ImportError(
        f"Package '{package}' not installed.\n"
        f"Run: pip install sumospace[{extra}]"
    )
