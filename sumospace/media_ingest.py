# sumospace/multimodal_ingest.py
"""
Multimodal ingestion pipeline.
Routes each file to the correct loader based on extension.
Stores embeddings in separate ChromaDB collections per modality.
Tracks ingested files by hash to enable incremental re-ingest.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass, field

import chromadb

from sumospace.embedders import TextEmbedder, CLIPEmbedder, WhisperTranscriber, BLIPCaptioner
from sumospace.settings import SumoSettings
from sumospace.exceptions import IngestError

# File routing
TEXT_EXT  = {".txt",".md",".py",".js",".ts",".jsx",".tsx",".java",
             ".cpp",".c",".h",".go",".rs",".rb",".php",".swift",
             ".kt",".yaml",".yml",".toml",".json",".xml",".html",
             ".css",".sh",".bash",".env",".csv",".rst"}
IMAGE_EXT = {".jpg",".jpeg",".png",".gif",".bmp",".webp",".tiff",".tif",".heic"}
AUDIO_EXT = {".mp3",".wav",".flac",".aac",".ogg",".m4a",".wma",".opus",".aiff"}
VIDEO_EXT = {".mp4",".avi",".mov",".mkv",".webm",".flv",".wmv",".m4v",".mpeg"}
PDF_EXT   = {".pdf"}


@dataclass
class IngestResult:
    source: str
    modality: str
    chunks_added: int
    skipped: bool               # True if file unchanged since last ingest
    duration_ms: float
    error: str = ""


class MultimodalIngestor:
    """
    Ingest any file type into separate ChromaDB collections.
    Collections:
        sumo_text   — text, code, PDFs, audio transcripts (text embeddings)
        sumo_image  — images, video frames (CLIP embeddings)
        sumo_audio  — audio semantic chunks (text embeddings of transcripts)
        sumo_video  — video metadata linking frames + audio (text embeddings)
    """

    def __init__(self, settings: SumoSettings):
        self._settings = settings
        self._db_path  = Path(settings.chroma_base) / "multimodal"
        self._idx_path = Path(settings.chroma_base) / "multimodal_index.json"
        self._db_path.mkdir(parents=True, exist_ok=True)

        # ChromaDB client
        self._client = chromadb.PersistentClient(path=str(self._db_path))

        # Collections — created lazily
        self._cols: dict[str, chromadb.Collection] = {}

        # Embedders — lazy loaded
        self._text_embedder:  TextEmbedder       | None = None
        self._clip_embedder:  CLIPEmbedder        | None = None
        self._transcriber:    WhisperTranscriber  | None = None
        self._captioner:      BLIPCaptioner       | None = None

        # Hash index for incremental ingest
        self._index: dict[str, dict] = self._load_index()

    # ── Index management ──────────────────────────────────────────────────────
    def _load_index(self) -> dict:
        if self._idx_path.exists():
            return json.loads(self._idx_path.read_text())
        return {}

    def _save_index(self):
        self._idx_path.write_text(json.dumps(self._index, indent=2))

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _is_changed(self, path: Path) -> bool:
        current_hash = self._file_hash(path)
        stored = self._index.get(str(path), {})
        return stored.get("hash") != current_hash

    def _mark_ingested(self, path: Path, modality: str, chunks: int):
        self._index[str(path)] = {
            "hash":     self._file_hash(path),
            "modality": modality,
            "chunks":   chunks,
            "ingested_at": time.time(),
        }
        self._save_index()

    # ── Collection access ─────────────────────────────────────────────────────
    def _col(self, name: str) -> chromadb.Collection:
        if name not in self._cols:
            self._cols[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._cols[name]

    # ── Embedder access ───────────────────────────────────────────────────────
    def _text(self) -> TextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder(self._settings.embedding_model)
        return self._text_embedder

    def _clip(self) -> CLIPEmbedder:
        if self._clip_embedder is None:
            self._clip_embedder = CLIPEmbedder(self._settings.clip_model)
        return self._clip_embedder

    def _whisper(self) -> WhisperTranscriber:
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber(
                model_size=self._settings.whisper_model,
                use_faster=self._settings.whisper_use_faster,
            )
        return self._transcriber

    def _blip(self) -> BLIPCaptioner:
        if self._captioner is None:
            self._captioner = BLIPCaptioner()
        return self._captioner

    # ── Public API ────────────────────────────────────────────────────────────
    def ingest_path(self, path: str, force: bool = False) -> list[IngestResult]:
        """
        Ingest a file or directory.
        Recursively processes directories.
        Skips unchanged files unless force=True.
        """
        p = Path(path)
        if p.is_dir():
            results = []
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    results.append(self._ingest_file(child, force))
            # Cleanup deleted files
            self._cleanup_deleted()
            return results
        else:
            result = self._ingest_file(p, force)
            return [result]

    def _ingest_file(self, path: Path, force: bool) -> IngestResult:
        start = time.monotonic()
        ext = path.suffix.lower()

        # Determine modality
        if ext in TEXT_EXT:
            modality = "text"
        elif ext in PDF_EXT:
            modality = "text"
        elif ext in IMAGE_EXT:
            modality = "image"
        elif ext in AUDIO_EXT:
            modality = "audio"
        elif ext in VIDEO_EXT:
            modality = "video"
        else:
            return IngestResult(
                source=str(path), modality="unknown",
                chunks_added=0, skipped=True,
                duration_ms=0, error=f"Unsupported extension: {ext}"
            )

        # Skip unchanged files
        if not force and not self._is_changed(path):
            return IngestResult(
                source=str(path), modality=modality,
                chunks_added=0, skipped=True,
                duration_ms=(time.monotonic()-start)*1000,
            )

        try:
            chunks = 0
            if modality == "text":
                chunks = self._ingest_text(path)
            elif modality == "image":
                chunks = self._ingest_image(path)
            elif modality == "audio":
                chunks = self._ingest_audio(path)
            elif modality == "video":
                chunks = self._ingest_video(path)

            self._mark_ingested(path, modality, chunks)
            return IngestResult(
                source=str(path), modality=modality,
                chunks_added=chunks, skipped=False,
                duration_ms=(time.monotonic()-start)*1000,
            )
        except Exception as e:
            return IngestResult(
                source=str(path), modality=modality,
                chunks_added=0, skipped=False,
                duration_ms=(time.monotonic()-start)*1000,
                error=str(e),
            )

    # ── Text ingestor ─────────────────────────────────────────────────────────
    def _ingest_text(self, path: Path) -> int:
        """Chunk text, embed with sentence-transformers, store in sumo_text."""
        if path.suffix.lower() == ".pdf":
            text = self._extract_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")

        if not text.strip():
            return 0

        chunks = self._chunk_text(text, chunk_size=512, overlap=64)
        embeddings = self._text().embed(chunks)

        col = self._col("sumo_text")
        ids, docs, metas, embeds = [], [], [], []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"text_{path.stem}_{self._file_hash(path)}_{i}"
            ids.append(chunk_id)
            docs.append(chunk)
            embeds.append(emb)
            metas.append({
                "source":      str(path),
                "modality":    "text",
                "chunk_index": i,
                "total_chunks":len(chunks),
                "file_ext":    path.suffix,
                "filename":    path.name,
            })

        # Delete old chunks from this file before re-inserting
        try:
            col.delete(where={"source": str(path)})
        except Exception:
            pass

        col.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
        return len(chunks)

    # ── Image ingestor ────────────────────────────────────────────────────────
    def _ingest_image(self, path: Path) -> int:
        """
        Embed image with CLIP → stored in sumo_image (512-dim).
        Optionally generate caption with BLIP → also stored in sumo_text
        so text queries can find images cross-modally.
        """
        from PIL import Image
        img = Image.open(path).convert("RGB")
        width, height = img.size

        # CLIP embedding for image search
        clip_emb = self._clip().embed_image(img)

        # Optional: generate caption for cross-modal text search
        caption = ""
        if self._settings.image_generate_caption:
            caption = self._blip().caption(str(path))

        chunk_id = f"image_{path.stem}_{self._file_hash(path)}"

        # Store in image collection with CLIP embedding
        col_img = self._col("sumo_image")
        try:
            col_img.delete(where={"source": str(path)})
        except Exception:
            pass
        col_img.add(
            ids=[chunk_id],
            documents=[caption or path.name],
            embeddings=[clip_emb],
            metadatas=[{
                "source":   str(path),
                "modality": "image",
                "width":    width,
                "height":   height,
                "format":   img.format or path.suffix,
                "caption":  caption,
                "filename": path.name,
            }]
        )

        # If captioned, also store in text collection for cross-modal search
        if caption:
            text_emb = self._text().embed_one(caption)
            col_txt = self._col("sumo_text")
            col_txt.add(
                ids=[f"imgcaption_{chunk_id}"],
                documents=[f"[IMAGE CAPTION] {caption}"],
                embeddings=[text_emb],
                metadatas=[{
                    "source":      str(path),
                    "modality":    "image",
                    "type":        "caption",
                    "caption":     caption,
                    "filename":    path.name,
                }]
            )

        return 1

    # ── Audio ingestor ────────────────────────────────────────────────────────
    def _ingest_audio(self, path: Path) -> int:
        """
        Transcribe audio with Whisper → chunk transcript →
        embed with sentence-transformers → store in sumo_audio.
        Each chunk has timestamps so retrieval can point to the exact moment.
        """
        chunks = self._whisper().transcribe_chunks(
            str(path),
            chunk_seconds=self._settings.audio_chunk_seconds,
        )

        if not chunks:
            # Fallback: transcribe as one block
            full = self._whisper().transcribe(str(path))
            if not full.strip():
                return 0
            chunks = [(0.0, 0.0, full)]

        texts = [text for _, _, text in chunks]
        embeddings = self._text().embed(texts)

        col = self._col("sumo_audio")
        try:
            col.delete(where={"source": str(path)})
        except Exception:
            pass

        ids, docs, metas, embeds = [], [], [], []
        for i, ((start_s, end_s, text), emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"audio_{path.stem}_{self._file_hash(path)}_{i}"
            ids.append(chunk_id)
            docs.append(text)
            embeds.append(emb)
            metas.append({
                "source":   str(path),
                "modality": "audio",
                "start_s":  start_s,
                "end_s":    end_s,
                "chunk_index": i,
                "filename": path.name,
            })

        col.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
        return len(chunks)

    # ── Video ingestor ────────────────────────────────────────────────────────
    def _ingest_video(self, path: Path) -> int:
        """
        Extract video frames → embed each with CLIP → store in sumo_image.
        Extract audio track → transcribe → store in sumo_audio.
        Store video metadata in sumo_video linking both.
        """
        import cv2
        import tempfile as tf
        import os

        total_chunks = 0

        # ── Extract and embed frames ──────────────────────────────────────────
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_interval = max(1, int(fps / self._settings.video_fps_sample))
        max_frames = self._settings.video_max_frames

        frames_to_embed = []
        frame_numbers   = []
        timestamps      = []

        frame_num = 0
        while len(frames_to_embed) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % frame_interval == 0:
                # Convert BGR→RGB for PIL/CLIP
                import numpy as np
                from PIL import Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames_to_embed.append(pil_frame)
                frame_numbers.append(frame_num)
                timestamps.append(frame_num / fps)
            frame_num += 1

        cap.release()

        if frames_to_embed:
            # Batch embed all frames
            clip_embeddings = self._clip().embed_images_batch(frames_to_embed)

            col_img = self._col("sumo_image")
            # Delete old frames
            try:
                col_img.delete(where={"source": str(path)})
            except Exception:
                pass

            ids, docs, metas, embeds = [], [], [], []
            for i, (emb, frame_n, ts) in enumerate(zip(clip_embeddings, frame_numbers, timestamps)):
                chunk_id = f"videoframe_{path.stem}_{self._file_hash(path)}_{i}"
                ids.append(chunk_id)
                docs.append(f"Video frame from {path.name} at {ts:.1f}s")
                embeds.append(emb)
                metas.append({
                    "source":       str(path),
                    "modality":     "video",
                    "type":         "frame",
                    "frame_number": frame_n,
                    "timestamp_s":  round(ts, 2),
                    "fps":          fps,
                    "filename":     path.name,
                })

            col_img.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
            total_chunks += len(frames_to_embed)

        # ── Extract audio track and transcribe ────────────────────────────────
        audio_tmp = None
        try:
            import shutil
            if shutil.which("ffmpeg") is None:
                from rich.console import Console
                console = Console()
                console.print(
                    "[yellow]ffmpeg not found — video audio track will not be transcribed. "
                    "Install: brew install ffmpeg / apt install ffmpeg[/yellow]"
                )
            else:
                audio_tmp = tf.NamedTemporaryFile(suffix=".wav", delete=False)
                audio_tmp.close()
                # Extract audio using cv2 + soundfile
                # Note: requires ffmpeg for full audio extraction
                os.system(f'ffmpeg -i "{path}" -vn -ar 16000 -ac 1 -f wav "{audio_tmp.name}" -y -loglevel quiet')
                if Path(audio_tmp.name).stat().st_size > 1000:
                    audio_chunks = self._whisper().transcribe_chunks(
                        audio_tmp.name,
                        chunk_seconds=self._settings.audio_chunk_seconds,
                    )
                    if audio_chunks:
                        texts = [t for _, _, t in audio_chunks]
                        embeddings = self._text().embed(texts)

                        col_aud = self._col("sumo_audio")
                        try:
                            col_aud.delete(where={"source": str(path)})
                        except Exception:
                            pass

                        ids, docs, metas, embeds = [], [], [], []
                        for i, ((start_s, end_s, text), emb) in enumerate(zip(audio_chunks, embeddings)):
                            chunk_id = f"videoaudio_{path.stem}_{self._file_hash(path)}_{i}"
                            ids.append(chunk_id)
                            docs.append(text)
                            embeds.append(emb)
                            metas.append({
                                "source":    str(path),
                                "modality":  "video",
                                "type":      "audio_track",
                                "start_s":   start_s,
                                "end_s":     end_s,
                                "filename":  path.name,
                            })
                        col_aud.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
                        total_chunks += len(audio_chunks)
        except Exception:
            pass
        finally:
            if audio_tmp and Path(audio_tmp.name).exists():
                os.unlink(audio_tmp.name)

        # ── Store video metadata ──────────────────────────────────────────────
        meta_text = f"Video file: {path.name}. Duration: {frame_num/fps:.1f}s. Frames indexed: {len(frames_to_embed)}."
        meta_emb = self._text().embed_one(meta_text)
        col_vid = self._col("sumo_video")
        try:
            col_vid.delete(where={"source": str(path)})
        except Exception:
            pass
        col_vid.add(
            ids=[f"videometa_{path.stem}_{self._file_hash(path)}"],
            documents=[meta_text],
            embeddings=[meta_emb],
            metadatas=[{
                "source":         str(path),
                "modality":       "video",
                "type":           "metadata",
                "total_frames":   len(frames_to_embed),
                "duration_s":     round(frame_num / fps, 2),
                "fps":            fps,
                "filename":       path.name,
            }]
        )

        return total_chunks

    # ── PDF extraction ────────────────────────────────────────────────────────
    def _extract_pdf(self, path: Path) -> str:
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            raise IngestError("Install pymupdf for PDF support: pip install pymupdf")

    # ── Text chunking ─────────────────────────────────────────────────────────
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
            if start >= len(text):
                break
        return [c for c in chunks if c.strip()]

    # ── Cleanup deleted files ─────────────────────────────────────────────────
    def _cleanup_deleted(self):
        """Remove chunks from ChromaDB for files that no longer exist on disk."""
        to_remove = [p for p in self._index if not Path(p).exists()]
        for p in to_remove:
            modality = self._index[p].get("modality", "text")
            for col_name in ["sumo_text", "sumo_image", "sumo_audio", "sumo_video"]:
                try:
                    self._col(col_name).delete(where={"source": p})
                except Exception:
                    pass
            del self._index[p]
        if to_remove:
            self._save_index()

    # ── Stats ─────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "indexed_files": len(self._index),
            "collections": {
                name: self._col(name).count()
                for name in ["sumo_text", "sumo_image", "sumo_audio", "sumo_video"]
            },
            "by_modality": {
                m: sum(1 for v in self._index.values() if v.get("modality") == m)
                for m in ["text", "image", "audio", "video"]
            }
        }
