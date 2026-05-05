# sumospace/multimodal_search.py
"""
Multimodal search engine.
Detects query type, routes to correct collection and embedding space,
returns top-K results with modality-aware context.

Query routing:
  text string  → text embedding → search sumo_text
                               → also search sumo_image (cross-modal via CLIP)
                               → also search sumo_audio
  image file   → CLIP embedding → search sumo_image
  audio file   → transcribe    → text embedding → search sumo_audio
  video file   → extract frame → CLIP embedding → search sumo_image

Cross-modal examples that work:
  "authentication flow" → finds text docs + images of auth diagrams
  query_image.jpg       → finds similar images + video frames
  voice_note.mp3        → finds similar audio + text with same topic
"""
from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass

import chromadb

from sumospace.embedders import TextEmbedder, CLIPEmbedder, WhisperTranscriber
from sumospace.settings import SumoSettings

IMAGE_EXT = {".jpg",".jpeg",".png",".gif",".bmp",".webp",".tiff",".tif",".heic"}
AUDIO_EXT = {".mp3",".wav",".flac",".aac",".ogg",".m4a",".wma",".opus",".aiff"}
VIDEO_EXT = {".mp4",".avi",".mov",".mkv",".webm",".flv",".wmv",".m4v",".mpeg"}


@dataclass
class SearchResult:
    rank:        int
    source_path: str
    modality:    str
    content:     str
    score:       float
    metadata:    dict

    def preview(self, max_chars: int = 120) -> str:
        mod = self.modality.upper()
        src = Path(self.source_path).name
        content_preview = self.content[:max_chars].replace("\n", " ")
        score_pct = self.score * 100

        extras = ""
        if self.metadata.get("timestamp_s"):
            extras = f" @ {self.metadata['timestamp_s']:.1f}s"
        if self.metadata.get("start_s") is not None:
            extras = f" [{self.metadata['start_s']:.0f}s–{self.metadata['end_s']:.0f}s]"
        if self.metadata.get("caption"):
            extras = f" [caption: {self.metadata['caption'][:40]}]"

        return f"[{mod}] {src}{extras} ({score_pct:.1f}% match)\n  {content_preview}"


class MultimodalSearchEngine:
    """
    Route queries to correct embedding space and collection.
    Returns top-K results, deduped by source file.
    """

    def __init__(self, settings: SumoSettings):
        self._settings = settings
        self._db_path  = Path(settings.chroma_base) / "multimodal"
        self._client   = chromadb.PersistentClient(path=str(self._db_path))
        self._cols: dict[str, chromadb.Collection] = {}

        # Lazy embedders
        self._text_embedder: TextEmbedder      | None = None
        self._clip_embedder: CLIPEmbedder      | None = None
        self._transcriber:   WhisperTranscriber | None = None

    def _col(self, name: str):
        if name not in self._cols:
            try:
                self._cols[name] = self._client.get_collection(name)
            except Exception:
                return None
        return self._cols[name]

    def _text(self) -> TextEmbedder:
        if not self._text_embedder:
            self._text_embedder = TextEmbedder(self._settings.embedding_model)
        return self._text_embedder

    def _clip(self) -> CLIPEmbedder:
        if not self._clip_embedder:
            self._clip_embedder = CLIPEmbedder(self._settings.clip_model)
        return self._clip_embedder

    def _whisper(self) -> WhisperTranscriber:
        if not self._transcriber:
            self._transcriber = WhisperTranscriber(
                model_size=self._settings.whisper_model,
                use_faster=self._settings.whisper_use_faster,
            )
        return self._transcriber

    def _detect_query_type(self, query: str) -> str:
        """Detect if query is text, image path, audio path, or video path."""
        p = Path(query)
        if p.exists():
            ext = p.suffix.lower()
            if ext in IMAGE_EXT: return "image"
            if ext in AUDIO_EXT: return "audio"
            if ext in VIDEO_EXT: return "video"
        return "text"

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Search across all modalities.
        Query type is auto-detected:
          - text string  → semantic text + cross-modal image search
          - image path   → visual similarity search
          - audio path   → transcript similarity search
          - video path   → frame-based visual search

        Returns top_k results sorted by similarity score.
        """
        query_type = self._detect_query_type(query)

        if query_type == "text":
            return self._search_text_query(query, top_k)
        elif query_type == "image":
            return self._search_image_query(query, top_k)
        elif query_type == "audio":
            return self._search_audio_query(query, top_k)
        elif query_type == "video":
            return self._search_video_query(query, top_k)
        return []

    def search_modality(
        self, query: str, modality: str, top_k: int = 3
    ) -> list[SearchResult]:
        """
        Force search within a specific modality.
        modality: "text" | "image" | "audio" | "video"
        Useful for: "find images similar to this image" when you
        want images only, not all modalities.
        """
        query_type = self._detect_query_type(query)
        col_map = {
            "text":  "sumo_text",
            "image": "sumo_image",
            "audio": "sumo_audio",
            "video": "sumo_image",  # video frames live in sumo_image
        }
        col_name = col_map.get(modality, "sumo_text")
        col = self._col(col_name)
        if col is None or col.count() == 0:
            return []

        # Get embedding based on query type
        if query_type == "image" and modality in ("image", "video"):
            from PIL import Image
            img = Image.open(query).convert("RGB")
            emb = self._clip().embed_image(img)
        elif query_type == "text" and modality in ("image", "video"):
            # Cross-modal: text→image via CLIP text encoder
            emb = self._clip().embed_text_for_image_search(query)
        elif query_type in ("audio", "video"):
            transcript = self._whisper().transcribe(query)
            emb = self._text().embed_one(transcript)
        else:
            emb = self._text().embed_one(query)

        results = col.query(query_embeddings=[emb], n_results=min(top_k, col.count()))
        return self._parse_results(results, top_k)

    # ── Query type handlers ───────────────────────────────────────────────────
    def _search_text_query(self, query: str, top_k: int) -> list[SearchResult]:
        """
        Text query → search across all modalities:
        1. Text collection (semantic)
        2. Image collection (CLIP cross-modal text→image)
        3. Audio collection (semantic)
        Merge and deduplicate results.
        """
        all_results = []
        text_emb = None

        # 1. Text search
        col_text = self._col("sumo_text")
        if col_text and col_text.count() > 0:
            text_emb = self._text().embed_one(query)
            r = col_text.query(
                query_embeddings=[text_emb],
                n_results=min(top_k * 2, col_text.count())
            )
            all_results.extend(self._parse_results(r, top_k * 2))

        # 2. Cross-modal image search via CLIP
        col_img = self._col("sumo_image")
        if col_img and col_img.count() > 0:
            clip_emb = self._clip().embed_text_for_image_search(query)
            r = col_img.query(
                query_embeddings=[clip_emb],
                n_results=min(top_k, col_img.count())
            )
            all_results.extend(self._parse_results(r, top_k))

        # 3. Audio search (via transcript embeddings)
        col_aud = self._col("sumo_audio")
        if col_aud and col_aud.count() > 0:
            if text_emb is None:
                text_emb = self._text().embed_one(query)
            r = col_aud.query(
                query_embeddings=[text_emb],
                n_results=min(top_k, col_aud.count())
            )
            all_results.extend(self._parse_results(r, top_k))

        return self._merge_and_rank(all_results, top_k)

    def _search_image_query(self, query_path: str, top_k: int) -> list[SearchResult]:
        """
        Image query → find visually similar images and video frames.
        Uses CLIP visual embedding — no text involved.
        """
        from PIL import Image
        img = Image.open(query_path).convert("RGB")
        clip_emb = self._clip().embed_image(img)

        results = []
        col_img = self._col("sumo_image")
        if col_img and col_img.count() > 0:
            r = col_img.query(
                query_embeddings=[clip_emb],
                n_results=min(top_k, col_img.count())
            )
            results.extend(self._parse_results(r, top_k))

        return self._merge_and_rank(results, top_k)

    def _search_audio_query(self, query_path: str, top_k: int) -> list[SearchResult]:
        """
        Audio query → transcribe → find semantically similar audio chunks.
        Also searches text collection for related documents.
        """
        transcript = self._whisper().transcribe(query_path)
        if not transcript.strip():
            return []

        text_emb = self._text().embed_one(transcript)
        results = []

        col_aud = self._col("sumo_audio")
        if col_aud and col_aud.count() > 0:
            r = col_aud.query(
                query_embeddings=[text_emb],
                n_results=min(top_k, col_aud.count())
            )
            results.extend(self._parse_results(r, top_k))

        # Also find related text documents
        col_txt = self._col("sumo_text")
        if col_txt and col_txt.count() > 0:
            r = col_txt.query(
                query_embeddings=[text_emb],
                n_results=min(top_k, col_txt.count())
            )
            results.extend(self._parse_results(r, top_k))

        return self._merge_and_rank(results, top_k)

    def _search_video_query(self, query_path: str, top_k: int) -> list[SearchResult]:
        """
        Video query → extract middle frame → CLIP → find similar frames/images.
        """
        import cv2
        import numpy as np
        from PIL import Image

        cap = cv2.VideoCapture(query_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        clip_emb  = self._clip().embed_image(pil_frame)

        col_img = self._col("sumo_image")
        if not col_img or col_img.count() == 0:
            return []

        r = col_img.query(
            query_embeddings=[clip_emb],
            n_results=min(top_k, col_img.count())
        )
        return self._merge_and_rank(self._parse_results(r, top_k), top_k)

    # ── Result parsing ────────────────────────────────────────────────────────
    def _parse_results(self, chroma_result: dict, top_k: int) -> list[SearchResult]:
        results = []
        ids       = chroma_result.get("ids",       [[]])[0]
        documents = chroma_result.get("documents", [[]])[0]
        metadatas = chroma_result.get("metadatas", [[]])[0]
        distances = chroma_result.get("distances", [[]])[0]

        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # ChromaDB cosine distance: 0=identical, 2=opposite
            # Convert to similarity score 0-1
            score = max(0.0, 1.0 - (dist / 2.0))
            results.append(SearchResult(
                rank=i + 1,
                source_path=meta.get("source", "unknown"),
                modality=meta.get("modality", "text"),
                content=doc,
                score=score,
                metadata=meta,
            ))
        return results

    def _merge_and_rank(
        self, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """
        Deduplicate by source file (keep highest scoring chunk per file),
        then sort by score descending, return top_k.
        """
        best: dict[str, SearchResult] = {}
        for r in results:
            key = r.source_path
            if key not in best or r.score > best[key].score:
                best[key] = r

        ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)
        for i, r in enumerate(ranked[:top_k]):
            r.rank = i + 1
        return ranked[:top_k]
