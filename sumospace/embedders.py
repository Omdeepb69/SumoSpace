# sumospace/embedders.py
"""
Embedding engines for each modality.
All models are lazy-loaded — imported only when first used.
This means users who only use text RAG pay zero cost for CLIP/Whisper.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from PIL import Image as PILImage


class TextEmbedder:
    """
    Sentence-transformers for text.
    Used for: .txt, .md, .py, transcripts, image captions.
    Dimension: 768 (bge-base-en-v1.5)
    """
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class CLIPEmbedder:
    """
    OpenAI CLIP for images and video frames.
    Used for: .jpg/.png/.gif, video keyframes.
    Supports cross-modal: text query → image results.
    Dimension: 512 (clip-vit-base-patch32)

    Cross-modal works because CLIP maps both text and images
    into the same embedding space — text "a cat" and a photo
    of a cat have similar vectors.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model.eval()
        return self._model, self._processor

    def embed_image(self, image: "PILImage.Image") -> list[float]:
        """Embed a PIL Image into CLIP space."""
        import torch
        model, processor = self._load()
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def embed_image_path(self, path: str) -> list[float]:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.embed_image(img)

    def embed_text_for_image_search(self, text: str) -> list[float]:
        """
        Embed text into CLIP space for cross-modal search.
        Use this to find images using a text query.
        The returned vector lives in the same space as embed_image().
        """
        import torch
        model, processor = self._load()
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def embed_images_batch(self, images: list["PILImage.Image"]) -> list[list[float]]:
        """Batch embed multiple images — much faster than one at a time."""
        import torch
        model, processor = self._load()
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.tolist()


class WhisperTranscriber:
    """
    OpenAI Whisper for audio and video audio tracks.
    Used for: .mp3, .wav, .flac, video audio.
    Output: transcribed text → fed into TextEmbedder for embedding.

    Supports faster-whisper (4x speed, same accuracy) when available.
    """
    def __init__(self, model_size: str = "base", use_faster: bool = False):
        self.model_size = model_size
        self.use_faster = use_faster
        self._model = None

    def _load(self):
        if self._model is None:
            if self.use_faster:
                try:
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(self.model_size, device="auto", compute_type="auto")
                    self._backend = "faster"
                    return self._model
                except ImportError:
                    pass
            import whisper
            self._model = whisper.load_model(self.model_size)
            self._backend = "openai"
        return self._model

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        Returns the full transcript as a single string.
        """
        model = self._load()
        if self._backend == "faster":
            segments, _ = model.transcribe(audio_path, beam_size=5)
            return " ".join(seg.text.strip() for seg in segments)
        else:
            result = model.transcribe(audio_path, fp16=False)
            return result["text"].strip()

    def transcribe_chunks(
        self, audio_path: str, chunk_seconds: int = 30
    ) -> list[tuple[float, float, str]]:
        """
        Transcribe with timestamps, returning (start_s, end_s, text) tuples.
        Used for long audio files — allows chunk-level retrieval.
        """
        model = self._load()
        if self._backend == "faster":
            segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=False)
            chunks = []
            current_start = 0.0
            current_text = []
            current_end = 0.0
            for seg in segments:
                current_text.append(seg.text.strip())
                current_end = seg.end
                if seg.end - current_start >= chunk_seconds:
                    chunks.append((current_start, current_end, " ".join(current_text)))
                    current_start = seg.end
                    current_text = []
            if current_text:
                chunks.append((current_start, current_end, " ".join(current_text)))
            return chunks
        else:
            import whisper
            result = model.transcribe(audio_path, fp16=False)
            # Group segments into chunks
            chunks = []
            current_start = 0.0
            current_text = []
            current_end = 0.0
            for seg in result["segments"]:
                current_text.append(seg["text"].strip())
                current_end = seg["end"]
                if seg["end"] - current_start >= chunk_seconds:
                    chunks.append((current_start, current_end, " ".join(current_text)))
                    current_start = seg["end"]
                    current_text = []
            if current_text:
                chunks.append((current_start, current_end, " ".join(current_text)))
            return chunks


class BLIPCaptioner:
    """
    BLIP image captioning for cross-modal text→image search.
    Optional — only loaded when image_generate_caption=True.
    Generates text descriptions of images so text queries can find them.
    """
    MODEL = "Salesforce/blip-image-captioning-base"

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self._processor = BlipProcessor.from_pretrained(self.MODEL)
            self._model = BlipForConditionalGeneration.from_pretrained(self.MODEL)
            self._model.eval()
        return self._model, self._processor

    def caption(self, image_path: str) -> str:
        """Generate a text caption for an image."""
        import torch
        from PIL import Image
        model, processor = self._load()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True)
