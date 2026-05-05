import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from sumospace.settings import SumoSettings
from sumospace.kernel import SumoKernel


@pytest.fixture
def media_settings(tmp_path):
    return SumoSettings(
        chroma_base=str(tmp_path / "chroma"),
        media_enabled=True,
        embedding_provider="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        clip_model="openai/clip-vit-base-patch32",
        whisper_model="tiny",
    )


@pytest.fixture
def mock_embedders():
    """Mock all heavy deep learning models to avoid 1GB+ downloads during tests."""
    with patch("sumospace.embedders.TextEmbedder.embed") as mock_text_batch, \
         patch("sumospace.embedders.TextEmbedder.embed_one") as mock_text_one, \
         patch("sumospace.embedders.CLIPEmbedder.embed_image") as mock_clip_img, \
         patch("sumospace.embedders.CLIPEmbedder.embed_text_for_image_search") as mock_clip_text, \
         patch("sumospace.embedders.WhisperTranscriber.transcribe") as mock_whisper_full, \
         patch("sumospace.embedders.WhisperTranscriber.transcribe_chunks") as mock_whisper_chunk, \
         patch("sentence_transformers.SentenceTransformer") as mock_st:
         
        mock_text_batch.return_value = [[0.1] * 768]
        mock_text_one.return_value = [0.1] * 768
        mock_clip_img.return_value = [0.5] * 512
        mock_clip_text.return_value = [0.5] * 512
        
        mock_whisper_full.return_value = "This is a mock transcription."
        mock_whisper_chunk.return_value = [(0.0, 5.0, "This is a mock transcription chunk.")]
        
        # Mock the underlying SentenceTransformer model instance
        mock_st_instance = MagicMock()
        mock_st_instance.encode.return_value = [[0.1] * 768]
        mock_st.return_value = mock_st_instance
        
        yield {
            "text_batch": mock_text_batch,
            "text_one": mock_text_one,
            "clip_img": mock_clip_img,
            "clip_text": mock_clip_text,
            "whisper_full": mock_whisper_full,
            "whisper_chunk": mock_whisper_chunk,
            "st": mock_st,
        }


@pytest.mark.asyncio
async def test_media_disabled_raises_error(mock_embedders, tmp_path):
    settings = SumoSettings(chroma_base=str(tmp_path), media_enabled=False)
    async with SumoKernel(settings=settings) as kernel:
        with pytest.raises(ValueError, match="Media features are disabled"):
            await kernel.ingest_media("test.jpg")
            
        with pytest.raises(ValueError, match="Media features are disabled"):
            await kernel.search_media("query")


@pytest.mark.asyncio
async def test_search_empty_db_returns_empty_list(mock_embedders, tmp_path):
    settings = SumoSettings(
        media_enabled=True,
        chroma_base=str(tmp_path / ".sumo_db"),
    )
    from sumospace.media_search import MediaSearchEngine
    engine = MediaSearchEngine(settings)
    results = engine.search("anything", top_k=3)
    assert results == []


@pytest.mark.asyncio
async def test_ingest_and_search_text(media_settings, mock_embedders, tmp_path):
    test_file = tmp_path / "doc.txt"
    test_file.write_text("Hello media world")
    
    async with SumoKernel(settings=media_settings) as kernel:
        results = await kernel.ingest_media(str(test_file))
        assert len(results) == 1
        assert results[0].chunks_added > 0
        assert results[0].modality == "text"
        
        # Search it
        search_results = await kernel.search_media("world")
        assert len(search_results) > 0
        assert search_results[0].modality == "text"


@pytest.mark.asyncio
@patch("PIL.Image.open")
async def test_ingest_and_search_image(mock_img_open, media_settings, mock_embedders, tmp_path):
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_img.size = (100, 100)
    mock_img.format = "JPEG"
    mock_img_open.return_value = mock_img
    
    test_file = tmp_path / "photo.jpg"
    test_file.write_bytes(b"fake image bytes")
    
    async with SumoKernel(settings=media_settings) as kernel:
        results = await kernel.ingest_media(str(test_file))
        assert len(results) == 1
        assert results[0].chunks_added == 1
        assert results[0].modality == "image"
        
        # Test cross modal search (text -> image)
        search_results = await kernel.search_media("find a photo")
        assert len(search_results) > 0
        assert search_results[0].modality == "image"
        
        # Test visual search (image -> image)
        search_results_img = await kernel.search_media(str(test_file))
        assert len(search_results_img) > 0
        assert search_results_img[0].modality == "image"


@pytest.mark.asyncio
async def test_ingest_and_search_audio(media_settings, mock_embedders, tmp_path):
    test_file = tmp_path / "audio.mp3"
    test_file.write_bytes(b"fake audio")
    
    async with SumoKernel(settings=media_settings) as kernel:
        results = await kernel.ingest_media(str(test_file))
        assert len(results) == 1
        assert results[0].chunks_added == 1
        assert results[0].modality == "audio"
        
        search_results = await kernel.search_media("transcription")
        # Could be 0 if text isn't embedded similarly to the mock transcription, 
        # but our mock_text_one returns [0.1]*768 for everything, so it should match.
        assert len(search_results) > 0
        assert search_results[0].modality == "audio"


@pytest.mark.asyncio
@patch("cv2.VideoCapture")
async def test_ingest_video(mock_videocapture, media_settings, mock_embedders, tmp_path):
    # Mock video with 1 frame
    mock_cap = MagicMock()
    mock_cap.get.return_value = 24.0 # fps
    mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
    mock_videocapture.return_value = mock_cap
    
    # Mock cvtColor for the frame
    with patch("cv2.cvtColor") as mock_cvt:
        mock_cvt.return_value = MagicMock()
        
        test_file = tmp_path / "video.mp4"
        test_file.write_bytes(b"fake video")
        
        async with SumoKernel(settings=media_settings) as kernel:
            # We must mock os.system to prevent ffmpeg from trying to run
            with patch("os.system") as mock_os:
                results = await kernel.ingest_media(str(test_file))
                assert len(results) == 1
                assert results[0].modality == "video"
                # chunks = 1 frame + 0 audio (ffmpeg mocked out)
                assert results[0].chunks_added > 0
