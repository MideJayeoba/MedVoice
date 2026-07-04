"""Text-to-speech — Microsoft Edge TTS (en-NG) — returns MP3 directly (FR-05)."""

import logging
from backend.config import AUDIO_CACHE_DIR

logger = logging.getLogger(__name__)

AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def run_async(coro):
    import threading
    import asyncio

    result = []
    error = []

    def target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result.append(loop.run_until_complete(coro))
        except Exception as e:
            error.append(e)
        finally:
            loop.close()

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    return result[0]


async def _async_synth_edge_tts(text: str, voice_name: str) -> bytes:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice_name)
    mp3_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_chunks.append(chunk["data"])
    return b"".join(mp3_chunks)


def synthesize_speech(text: str, voice: str = "Ezinne") -> bytes:
    """Synthesize spoken guidance using Edge TTS. Returns raw MP3 bytes."""
    text = text.strip()
    if not text:
        text = "Please try again."

    # Map selection to Edge TTS neural voice name
    voice_name = "en-NG-EzinneNeural"
    if voice == "Abeo":
        voice_name = "en-NG-AbeoNeural"

    mp3_bytes = run_async(_async_synth_edge_tts(text, voice_name))
    if not mp3_bytes:
        raise RuntimeError("Edge TTS produced empty MP3 output")

    logger.info("TTS synthesized %d bytes MP3 for voice %s", len(mp3_bytes), voice)
    return mp3_bytes


def get_tts_status() -> dict:
    return {"engine": "edge", "format": "mp3", "available": ["edge"]}

