"""Text-to-speech — Microsoft Edge TTS (en-NG) only (FR-05)."""

import io
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


def _synthesize_edge_tts(text: str, voice_pref: str) -> bytes:
    """Synthesize speech using Microsoft Edge TTS (en-NG) and convert to WAV in-memory."""
    import av

    # Map selection to Edge TTS neural voice name
    voice_name = "en-NG-EzinneNeural"
    if voice_pref == "Abeo":
        voice_name = "en-NG-AbeoNeural"

    mp3_bytes = run_async(_async_synth_edge_tts(text, voice_name))
    if not mp3_bytes:
        raise RuntimeError("Edge TTS produced empty MP3 output")

    # Convert MP3 to WAV using PyAV
    input_file = io.BytesIO(mp3_bytes)
    output_file = io.BytesIO()

    with av.open(input_file, mode='r') as in_container:
        in_stream = in_container.streams.audio[0]
        with av.open(output_file, mode='w', format='wav') as out_container:
            # PCM 16-bit Mono WAV (standard for VoiceMedAI backend)
            out_stream = out_container.add_stream(
                'pcm_s16le',
                rate=24000,
                layout='mono'
            )
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=24000
            )
            for frame in in_container.decode(in_stream):
                resampled_frames = resampler.resample(frame)
                for r_frame in resampled_frames:
                    for packet in out_stream.encode(r_frame):
                        out_container.mux(packet)
            
            # Flush encoder
            for packet in out_stream.encode(None):
                out_container.mux(packet)

    return output_file.getvalue()


def get_tts_status() -> dict:
    return {"engine": "edge", "available": ["edge"]}


def synthesize_speech(text: str, voice: str = "Ezinne") -> bytes:
    """Synthesize spoken guidance using Microsoft Edge TTS."""
    text = text.strip()
    if not text:
        text = "Please try again."

    return _synthesize_edge_tts(text, voice)
