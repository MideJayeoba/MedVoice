"""Audio normalization — decode browser blobs to 16 kHz mono WAV for ASR."""

import io
import logging
import struct
import wave

logger = logging.getLogger(__name__)

TARGET_RATE = 16000


def ensure_wav_bytes(audio_bytes: bytes, content_type: str | None = None) -> bytes:
    """
    Return PCM WAV at 16 kHz mono.
    Accepts WAV directly; decodes WebM/OGG via PyAV when available.
    """
    if not audio_bytes:
        raise ValueError("Empty audio payload")

    ctype = (content_type or "").lower()
    if "wav" in ctype or audio_bytes[:4] == b"RIFF":
        try:
            return _normalize_wav(audio_bytes)
        except Exception as exc:
            logger.warning("WAV normalize failed, attempting decode: %s", exc)

    decoded = _decode_with_av(audio_bytes)
    if decoded is not None:
        return decoded

    raise ValueError(
        "Could not decode audio. Send WAV from the browser or install PyAV: pip install av"
    )


def _normalize_wav(audio_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(audio_bytes), "rb") as src:
        channels = src.getnchannels()
        sample_width = src.getsampwidth()
        rate = src.getframerate()
        frames = src.readframes(src.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))
    if channels > 1:
        samples = _mix_to_mono(samples, channels)

    if rate != TARGET_RATE:
        samples = _resample_linear(samples, rate, TARGET_RATE)

    return _pack_wav(samples, TARGET_RATE)


def _mix_to_mono(samples: list[int], channels: int) -> list[int]:
    mono = []
    for i in range(0, len(samples), channels):
        chunk = samples[i : i + channels]
        mono.append(int(sum(chunk) / len(chunk)))
    return mono


def _resample_linear(samples: list[int], src_rate: int, dst_rate: int) -> list[int]:
    if src_rate == dst_rate or not samples:
        return samples
    ratio = dst_rate / src_rate
    out_len = int(len(samples) * ratio)
    if out_len < 1:
        return samples
    output = []
    for i in range(out_len):
        src_index = i / ratio
        left = int(src_index)
        right = min(left + 1, len(samples) - 1)
        frac = src_index - left
        value = samples[left] * (1 - frac) + samples[right] * frac
        output.append(int(max(-32768, min(32767, value))))
    return output


def _pack_wav(samples: list[int], rate: int) -> bytes:
    frames = struct.pack(f"<{len(samples)}h", *samples)
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return out.getvalue()


def _decode_with_av(audio_bytes: bytes) -> bytes | None:
    try:
        import av
        import numpy as np
    except ImportError:
        return None

    try:
        container = av.open(io.BytesIO(audio_bytes))
        stream = container.streams.audio[0]
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            chunks.append(arr.astype(np.float32))

        if not chunks:
            return None

        audio = np.concatenate(chunks)
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = audio / peak
        pcm = (audio * 32767).astype(np.int16).tolist()
        src_rate = int(stream.rate or 48000)
        pcm = _resample_linear(pcm, int(src_rate), TARGET_RATE)
        return _pack_wav(pcm, TARGET_RATE)
    except Exception as exc:
        logger.warning("PyAV decode failed: %s", exc)
        return None
