/** Convert browser recording (WebM) to 16 kHz mono WAV for backend ASR. */

export async function blobToWav(blob, targetSampleRate = 16000) {
  const arrayBuffer = await blob.arrayBuffer()
  const ctx = new AudioContext()
  let audioBuffer

  try {
    audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0))
  } finally {
    await ctx.close()
  }

  const length = Math.max(1, Math.ceil(audioBuffer.duration * targetSampleRate))
  const offline = new OfflineAudioContext(1, length, targetSampleRate)

  const source = offline.createBufferSource()
  source.buffer = audioBuffer
  source.connect(offline.destination)
  source.start(0)

  const rendered = await offline.startRendering()
  return encodeWav(rendered.getChannelData(0), targetSampleRate)
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, samples.length * 2, true)

  let offset = 44
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const sample = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
  }

  return new Blob([buffer], { type: 'audio/wav' })
}
