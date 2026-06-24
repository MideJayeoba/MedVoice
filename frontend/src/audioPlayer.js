/** Play WAV responses — unlock output during mic tap so long API calls can still speak. */

let sharedContext = null
let _currentAudioElement = null
let _currentBufferSource = null
// Set to true by stopPlayback() so promise callbacks don't trigger fallback strategy
let _stopped = false

export function stopPlayback() {
  _stopped = true
  if (_currentAudioElement) {
    // Nullify handlers BEFORE pause/src-clear so onerror doesn't fire into our promise
    _currentAudioElement.onended = null
    _currentAudioElement.onerror = null
    _currentAudioElement.pause()
    _currentAudioElement.src = ''
    _currentAudioElement = null
  }
  if (_currentBufferSource) {
    try { _currentBufferSource.stop() } catch {}
    _currentBufferSource = null
  }
}

export function unlockAudioOutput() {
  if (!sharedContext) {
    sharedContext = new AudioContext()
  }
  const resume =
    sharedContext.state === 'suspended' ? sharedContext.resume() : Promise.resolve()
  return resume.then(() => {
    const buffer = sharedContext.createBuffer(1, 1, 22050)
    const source = sharedContext.createBufferSource()
    source.buffer = buffer
    source.connect(sharedContext.destination)
    source.start(0)
  })
}

export async function playWavBlob(blob) {
  if (!blob || blob.size < 500) throw new Error('empty_audio')

  _stopped = false  // reset for this new playback

  const typed = blob.type?.includes('wav')
    ? blob
    : new Blob([blob], { type: 'audio/wav' })

  // Strategy 1: HTML Audio element
  let needsFallback = false
  try {
    const url = URL.createObjectURL(typed)
    const element = new Audio(url)
    element.volume = 1
    _currentAudioElement = element
    await element.play()
    await new Promise((resolve, reject) => {
      element.onended = () => resolve()
      element.onerror = () => reject(new Error('html_audio_failed'))
    })
    URL.revokeObjectURL(url)
    _currentAudioElement = null
    return
  } catch (err) {
    _currentAudioElement = null
    // User tapped stop — handlers were already nullified, but resolve/reject
    // may have been called by onerror before we nullified. Either way, if
    // _stopped is true we must not fall through to strategy 2.
    if (_stopped) return
    if (err.message === 'html_audio_failed') {
      needsFallback = true
      console.warn('[AudioPlayer] HTML Audio failed, trying Web Audio API...')
    }
    // Any other error (AbortError from play()) — exit silently
  }

  if (!needsFallback) return

  // Strategy 2: Web Audio API fallback
  try {
    const arrayBuffer = await typed.arrayBuffer()
    if (_stopped) return
    const ctx = sharedContext || new AudioContext()
    if (ctx.state === 'suspended') await ctx.resume()
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0))
    if (_stopped) return
    const source = ctx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(ctx.destination)
    _currentBufferSource = source
    await new Promise((resolve, reject) => {
      source.onended = () => resolve()
      source.addEventListener('error', () => reject(new Error('web_audio_failed')), { once: true })
      source.start(0)
    })
    _currentBufferSource = null
  } catch (err) {
    _currentBufferSource = null
    if (_stopped || err.message !== 'web_audio_failed') return
    throw err
  }
}
