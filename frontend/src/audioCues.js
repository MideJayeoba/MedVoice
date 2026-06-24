let audioContext

function getContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)()
  }
  return audioContext
}

function playTone(frequency, duration, volume = 0.08) {
  const ctx = getContext()
  const oscillator = ctx.createOscillator()
  const gain = ctx.createGain()

  oscillator.type = 'sine'
  oscillator.frequency.value = frequency
  gain.gain.value = volume

  oscillator.connect(gain)
  gain.connect(ctx.destination)

  const now = ctx.currentTime
  oscillator.start(now)
  oscillator.stop(now + duration)
}

export function playReadyCue() {
  playTone(440, 0.12)
}

export function playListeningCue() {
  playTone(523, 0.1)
}

export function playProcessingCue() {
  playTone(330, 0.15)
}

export function playSpeakingCue() {
  playTone(392, 0.12)
}

export function playErrorCue() {
  playTone(220, 0.25, 0.12)
}
