import { useCallback, useEffect, useRef, useState } from 'react'
import './index.css'
import {
  consultAudio, getConversationTurns, getUserProfile,
  loginUser, logoutUser, registerUser, speakText,
  getStoredToken, updateUserVoice,
} from './api'
import { playWavBlob, stopPlayback, unlockAudioOutput } from './audioPlayer'
import { blobToWav } from './audioUtils'
import { playErrorCue, playListeningCue, playProcessingCue, playReadyCue, playSpeakingCue } from './audioCues'

const S = { IDLE: 'idle', LISTENING: 'listening', PROCESSING: 'processing', SPEAKING: 'speaking' }
const CONV_KEY  = 'voicemed_conv_id'
const THEME_KEY = 'voicemed_theme'
const ERR = 'Sorry, something no work. Please try again, or ask the CHEW for help.'

// ─── Theme hook ───────────────────────────────────────────────
function useTheme() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem(THEME_KEY)
    return saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem(THEME_KEY, dark ? 'dark' : 'light')
  }, [dark])

  return [dark, () => setDark(d => !d)]
}

// ─── Root ────────────────────────────────────────────────────
export default function App() {
  const [user, setUser]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [dark, toggleTheme] = useTheme()

  useEffect(() => {
    if (!getStoredToken()) { setLoading(false); return }
    getUserProfile().then(p => p && setUser(p)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const refresh = () => getUserProfile().then(p => p && setUser(p)).catch(() => {})
  const logout  = async () => { await logoutUser(); setUser(null) }

  if (loading) return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex items-center justify-center">
      <div className="w-10 h-10 rounded-full border-3 border-emerald-500 border-t-transparent spinner" />
    </div>
  )

  if (!user) return <AuthScreen onLogin={setUser} dark={dark} onToggleTheme={toggleTheme} />
  return <VoiceApp user={user} onLogout={logout} onRefresh={refresh} dark={dark} onToggleTheme={toggleTheme} />
}

// ─── Auth Screen ─────────────────────────────────────────────
function AuthScreen({ onLogin, dark, onToggleTheme }) {
  const [mode, setMode]         = useState('login')
  const [username, setUsername] = useState('')
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [busy, setBusy]         = useState(false)

  async function submit(e) {
    e.preventDefault(); setError(''); setBusy(true)
    try {
      if (mode === 'register') await registerUser(username, email, password)
      await loginUser(mode === 'register' ? email : username, password)
      onLogin(await getUserProfile())
    } catch (err) { setError(err.message || 'Something went wrong') }
    finally { setBusy(false) }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 flex flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-emerald-600 flex items-center justify-center shadow-lg">
            <span className="text-lg">🏥</span>
          </div>
          <div>
            <span className="text-slate-900 dark:text-white font-bold text-base leading-none">VoiceMed</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-bold text-base leading-none">AI</span>
          </div>
        </div>
        <ThemeToggle dark={dark} onToggle={onToggleTheme} />
      </div>

      {/* Centre card */}
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md">

          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-extrabold text-slate-900 dark:text-white">Welcome back</h1>
            <p className="text-slate-500 dark:text-slate-400 mt-2">PHC Voice Assistant · Ondo State</p>
          </div>

          {/* Card */}
          <div className="bg-white dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 rounded-3xl shadow-xl shadow-slate-200/60 dark:shadow-none p-8">

            {/* Tabs */}
            <div className="flex bg-slate-100 dark:bg-slate-900/60 rounded-2xl p-1 mb-7 gap-1">
              {['login', 'register'].map(m => (
                <button key={m} type="button" onClick={() => { setMode(m); setError('') }}
                  className={`flex-1 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 ${
                    mode === m
                      ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/25'
                      : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                  }`}>
                  {m === 'login' ? 'Sign In' : 'Register'}
                </button>
              ))}
            </div>

            <form onSubmit={submit} className="space-y-5">
              <AuthInput
                label={mode === 'login' ? 'Username or Email' : 'Username'}
                type="text" autoComplete="username" required
                minLength={mode === 'register' ? 3 : 1}
                placeholder={mode === 'login' ? 'Enter username or email' : 'Choose a username'}
                value={username} onChange={e => setUsername(e.target.value)}
              />

              {mode === 'register' && (
                <AuthInput label="Email Address" type="email" autoComplete="email" required
                  placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)} />
              )}

              <AuthInput label="Password" type="password" required minLength={6}
                autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
                placeholder="Minimum 6 characters"
                value={password} onChange={e => setPassword(e.target.value)} />

              {error && (
                <div className="flex items-center gap-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 rounded-xl px-4 py-3">
                  <span className="text-red-500 shrink-0">⚠</span>
                  <p className="text-red-600 dark:text-red-400 text-sm">{error}</p>
                </div>
              )}

              <button type="submit" disabled={busy}
                className="w-full py-3.5 rounded-2xl font-bold text-white bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 disabled:opacity-60 transition-colors shadow-lg shadow-emerald-600/25 flex items-center justify-center gap-2 text-base">
                {busy
                  ? <span className="w-5 h-5 rounded-full border-2 border-white/40 border-t-white spinner" />
                  : mode === 'login' ? 'Sign In' : 'Create Account'}
              </button>
            </form>
          </div>

          <p className="text-center text-slate-400 dark:text-slate-600 text-xs mt-6">
            Secure · Private · Voice-first healthcare
          </p>
        </div>
      </div>
    </div>
  )
}

function AuthInput({ label, ...props }) {
  return (
    <div>
      <label className="block text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">{label}</label>
      <input {...props}
        className="w-full bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-3.5 text-slate-900 dark:text-white text-sm placeholder-slate-400 dark:placeholder-slate-600 outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/15 transition-all" />
    </div>
  )
}

// ─── Voice App ────────────────────────────────────────────────
function VoiceApp({ user, onLogout, onRefresh, dark, onToggleTheme }) {
  const [state, setState]       = useState(S.IDLE)
  const [panelOpen, setPanelOpen] = useState(false)
  const [voice, setVoice]       = useState(user.tts_voice || 'Ezinne')
  const [convId, setConvId]     = useState(() => sessionStorage.getItem(CONV_KEY) || null)
  const [turns, setTurns]       = useState([])
  const [convos, setConvos]     = useState(user.conversations || [])
  const [resuming, setResuming] = useState(false)
  const bottomRef   = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef   = useRef([])
  const streamRef   = useRef(null)

  useEffect(() => { setConvos(user.conversations || []) }, [user.conversations])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [turns])
  useEffect(() => {
    playReadyCue()
    return () => streamRef.current?.getTracks().forEach(t => t.stop())
  }, [])

  async function changeVoice(v) {
    if (v === voice) return
    try { await updateUserVoice(v); setVoice(v); user.tts_voice = v } catch {}
  }

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null
  }, [])

  const playResp = useCallback(async (blob) => {
    setState(S.SPEAKING); playSpeakingCue()
    try { await playWavBlob(blob) } catch {}
    setState(S.IDLE); playReadyCue()
  }, [])

  const stopSpeaking = useCallback(() => {
    stopPlayback()
    setState(S.IDLE)
    playReadyCue()
  }, [])

  const processBlob = useCallback(async (blob) => {
    setState(S.PROCESSING); playProcessingCue()
    const cid = convId || crypto.randomUUID()
    if (!convId) { setConvId(cid); sessionStorage.setItem(CONV_KEY, cid) }
    try {
      const { blob: audio, transcript: t, guidance: g, escalate: e } =
        await consultAudio(await blobToWav(blob), cid)
      setTurns(prev => [...prev, { id: Date.now(), transcript: t, guidance: g, escalate: e, created_at: new Date().toISOString() }])
      // Only auto-open panel on desktop; on mobile the preview card is shown instead
      if (window.innerWidth >= 640) setPanelOpen(true)
      onRefresh()
      await playResp(audio)
    } catch {
      playErrorCue()
      try { await playResp(await speakText(ERR)) } catch { setState(S.IDLE); playReadyCue() }
    }
  }, [playResp, onRefresh, convId])

  const startRec = useCallback(async () => {
    if (state !== S.IDLE) return
    try {
      await unlockAudioOutput()
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream; chunksRef.current = []
      const rec = new MediaRecorder(stream, { mimeType: getMimeType() })
      recorderRef.current = rec
      rec.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      rec.onstop = async () => {
        stopStream()
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || 'audio/webm' })
        blob.size > 0 ? await processBlob(blob) : (setState(S.IDLE), playReadyCue())
      }
      rec.start(); setState(S.LISTENING); playListeningCue()
    } catch { playErrorCue(); setState(S.IDLE) }
  }, [state, processBlob, stopStream])

  const stopRec = useCallback(() => {
    if (state === S.LISTENING && recorderRef.current?.state === 'recording')
      recorderRef.current.stop()
  }, [state])

  function newConvo() {
    setConvId(null); setTurns([]); setPanelOpen(false); sessionStorage.removeItem(CONV_KEY)
  }

  async function resume(id) {
    setResuming(true)
    try {
      const t = await getConversationTurns(id)
      setConvId(id); setTurns(t); sessionStorage.setItem(CONV_KEY, id); setPanelOpen(true)
    } catch {} finally { setResuming(false) }
  }

  // Per-state config
  const stateConfig = {
    [S.IDLE]:       { label: 'Tap to speak',     sub: 'Priscilla is ready', color: 'text-emerald-600 dark:text-emerald-400', micCls: 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-400 hover:bg-emerald-100 dark:hover:bg-emerald-900/40 hover:scale-105 active:scale-95 mic-idle',    icon: <MicIcon cls="w-16 h-16 text-emerald-600 dark:text-emerald-400" /> },
    [S.LISTENING]:  { label: 'Listening…',        sub: 'Tap to stop',        color: 'text-red-500 dark:text-red-400',         micCls: 'bg-red-50 dark:bg-red-900/20 border-red-400 scale-105 mic-listening',                                                                                     icon: <WaveIcon cls="w-16 h-8 text-red-500 dark:text-red-400" /> },
    [S.PROCESSING]: { label: 'Thinking…',         sub: 'Please wait',        color: 'text-sky-600 dark:text-sky-400',         micCls: 'bg-sky-50 dark:bg-sky-900/20 border-sky-300 dark:border-sky-700 cursor-default',                                                                         icon: <div className="w-10 h-10 rounded-full border-3 border-sky-200 dark:border-sky-800 border-t-sky-600 dark:border-t-sky-400 spinner" /> },
    [S.SPEAKING]:   { label: 'Speaking…',         sub: 'Tap to stop',          color: 'text-sky-600 dark:text-sky-400',       micCls: 'bg-sky-50 dark:bg-sky-900/20 border-sky-400 hover:bg-sky-100 dark:hover:bg-sky-900/40 hover:scale-105 active:scale-95 mic-speaking',                icon: <SpeakerIcon cls="w-16 h-16 text-sky-600 dark:text-sky-400" /> },
  }
  const cfg = stateConfig[state]

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex flex-col transition-colors duration-300">

      {/* Header */}
      <header className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 px-6 py-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-emerald-600 flex items-center justify-center shadow-md">
            <span className="text-lg">🏥</span>
          </div>
          <div className="leading-none">
            <span className="text-slate-900 dark:text-white font-bold text-lg">VoiceMed</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-bold text-lg">AI</span>
            <p className="text-slate-400 dark:text-slate-500 text-xs mt-0.5">PHC Assistant · Ondo State</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <ThemeToggle dark={dark} onToggle={onToggleTheme} />
          <button onClick={() => setPanelOpen(p => !p)}
            className="relative flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm font-semibold transition-colors">
            💬 <span className="hidden sm:inline">Consults</span>
            {turns.length > 0 && (
              <span className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-emerald-600 text-white text-[10px] font-bold flex items-center justify-center">
                {turns.length}
              </span>
            )}
          </button>
          <button onClick={onLogout}
            className="px-4 py-2 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-red-50 dark:hover:bg-red-900/20 text-slate-500 dark:text-slate-400 hover:text-red-600 dark:hover:text-red-400 text-sm font-semibold transition-colors">
            Sign out
          </button>
        </div>
      </header>

      {/* Main stage */}
      <main className="flex-1 flex flex-col items-center justify-center gap-10 p-6 lg:p-12">

        {/* Greeting */}
        <div className="text-center">
          <p className="text-slate-500 dark:text-slate-400 text-base mb-1">Good day, <span className="text-slate-800 dark:text-slate-200 font-semibold">{user.username}</span></p>
          <h2 className="text-3xl lg:text-4xl font-extrabold text-slate-900 dark:text-white">How can I help you today?</h2>
          <p className="text-slate-400 dark:text-slate-500 text-sm mt-2">Speak in English, Yoruba, or Pidgin</p>
        </div>

        {/* Mic button */}
        <div className="flex flex-col items-center gap-6">
          <button
            type="button"
            onClick={state === S.IDLE ? startRec : state === S.LISTENING ? stopRec : state === S.SPEAKING ? stopSpeaking : undefined}
            disabled={state === S.PROCESSING}
            aria-label={cfg.label}
            className={`w-48 h-48 lg:w-56 lg:h-56 rounded-full border-4 flex items-center justify-center transition-all duration-300 ${cfg.micCls}`}
          >
            {cfg.icon}
          </button>

          <div className="text-center">
            <p className={`text-xl font-bold ${cfg.color}`}>{cfg.label}</p>
            <p className="text-slate-400 dark:text-slate-500 text-sm mt-0.5">{cfg.sub}</p>
          </div>
        </div>

        {/* Voice toggle */}
        <div className="flex flex-col items-center gap-3">
          <p className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">Voice Assistant</p>
          <div className="flex bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl p-1.5 gap-1.5 shadow-sm">
            {[
              { id: 'Ezinne', label: 'Ezinne', emoji: '👩‍⚕️', desc: 'Female' },
              { id: 'Abeo',   label: 'Abeo',   emoji: '👨‍⚕️', desc: 'Male'   },
            ].map(({ id, label, emoji, desc }) => (
              <button key={id} type="button" onClick={() => changeVoice(id)}
                className={`flex items-center gap-2.5 px-5 py-3 rounded-xl text-sm font-bold transition-all duration-200 ${
                  voice === id
                    ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/30 scale-105'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700/50'
                }`}>
                <span className="text-xl">{emoji}</span>
                <div className="text-left">
                  <div>{label}</div>
                  <div className={`text-xs font-normal ${voice === id ? 'text-emerald-100' : 'text-slate-400 dark:text-slate-500'}`}>{desc}</div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Latest response preview (if turns exist and panel closed) */}
        {turns.length > 0 && !panelOpen && (
          <div className="w-full max-w-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl p-5 shadow-sm">
            <p className="text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mb-2">Last Response · Priscilla</p>
            <p className="text-slate-700 dark:text-slate-200 text-sm leading-relaxed line-clamp-3">{turns[turns.length - 1].guidance}</p>
            <button onClick={() => setPanelOpen(true)} className="mt-3 text-xs font-semibold text-emerald-600 dark:text-emerald-400 hover:underline">
              View full conversation →
            </button>
          </div>
        )}
      </main>

      {/* ── Consult Panel ── */}
      <div className={`fixed inset-0 z-40 transition-opacity duration-300 ${panelOpen ? 'pointer-events-auto' : 'pointer-events-none opacity-0'}`}>
        {/* Backdrop */}
        <div className={`absolute inset-0 bg-black/30 dark:bg-black/50 transition-opacity duration-300 ${panelOpen ? 'opacity-100' : 'opacity-0'}`}
          onClick={() => setPanelOpen(false)} />

        {/* Drawer */}
        <aside className={`absolute top-0 right-0 h-full w-full sm:w-[420px] lg:w-[480px] bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-700/60 flex flex-col shadow-2xl transition-transform duration-300 ${panelOpen ? 'translate-x-0' : 'translate-x-full'}`}>

          {/* Panel header */}
          <div className="flex items-center justify-between px-6 py-5 border-b border-slate-200 dark:border-slate-700/60 shrink-0">
            <div>
              <h2 className="text-slate-900 dark:text-white font-bold text-lg">Consult Panel</h2>
              <p className="text-slate-400 dark:text-slate-500 text-xs mt-0.5">@{user.username}</p>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={newConvo}
                className="px-4 py-2 rounded-xl border border-emerald-200 dark:border-emerald-800/60 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-400 text-xs font-bold hover:bg-emerald-100 dark:hover:bg-emerald-900/40 transition-colors">
                + New
              </button>
              <button onClick={() => setPanelOpen(false)}
                className="w-9 h-9 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 flex items-center justify-center text-lg transition-colors">
                ✕
              </button>
            </div>
          </div>

          {/* Panel body */}
          <div className="flex-1 overflow-y-auto px-6 py-5 space-y-8">

            {/* Chat thread */}
            <div>
              <SectionHeading>
                {turns.length > 0
                  ? `Current Conversation · ${turns.length} exchange${turns.length !== 1 ? 's' : ''}`
                  : 'Current Conversation'}
              </SectionHeading>

              {turns.length === 0 ? (
                <div className="flex flex-col items-center gap-3 py-12 text-center">
                  <div className="w-14 h-14 rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-2xl">🎙️</div>
                  <p className="text-slate-400 dark:text-slate-500 text-sm">No consultation yet.<br />Tap the mic button to start.</p>
                </div>
              ) : (
                <div className="space-y-5">
                  {turns.map(turn => (
                    <div key={turn.id} className="space-y-3">
                      {/* Patient bubble */}
                      <div className="flex justify-end">
                        <div className="max-w-[80%] bg-slate-100 dark:bg-slate-800 rounded-2xl rounded-br-md px-4 py-3">
                          <p className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-1.5">You (Patient)</p>
                          <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed">{turn.transcript || '—'}</p>
                        </div>
                      </div>
                      {/* Priscilla bubble */}
                      <div className="flex justify-start">
                        <div className="max-w-[80%] bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-100 dark:border-emerald-800/40 rounded-2xl rounded-bl-md px-4 py-3">
                          <p className="text-[10px] font-bold text-emerald-600 dark:text-emerald-500 uppercase tracking-wider mb-1.5">👩‍⚕️ Priscilla</p>
                          <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed">{turn.guidance || '…'}</p>
                          {turn.escalate && (
                            <div className="mt-3 flex items-start gap-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 rounded-xl px-3 py-2.5">
                              <span className="shrink-0">⚠️</span>
                              <p className="text-red-700 dark:text-red-400 text-xs font-semibold leading-snug">Refer this patient to a doctor immediately.</p>
                            </div>
                          )}
                          <p className="text-[10px] text-slate-400 dark:text-slate-600 mt-2">{fmtDate(turn.created_at)}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>

            {/* Past conversations */}
            {convos.length > 0 && (
              <div>
                <SectionHeading>
                  Past Conversations <span className="ml-1.5 px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 text-[11px] font-bold">{convos.length}</span>
                </SectionHeading>
                <ul className="space-y-2.5">
                  {convos.map(c => {
                    const active = convId === c.conversation_id
                    return (
                      <li key={c.conversation_id}
                        className={`rounded-2xl border p-4 transition-all ${
                          active
                            ? 'bg-emerald-50 dark:bg-emerald-900/15 border-emerald-200 dark:border-emerald-800/50'
                            : 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700/60 hover:border-slate-300 dark:hover:border-slate-600'
                        }`}>
                        <p className="text-sm text-slate-700 dark:text-slate-300 line-clamp-2 mb-3 leading-relaxed">
                          {c.first_transcript?.slice(0, 85) || 'Conversation'}…
                        </p>
                        <div className="flex items-center justify-between">
                          <div className="text-xs text-slate-400 dark:text-slate-500">
                            {fmtDate(c.last_at)} · <span className="text-emerald-600 dark:text-emerald-500 font-semibold">{c.turn_count} turn{c.turn_count !== 1 ? 's' : ''}</span>
                          </div>
                          {active
                            ? <span className="flex items-center gap-1.5 text-xs font-bold text-emerald-600 dark:text-emerald-400"><span className="w-2 h-2 rounded-full bg-emerald-500 mic-idle" />Active</span>
                            : <button disabled={resuming} onClick={() => resume(c.conversation_id)}
                                className="px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:border-emerald-300 dark:hover:border-emerald-700 hover:text-emerald-700 dark:hover:text-emerald-400 text-xs font-bold disabled:opacity-40 transition-all">
                                {resuming ? '…' : 'Resume'}
                              </button>
                          }
                        </div>
                      </li>
                    )
                  })}
                </ul>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}

// ─── Shared UI ────────────────────────────────────────────────
function ThemeToggle({ dark, onToggle }) {
  return (
    <button onClick={onToggle} aria-label="Toggle theme"
      className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 flex items-center justify-center text-lg transition-colors">
      {dark ? '☀️' : '🌙'}
    </button>
  )
}

function SectionHeading({ children }) {
  return (
    <h3 className="flex items-center gap-2 text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest mb-4">
      {children}
    </h3>
  )
}

// ─── Helpers ──────────────────────────────────────────────────
function getMimeType() {
  if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) return 'audio/webm;codecs=opus'
  if (MediaRecorder.isTypeSupported('audio/webm')) return 'audio/webm'
  return ''
}

function fmtDate(iso) {
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

// ─── Icons ────────────────────────────────────────────────────
function MicIcon({ cls }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={cls}>
      <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm0 13a6 6 0 0 0 6-6H16a4 4 0 0 1-8 0H6a6 6 0 0 0 6 6zm-1 3v3h2v-3h-2z" />
    </svg>
  )
}

function WaveIcon({ cls }) {
  return (
    <svg viewBox="0 0 60 28" fill="none" aria-hidden="true" className={cls}>
      {[6, 14, 20, 14, 6].map((h, i) => (
        <rect key={i} x={6 + i * 12} y={14 - h / 2} width="7" height={h} rx="3.5"
          fill="currentColor" className="wave-bar" />
      ))}
    </svg>
  )
}

function SpeakerIcon({ cls }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={cls}>
      <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.06c1.48-.74 2.5-2.26 2.5-4.03zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
    </svg>
  )
}
