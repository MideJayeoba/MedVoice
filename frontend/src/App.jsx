import { useCallback, useEffect, useRef, useState } from 'react'
import './index.css'
import {
  consultAudio, consultText, getConversationTurns, getUserProfile,
  loginUser, loginWithGoogle, logoutUser, registerUser, speakText, predictTriageNow,
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
  const [user, setUser]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [dark, toggleTheme]   = useTheme()

  useEffect(() => {
    if (!getStoredToken()) { setLoading(false); return }
    getUserProfile().then(p => p && setUser(p)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const refresh = () => getUserProfile().then(p => p && setUser(p)).catch(() => {})
  const logout  = async () => { await logoutUser(); setUser(null) }

  if (loading) return (
    <div className="min-h-screen bg-mint dark:bg-slate-950 flex items-center justify-center">
      <div className="w-10 h-10 rounded-full border-3 border-emerald-600 border-t-transparent spinner" />
    </div>
  )

  if (!user) return <AuthScreen onLogin={setUser} dark={dark} onToggleTheme={toggleTheme} />
  return <Shell user={user} onLogout={logout} onRefresh={refresh} dark={dark} onToggleTheme={toggleTheme} />
}

// ─── Auth Screen (Stitch: auth_refined) ──────────────────────
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || ''

function AuthScreen({ onLogin, dark, onToggleTheme }) {
  const [mode, setMode]         = useState('login')
  const [username, setUsername] = useState('')
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [firstName, setFirstName]   = useState('')
  const [middleName, setMiddleName] = useState('')
  const [lastName, setLastName]     = useState('')
  const [error, setError]       = useState('')
  const [busy, setBusy]         = useState(false)
  const googleBtnRef = useRef(null)

  async function submit(e) {
    e.preventDefault(); setError(''); setBusy(true)
    try {
      if (mode === 'register') await registerUser(username, email, password, firstName, middleName, lastName)
      await loginUser(mode === 'register' ? email : username, password)
      onLogin(await getUserProfile())
    } catch (err) { setError(err.message || 'Something went wrong') }
    finally { setBusy(false) }
  }

  // Google Identity Services: load script once, render the official button
  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) return
    const init = () => {
      if (!window.google?.accounts?.id || !googleBtnRef.current) return
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: async ({ credential }) => {
          setError(''); setBusy(true)
          try {
            await loginWithGoogle(credential)
            onLogin(await getUserProfile())
          } catch (err) { setError(err.message || 'Google sign-in failed') }
          finally { setBusy(false) }
        },
      })
      window.google.accounts.id.renderButton(googleBtnRef.current, {
        theme: dark ? 'filled_black' : 'outline',
        size: 'large', shape: 'pill', width: 320, text: 'continue_with',
      })
    }
    if (window.google?.accounts?.id) { init(); return }
    const s = document.createElement('script')
    s.src = 'https://accounts.google.com/gsi/client'
    s.async = true
    s.onload = init
    document.head.appendChild(s)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dark])

  return (
    <div className="min-h-screen bg-mint dark:bg-slate-950 flex flex-col transition-colors">
      <div className="flex justify-end px-5 pt-5">
        <ThemeToggle dark={dark} onToggle={onToggleTheme} />
      </div>

      <div className="flex-1 flex flex-col items-center justify-center px-4 pb-10">
        {/* Logo block */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-20 h-20 rounded-full bg-white dark:bg-slate-800 shadow-lg shadow-emerald-900/10 flex items-center justify-center mb-5 ring-8 ring-emerald-600/5">
            <MicPlusIcon cls="w-9 h-9 text-emerald-700 dark:text-emerald-400" />
          </div>
          <h1 className="text-3xl font-extrabold text-emerald-900 dark:text-white tracking-tight">
            VoiceMed<span className="text-emerald-600">AI</span>
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1.5 text-sm">Nigerian Primary Health Care Portal</p>
        </div>

        {/* Card */}
        <div className="w-full max-w-md bg-white dark:bg-slate-900 rounded-[2rem] shadow-xl shadow-emerald-900/5 dark:shadow-none border border-emerald-900/5 dark:border-slate-800 p-7 sm:p-9">
          {/* Pill tabs */}
          <div className="flex bg-emerald-900/5 dark:bg-slate-800 rounded-full p-1.5 mb-8">
            {['login', 'register'].map(m => (
              <button key={m} type="button" onClick={() => { setMode(m); setError('') }}
                className={`flex-1 py-2.5 rounded-full text-sm font-bold transition-all duration-200 ${
                  mode === m
                    ? 'bg-emerald-700 text-white shadow-md shadow-emerald-700/30'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                }`}>
                {m === 'login' ? 'Login' : 'Register'}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="space-y-5">
            <AuthInput
              label={mode === 'login' ? 'Email or Username' : 'Username'}
              icon="👤" type="text" autoComplete="username" required
              minLength={mode === 'register' ? 3 : 1}
              placeholder={mode === 'login' ? 'nurse.chioma@phc.ng' : 'Choose a username'}
              value={username} onChange={e => setUsername(e.target.value)}
            />
            {mode === 'register' && (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <AuthInput label="First Name" icon="🪪" type="text" autoComplete="given-name" required
                    maxLength={100} placeholder="Chioma"
                    value={firstName} onChange={e => setFirstName(e.target.value)} />
                  <AuthInput label="Last Name" icon="🪪" type="text" autoComplete="family-name" required
                    maxLength={100} placeholder="Adeyemi"
                    value={lastName} onChange={e => setLastName(e.target.value)} />
                </div>
                <AuthInput label="Middle Name (optional)" icon="🪪" type="text" autoComplete="additional-name"
                  maxLength={100} placeholder="Amara"
                  value={middleName} onChange={e => setMiddleName(e.target.value)} />
                <AuthInput label="Email Address" icon="✉️" type="email" autoComplete="email" required
                  placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)} />
              </>
            )}
            <AuthInput label="Password" icon="🔒" type="password" required minLength={6}
              autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
              placeholder="••••••••"
              value={password} onChange={e => setPassword(e.target.value)} />

            {error && (
              <div className="flex items-center gap-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 rounded-2xl px-4 py-3">
                <span className="text-red-500 shrink-0">⚠</span>
                <p className="text-red-600 dark:text-red-400 text-sm">{error}</p>
              </div>
            )}

            <button type="submit" disabled={busy}
              className="w-full py-4 rounded-full font-bold text-white text-base bg-emerald-700 hover:bg-emerald-800 active:bg-emerald-900 disabled:opacity-60 transition-colors shadow-lg shadow-emerald-700/25 flex items-center justify-center gap-2">
              {busy
                ? <span className="w-5 h-5 rounded-full border-2 border-white/40 border-t-white spinner" />
                : <>{mode === 'login' ? 'Access Account' : 'Create Account'} <span aria-hidden>→</span></>}
            </button>
          </form>

          {GOOGLE_CLIENT_ID && (
            <>
              <div className="flex items-center gap-3 my-6">
                <span className="flex-1 h-px bg-emerald-900/10 dark:bg-slate-700" />
                <span className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">or</span>
                <span className="flex-1 h-px bg-emerald-900/10 dark:bg-slate-700" />
              </div>
              <div ref={googleBtnRef} className="flex justify-center" />
            </>
          )}
        </div>

        <p className="text-center text-slate-400 dark:text-slate-600 text-xs mt-7">
          Secure · Private · Voice-first healthcare
        </p>
      </div>
    </div>
  )
}

function AuthInput({ label, icon, ...props }) {
  return (
    <div>
      <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">{label}</label>
      <div className="relative">
        <span className="absolute left-4 top-1/2 -translate-y-1/2 text-base opacity-50" aria-hidden>{icon}</span>
        <input {...props}
          className="w-full bg-emerald-900/5 dark:bg-slate-800 border border-transparent rounded-2xl pl-11 pr-4 py-3.5 text-slate-900 dark:text-white text-sm placeholder-slate-400 dark:placeholder-slate-500 outline-none focus:border-emerald-500 focus:bg-white dark:focus:bg-slate-800 focus:ring-2 focus:ring-emerald-500/15 transition-all" />
      </div>
    </div>
  )
}

// ─── Shell: responsive layout with views ─────────────────────
// Mobile: bottom nav (Listen / Chat / History). Desktop: sidebar + stage.
function Shell({ user, onLogout, onRefresh, dark, onToggleTheme }) {
  const [state, setState]       = useState(S.IDLE)
  const [view, setView]         = useState('listen')     // listen | chat | history
  const [voice, setVoice]       = useState(user.tts_voice || 'Ezinne')
  const [convId, setConvId]     = useState(() => sessionStorage.getItem(CONV_KEY) || null)
  const [turns, setTurns]       = useState([])
  const [convos, setConvos]     = useState(user.conversations || [])
  const [resuming, setResuming] = useState(false)
  const [triage, setTriage]     = useState(null)
  const [chatText, setChatText] = useState('')
  const [chatBusy, setChatBusy] = useState(false)
  const [readingId, setReadingId] = useState(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [predictMsg, setPredictMsg] = useState('')   // transient note when nothing predicted
  const [triagePopup, setTriagePopup] = useState(null) // triage shown in the modal
  const bottomRef   = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef   = useRef([])
  const streamRef   = useRef(null)

  useEffect(() => { setConvos(user.conversations || []) }, [user.conversations])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [turns, view])
  useEffect(() => {
    playReadyCue()
    return () => streamRef.current?.getTracks().forEach(t => t.stop())
  }, [])

  // Restore the active conversation after a page reload
  useEffect(() => {
    const saved = sessionStorage.getItem(CONV_KEY)
    if (saved) restoreTurns(saved)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function restoreTurns(id) {
    try {
      const rows = await getConversationTurns(id)
      if (!rows.length) return
      const t = rows.map(mapHistoryRow)
      setTurns(t)
      const last = [...t].reverse().find(x => x.triage)
      if (last) setTriage(last.triage)
    } catch {}
  }

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
    stopPlayback(); setState(S.IDLE); playReadyCue()
  }, [])

  const ensureConvId = useCallback(() => {
    if (convId) return convId
    const cid = crypto.randomUUID()
    setConvId(cid); sessionStorage.setItem(CONV_KEY, cid)
    return cid
  }, [convId])

  const processBlob = useCallback(async (blob) => {
    setState(S.PROCESSING); playProcessingCue()
    const cid = ensureConvId()
    try {
      const { blob: audio, transcript: t, guidance: g, escalate: e, triage: tr } =
        await consultAudio(await blobToWav(blob), cid)
      setTurns(prev => [...prev, { id: Date.now(), transcript: t, guidance: g, escalate: e, triage: tr, created_at: new Date().toISOString() }])
      if (tr) { setTriage(tr); setTriagePopup(tr) }
      onRefresh()
      await playResp(audio)
    } catch {
      playErrorCue()
      try { await playResp(await speakText(ERR)) } catch { setState(S.IDLE); playReadyCue() }
    }
  }, [playResp, onRefresh, ensureConvId])

  // Text chat — replies stay text; user taps 🔊 to hear them
  async function sendChat(e) {
    e?.preventDefault()
    const msg = chatText.trim()
    if (!msg || chatBusy) return
    setChatBusy(true); setChatText(''); setView('chat')
    const cid = ensureConvId()
    const pendingId = Date.now()
    setTurns(prev => [...prev, { id: pendingId, transcript: msg, guidance: '', pending: true, created_at: new Date().toISOString() }])
    try {
      const res = await consultText(msg, cid)
      setTurns(prev => prev.map(t => t.id === pendingId
        ? { ...t, guidance: res.reply, escalate: res.escalate, triage: res.triage, pending: false }
        : t))
      if (res.triage) { setTriage(res.triage); setTriagePopup(res.triage) }
      onRefresh()
    } catch {
      setTurns(prev => prev.map(t => t.id === pendingId ? { ...t, guidance: ERR, pending: false } : t))
    } finally { setChatBusy(false) }
  }

  async function readAloud(turn) {
    if (readingId || !turn.guidance) return
    setReadingId(turn.id)
    try { await playWavBlob(await speakText(turn.guidance)) } catch {} finally { setReadingId(null) }
  }

  // Manual "Predict now" — forces a fresh prediction on the whole conversation,
  // plus whatever is currently typed in the box but not yet sent.
  async function predictNow() {
    if (predicting) return
    setPredicting(true); setPredictMsg('')
    try {
      const { triage: tr, detail } = await predictTriageNow(convId, chatText.trim() || null)
      if (tr) setTriage(tr)
      else setPredictMsg(detail || 'No clear symptoms detected yet.')
    } catch {
      setPredictMsg('Could not predict right now — please try again.')
    } finally {
      setPredicting(false)
      setTimeout(() => setPredictMsg(''), 4000)
    }
  }

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
    setConvId(null); setTurns([]); setTriage(null); setView('listen')
    sessionStorage.removeItem(CONV_KEY)
  }

  async function resume(id) {
    setResuming(true)
    try {
      const rows = await getConversationTurns(id)
      const t = rows.map(mapHistoryRow)
      setConvId(id); setTurns(t); sessionStorage.setItem(CONV_KEY, id); setView('chat')
      const last = [...t].reverse().find(x => x.triage)
      setTriage(last ? last.triage : null)
    } catch {} finally { setResuming(false) }
  }

  const micHandler = state === S.IDLE ? startRec
    : state === S.LISTENING ? stopRec
    : state === S.SPEAKING ? stopSpeaking : undefined

  const shared = {
    user, state, view, setView, turns, convos, convId, triage, resuming,
    chatText, setChatText, chatBusy, sendChat, readAloud, readingId,
    micHandler, resume, newConvo, bottomRef,
    predictNow, predicting, predictMsg, setTriagePopup,
  }

  return (
    <div className="h-dvh bg-mint dark:bg-slate-950 flex flex-col lg:flex-row overflow-hidden transition-colors">

      {/* ── Desktop sidebar (Stitch: desktop_consultation_light) ── */}
      <aside className="hidden lg:flex flex-col w-72 xl:w-80 shrink-0 bg-white/70 dark:bg-slate-900/70 backdrop-blur border-r border-emerald-900/10 dark:border-slate-800">
        <div className="px-5 py-5 border-b border-emerald-900/5 dark:border-slate-800">
          <Logo />
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-5">
          <p className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-[0.15em] mb-3 px-1">Recent</p>
          {convos.length === 0
            ? <p className="text-sm text-slate-400 dark:text-slate-600 px-1">No consultations yet</p>
            : <ul className="space-y-1.5">
                {convos.map(c => {
                  const active = convId === c.conversation_id
                  return (
                    <li key={c.conversation_id}>
                      <button onClick={() => !active && resume(c.conversation_id)} disabled={resuming}
                        className={`w-full text-left rounded-2xl px-4 py-3 transition-colors ${
                          active
                            ? 'bg-emerald-700/10 dark:bg-emerald-500/10 ring-1 ring-emerald-700/20'
                            : 'hover:bg-emerald-900/5 dark:hover:bg-slate-800'
                        }`}>
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-xs text-slate-400 dark:text-slate-500">{fmtDate(c.last_at)}</p>
                          {c.priority && <UrgencyDot priority={c.priority} />}
                        </div>
                        <p className="text-sm font-semibold text-slate-800 dark:text-slate-200 truncate mt-0.5">
                          {c.first_transcript?.slice(0, 55) || 'Consultation'}
                        </p>
                        {c.department && (
                          <p className="text-xs text-emerald-700/80 dark:text-emerald-400/80 truncate mt-0.5">{c.department}</p>
                        )}
                      </button>
                    </li>
                  )
                })}
              </ul>}
        </div>

        <div className="p-4 border-t border-emerald-900/5 dark:border-slate-800">
          <button onClick={newConvo}
            className="w-full py-3.5 rounded-full bg-emerald-700 hover:bg-emerald-800 text-white text-sm font-bold shadow-lg shadow-emerald-700/25 transition-colors">
            + New Consultation
          </button>
        </div>
      </aside>

      {/* ── Main column ── */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 relative">

        {/* Top bar */}
        <header className="flex items-center justify-between px-4 sm:px-6 py-3.5 shrink-0 bg-white/60 dark:bg-slate-900/60 backdrop-blur border-b border-emerald-900/5 dark:border-slate-800 lg:bg-transparent lg:dark:bg-transparent lg:border-none">
          <div className="lg:hidden"><Logo compact /></div>
          <div className="hidden lg:flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-emerald-700 flex items-center justify-center text-white text-sm font-bold">P</div>
            <div className="leading-tight">
              <p className="text-sm font-bold text-slate-800 dark:text-slate-200">Priscilla</p>
              <p className="text-xs text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" /> Ready to assist
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Desktop view switcher — mobile uses the bottom nav instead */}
            <div className="hidden lg:flex items-center gap-1.5 mr-2">
              {[['listen', 'Voice'], ['chat', 'Chat'], ['history', 'History']].map(([id, label]) => (
                <button key={id} onClick={() => setView(id)}
                  className={`relative px-4 py-2 rounded-full text-sm font-bold transition-colors ${
                    view === id
                      ? 'bg-emerald-700 text-white shadow-md shadow-emerald-700/25'
                      : 'bg-white dark:bg-slate-800 border border-emerald-900/10 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-emerald-50 dark:hover:bg-slate-700'
                  }`}>
                  {label}
                  {id === 'chat' && turns.length > 0 && view !== 'chat' && (
                    <span className="absolute -top-1 -right-1 w-4.5 h-4.5 min-w-4 rounded-full bg-emerald-600 text-white text-[10px] font-bold flex items-center justify-center px-1">{turns.length}</span>
                  )}
                </button>
              ))}
            </div>
            <ThemeToggle dark={dark} onToggle={onToggleTheme} />
            <div className="relative">
              <button onClick={() => setSettingsOpen(o => !o)} aria-label="Settings"
                className="w-10 h-10 rounded-full bg-white dark:bg-slate-800 border border-emerald-900/10 dark:border-slate-700 text-slate-600 dark:text-slate-300 flex items-center justify-center transition-colors hover:bg-emerald-900/5 dark:hover:bg-slate-700">
                ⚙️
              </button>
              {settingsOpen && (
                <div className="absolute right-0 top-12 z-50 w-64 bg-white dark:bg-slate-900 rounded-3xl shadow-2xl shadow-emerald-900/10 border border-emerald-900/10 dark:border-slate-700 p-4">
                  <p className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest mb-2.5">Assistant voice</p>
                  <div className="flex gap-2 mb-4">
                    {[['Ezinne', '👩🏾‍⚕️', 'Female'], ['Abeo', '👨🏾‍⚕️', 'Male']].map(([id, emoji, desc]) => (
                      <button key={id} onClick={() => changeVoice(id)}
                        className={`flex-1 rounded-2xl px-3 py-3 text-center transition-all ${
                          voice === id
                            ? 'bg-emerald-700 text-white shadow-md'
                            : 'bg-emerald-900/5 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-emerald-900/10'
                        }`}>
                        <div className="text-xl mb-0.5">{emoji}</div>
                        <div className="text-xs font-bold">{id}</div>
                        <div className={`text-[10px] ${voice === id ? 'text-emerald-100' : 'text-slate-400'}`}>{desc}</div>
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mb-3 px-1">Signed in as <b className="text-slate-600 dark:text-slate-300">@{user.username}</b></p>
                  <button onClick={onLogout}
                    className="w-full py-2.5 rounded-full border border-red-200 dark:border-red-900/50 text-red-600 dark:text-red-400 text-sm font-bold hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors">
                    Sign out
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Views */}
        {/* overflow-hidden here — each view manages its own scroll area,
            so the chat thread scrolls while its composer stays pinned */}
        <main className="flex-1 overflow-hidden flex flex-col min-h-0">
          {view === 'listen'  && <ListenView {...shared} />}
          {view === 'chat'    && <ChatView {...shared} />}
          {view === 'history' && <HistoryView {...shared} />}
        </main>

        {/* Floating analysis card — desktop only (Stitch mock) */}
        {triage && view === 'listen' && (
          <div className="hidden lg:block absolute bottom-8 right-8 w-80 z-30">
            <TriageCard triage={triage} floating />
          </div>
        )}

        {/* Triage popup — opens when a new prediction arrives, or via the
            "View analysis" button on any past reply */}
        {triagePopup && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
            role="dialog" aria-modal="true" aria-label="Health analysis">
            <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
              onClick={() => setTriagePopup(null)} />
            <div className="relative w-full max-w-sm">
              <TriageCard triage={triagePopup} floating />
              <button onClick={() => setTriagePopup(null)} aria-label="Close analysis"
                className="absolute -top-3 -right-3 w-9 h-9 rounded-full bg-white dark:bg-slate-800 border border-emerald-900/10 dark:border-slate-600 shadow-lg text-slate-500 dark:text-slate-300 font-bold flex items-center justify-center hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors">
                ✕
              </button>
              <button onClick={() => setTriagePopup(null)}
                className="mt-3 w-full py-3 rounded-full bg-emerald-700 hover:bg-emerald-800 text-white text-sm font-bold shadow-lg shadow-emerald-700/25 transition-colors">
                Okay, got it
              </button>
            </div>
          </div>
        )}

        {/* ── Mobile bottom nav (Stitch: chat_interface) ── */}
        <nav className="lg:hidden shrink-0 bg-white dark:bg-slate-900 border-t border-emerald-900/10 dark:border-slate-800 px-6 pb-[max(env(safe-area-inset-bottom),0.5rem)] pt-2 flex items-center justify-around">
          {[
            ['listen', 'Listen', <MicIcon key="i" cls="w-5 h-5" />],
            ['chat', 'Chat', <ChatIcon key="i" cls="w-5 h-5" />],
            ['history', 'History', <HistoryIcon key="i" cls="w-5 h-5" />],
          ].map(([id, label, icon]) => {
            const active = view === id
            return (
              <button key={id} onClick={() => setView(id)}
                className="flex flex-col items-center gap-1 px-4 py-1.5 relative">
                <span className={`flex items-center justify-center w-12 h-8 rounded-full transition-all ${
                  active ? 'bg-emerald-700 text-white shadow-md shadow-emerald-700/30' : 'text-slate-400 dark:text-slate-500'
                }`}>
                  {icon}
                  {id === 'chat' && turns.length > 0 && !active && (
                    <span className="absolute top-0 right-2.5 w-4 h-4 rounded-full bg-emerald-600 text-white text-[9px] font-bold flex items-center justify-center">{turns.length}</span>
                  )}
                </span>
                <span className={`text-[11px] font-bold ${active ? 'text-emerald-700 dark:text-emerald-400' : 'text-slate-400 dark:text-slate-500'}`}>{label}</span>
              </button>
            )
          })}
        </nav>
      </div>
    </div>
  )
}

// ─── Listen view (Stitch: mobile/desktop consultation) ───────
function ListenView({ state, micHandler, triage, chatText, setChatText, chatBusy, sendChat, turns, user, predictNow, predicting, predictMsg }) {
  const cfg = {
    [S.IDLE]:       { label: 'Tap to Speak',  sub: 'Priscilla is ready',   glow: 'mic-idle',      btn: 'bg-emerald-700 hover:bg-emerald-800',       icon: <MicIcon cls="w-14 h-14 lg:w-16 lg:h-16 text-white" /> },
    [S.LISTENING]:  { label: 'Listening…',    sub: 'Tap when you finish',  glow: 'mic-listening', btn: 'bg-rose-600',                               icon: <WaveIcon cls="w-16 h-9 text-white" /> },
    [S.PROCESSING]: { label: 'Thinking…',     sub: 'Please wait',          glow: '',              btn: 'bg-emerald-900/80 cursor-default',          icon: <span className="w-11 h-11 rounded-full border-3 border-white/30 border-t-white spinner" /> },
    [S.SPEAKING]:   { label: 'Speaking…',     sub: 'Tap to stop',          glow: 'mic-speaking',  btn: 'bg-sky-600 hover:bg-sky-700',               icon: <SpeakerIcon cls="w-13 h-13 lg:w-14 lg:h-14 text-white" /> },
  }[state]

  const statusColor = state === S.LISTENING ? 'text-rose-600 dark:text-rose-400'
    : state === S.SPEAKING || state === S.PROCESSING ? 'text-sky-600 dark:text-sky-400'
    : 'text-emerald-700 dark:text-emerald-400'

  return (
    <div className="flex-1 flex flex-col items-center justify-between px-4 py-6 lg:py-10 min-h-0 overflow-y-auto">
      {/* Greeting — desktop gets more headline */}
      <div className="text-center shrink-0">
        <h2 className="hidden lg:block text-3xl xl:text-4xl font-extrabold text-slate-900 dark:text-white tracking-tight">
          How can I help, {user.first_name || user.username}?
        </h2>
        <p className="hidden lg:block text-slate-400 dark:text-slate-500 text-sm mt-2">
          Speak naturally about how you are feeling
        </p>
      </div>

      {/* Mic stage */}
      <div className="flex flex-col items-center gap-7 flex-1 justify-center">
        <button type="button" onClick={micHandler} disabled={state === S.PROCESSING} aria-label={cfg.label}
          className={`w-44 h-44 sm:w-48 sm:h-48 lg:w-56 lg:h-56 rounded-full flex items-center justify-center shadow-2xl shadow-emerald-900/25 transition-all duration-300 active:scale-95 ${cfg.btn} ${cfg.glow}`}>
          {cfg.icon}
        </button>
        <div className="text-center">
          <p className={`text-2xl font-extrabold tracking-tight ${statusColor}`}>{cfg.label}</p>
          <p className="text-slate-400 dark:text-slate-500 text-sm mt-1">{cfg.sub}</p>
        </div>
      </div>

      {/* Bottom cluster: triage chips (mobile) + predict + type bar */}
      <div className="w-full max-w-2xl shrink-0 space-y-4">
        {triage && (
          <div className="lg:hidden flex justify-center">
            <TriageChipsRow triage={triage} />
          </div>
        )}
        {triage?.priority === 'Emergency' && (
          <div className="lg:hidden flex items-center gap-2 justify-center bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 rounded-2xl px-4 py-3">
            <span>🚨</span>
            <p className="text-red-700 dark:text-red-400 text-xs font-bold">This looks urgent — seek care immediately</p>
          </div>
        )}
        {(turns.length > 0 || chatText.trim()) && (
          <div className="flex flex-col items-center gap-1.5">
            <PredictButton onClick={predictNow} predicting={predicting} hasTriage={!!triage} />
            {predictMsg && <p className="text-xs text-slate-500 dark:text-slate-400 text-center">{predictMsg}</p>}
          </div>
        )}

        <form onSubmit={sendChat} className="flex items-center gap-2">
          <div className="flex-1 flex items-center bg-white dark:bg-slate-900 rounded-full shadow-lg shadow-emerald-900/5 border border-emerald-900/10 dark:border-slate-700 pl-5 pr-1.5 py-1.5">
            <input
              value={chatText}
              onChange={e => setChatText(e.target.value)}
              placeholder="Or type symptoms manually…"
              className="flex-1 bg-transparent outline-none text-sm text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 py-2"
            />
            <button type="submit" disabled={chatBusy || !chatText.trim()} aria-label="Send"
              className="shrink-0 w-10 h-10 rounded-full bg-emerald-700 hover:bg-emerald-800 disabled:opacity-40 text-white flex items-center justify-center transition-colors">
              {chatBusy
                ? <span className="w-4 h-4 rounded-full border-2 border-white/40 border-t-white spinner" />
                : <SendIcon cls="w-4 h-4" />}
            </button>
          </div>
        </form>
        {turns.length > 0 && (
          <p className="text-center text-xs text-slate-400 dark:text-slate-500 lg:hidden">
            {turns.length} exchange{turns.length !== 1 ? 's' : ''} in this consultation — see Chat tab
          </p>
        )}
      </div>
    </div>
  )
}

// ─── Chat view (Stitch: chat_interface) ───────────────────────
function ChatView({ turns, chatText, setChatText, chatBusy, sendChat, readAloud, readingId, bottomRef, newConvo, convId, triage, setView, predictNow, predicting, predictMsg, setTriagePopup }) {
  return (
    <div className="flex-1 flex flex-col min-h-0 w-full max-w-3xl mx-auto">
      {/* Chat header: back-to-voice + predict + new conversation */}
      <div className="shrink-0 flex items-center justify-between gap-2 px-4 py-3 border-b border-emerald-900/5 dark:border-slate-800">
        <button onClick={() => setView('listen')}
          className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white dark:bg-slate-800 border border-emerald-900/10 dark:border-slate-700 text-sm font-bold text-emerald-700 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-slate-700 transition-colors shadow-sm">
          <MicIcon cls="w-4 h-4" /> Voice
        </button>
        <div className="flex items-center gap-2 shrink-0">
          <PredictButton onClick={predictNow} predicting={predicting} hasTriage={!!triage} />
          {convId
            ? <button onClick={newConvo}
                className="px-3.5 py-2 rounded-full bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold shadow-sm shadow-emerald-700/25 transition-colors">
                + New
              </button>
            : null}
        </div>
      </div>

      {/* Transient note when a manual predict found nothing */}
      {predictMsg && (
        <div className="shrink-0 px-4 pt-3">
          <p className="text-xs text-slate-500 dark:text-slate-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/40 rounded-xl px-3.5 py-2.5">ℹ️ {predictMsg}</p>
        </div>
      )}


      {/* Thread */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-5 space-y-5">
        {turns.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-3 text-center py-16">
            <div className="w-16 h-16 rounded-full bg-emerald-700/10 flex items-center justify-center text-2xl">💬</div>
            <p className="text-slate-400 dark:text-slate-500 text-sm leading-relaxed">
              No messages yet.<br />Type below or tap Voice to speak.
            </p>
          </div>
        ) : (
          <>
            {turns.map(turn => (
              <div key={turn.id} className="space-y-3">
                {/* Patient bubble — right, filled green */}
                <div className="flex justify-end">
                  <div className="max-w-[85%] sm:max-w-[75%] bg-emerald-700 text-white rounded-3xl rounded-br-lg px-5 py-3.5 shadow-md shadow-emerald-700/15">
                    <p className="text-[15px] leading-relaxed">{turn.transcript || '—'}</p>
                    <p className="text-[10px] text-emerald-200/80 mt-1.5 text-right">{fmtTime(turn.created_at)}</p>
                  </div>
                </div>

                {/* Priscilla bubble — left, neutral */}
                <div className="flex justify-start items-end gap-2">
                  <div className="max-w-[85%] sm:max-w-[75%] bg-white dark:bg-slate-800 border border-emerald-900/5 dark:border-slate-700 rounded-3xl rounded-bl-lg px-5 py-3.5 shadow-sm">
                    <p className="text-xs font-bold text-emerald-700 dark:text-emerald-400 mb-1">Priscilla</p>
                    {turn.pending
                      ? <span className="inline-flex gap-1 py-1">
                          {[0, 1, 2].map(i => <span key={i} className="w-2 h-2 rounded-full bg-emerald-600/50 wave-bar" style={{ animationDelay: `${i * 0.15}s` }} />)}
                        </span>
                      : <p className="text-[15px] text-slate-800 dark:text-slate-200 leading-relaxed">{turn.guidance || '…'}</p>}
                    {turn.triage && !turn.pending && (
                      <button onClick={() => setTriagePopup(turn.triage)}
                        className="mt-2.5 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-700/10 dark:bg-emerald-500/15 text-emerald-800 dark:text-emerald-300 text-xs font-bold hover:bg-emerald-700/20 dark:hover:bg-emerald-500/25 transition-colors">
                        🩺 View analysis
                        {turn.triage.priority === 'Emergency' && <span className="w-2 h-2 rounded-full bg-red-500 mic-listening" />}
                      </button>
                    )}
                    {turn.escalate && (
                      <div className="mt-3 flex items-start gap-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 rounded-2xl px-3.5 py-2.5">
                        <span className="shrink-0">🚨</span>
                        <p className="text-red-700 dark:text-red-400 text-xs font-bold leading-snug">Please seek care at the nearest health facility immediately.</p>
                      </div>
                    )}
                    <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1.5">{fmtTime(turn.created_at)}</p>
                  </div>
                  {turn.guidance && !turn.pending && (
                    <button onClick={() => readAloud(turn)} disabled={readingId !== null}
                      aria-label="Read reply aloud"
                      className={`shrink-0 w-9 h-9 rounded-full flex items-center justify-center transition-colors mb-1 ${
                        readingId === turn.id
                          ? 'bg-sky-100 dark:bg-sky-900/40 text-sky-600'
                          : 'bg-white dark:bg-slate-800 border border-emerald-900/10 dark:border-slate-700 text-slate-400 hover:text-emerald-700 disabled:opacity-40'
                      }`}>
                      <SpeakerIcon cls="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      {/* Composer */}
      <form onSubmit={sendChat}
        className="shrink-0 px-4 pb-4 pt-2">
        <div className="flex items-center bg-white dark:bg-slate-900 rounded-full shadow-lg shadow-emerald-900/5 border border-emerald-900/10 dark:border-slate-700 pl-5 pr-1.5 py-1.5">
          <input
            value={chatText}
            onChange={e => setChatText(e.target.value)}
            placeholder="Or type your symptoms…"
            className="flex-1 bg-transparent outline-none text-sm text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 py-2"
          />
          <button type="submit" disabled={chatBusy || !chatText.trim()} aria-label="Send"
            className="shrink-0 w-10 h-10 rounded-full bg-emerald-700 hover:bg-emerald-800 disabled:opacity-40 text-white flex items-center justify-center transition-colors">
            {chatBusy
              ? <span className="w-4 h-4 rounded-full border-2 border-white/40 border-t-white spinner" />
              : <SendIcon cls="w-4 h-4" />}
          </button>
        </div>
      </form>
    </div>
  )
}

// ─── History view (Stitch: history) — grouped by date ────────
function HistoryView({ convos, convId, resume, resuming, newConvo, setView }) {
  const groups = groupByDate(convos)
  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-4 py-5 w-full max-w-3xl mx-auto">
      {/* Header: back-to-voice + title + new */}
      <div className="flex items-center justify-between gap-3 mb-5">
        <button onClick={() => setView('listen')}
          className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white dark:bg-slate-800 border border-emerald-900/10 dark:border-slate-700 text-sm font-bold text-emerald-700 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-slate-700 transition-colors shadow-sm">
          <MicIcon cls="w-4 h-4" /> Voice
        </button>
        <button onClick={newConvo}
          className="px-4 py-2 rounded-full bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold shadow-md shadow-emerald-700/25 transition-colors shrink-0">
          + New consultation
        </button>
      </div>

      <h2 className="text-2xl font-extrabold text-slate-900 dark:text-white tracking-tight mb-5">Consultation History</h2>

      {convos.length === 0 ? (
        <div className="flex flex-col items-center gap-3 py-20 text-center">
          <div className="w-16 h-16 rounded-full bg-emerald-700/10 flex items-center justify-center text-2xl">📋</div>
          <p className="text-slate-400 dark:text-slate-500 text-sm">No past consultations yet.</p>
        </div>
      ) : (
        <div className="space-y-7">
          {groups.map(([label, items]) => (
            <section key={label}>
              <p className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-[0.15em] mb-3 px-1">{label}</p>
              <ul className="space-y-3">
                {items.map(c => {
                  const active = convId === c.conversation_id
                  return (
                    <li key={c.conversation_id}
                      className={`bg-white dark:bg-slate-900 rounded-3xl border shadow-sm p-4 sm:p-5 transition-colors ${
                        active ? 'border-emerald-500/50 ring-1 ring-emerald-500/20' : 'border-emerald-900/5 dark:border-slate-800'
                      }`}>
                      <div className="flex items-start justify-between gap-3 mb-2.5">
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-slate-400 dark:text-slate-500">{fmtDayTime(c.last_at)}</p>
                          {c.department && (
                            <p className="text-sm font-bold text-emerald-800 dark:text-emerald-400 truncate mt-0.5">{c.department}</p>
                          )}
                        </div>
                        {c.priority && <PriorityPill priority={c.priority} />}
                      </div>
                      <p className="text-[15px] text-slate-700 dark:text-slate-300 leading-snug mb-4 line-clamp-2">
                        “{c.first_transcript?.slice(0, 110) || 'Consultation'}”
                      </p>
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-400 dark:text-slate-500">
                          {c.turn_count} exchange{c.turn_count !== 1 ? 's' : ''}
                        </span>
                        {active
                          ? <span className="flex items-center gap-1.5 text-xs font-bold text-emerald-700 dark:text-emerald-400 px-3 py-1.5">
                              <span className="w-2 h-2 rounded-full bg-emerald-500 mic-idle" /> Active now
                            </span>
                          : <button disabled={resuming} onClick={() => resume(c.conversation_id)}
                              className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold shadow-md shadow-emerald-700/20 disabled:opacity-40 transition-colors">
                              {resuming ? '…' : <>▶ Resume</>}
                            </button>}
                      </div>
                    </li>
                  )
                })}
              </ul>
            </section>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Triage UI ────────────────────────────────────────────────
const PRIORITY_STYLES = {
  Emergency: 'bg-red-600 text-white',
  High:      'bg-orange-500 text-white',
  Moderate:  'bg-red-400/90 text-white',
  Low:       'bg-emerald-600/15 text-emerald-700 dark:text-emerald-400',
}

const BAND_PHRASE = {
  high:   { prefix: 'Current analysis', note: null },
  medium: { prefix: 'Possible analysis', note: 'Preliminary — a health worker should confirm.' },
  low:    { prefix: 'Uncertain analysis', note: 'We are not confident about this — describe more symptoms or see a health worker.' },
}

function PriorityPill({ priority }) {
  const cls = PRIORITY_STYLES[priority] || PRIORITY_STYLES.Low
  return (
    <span className={`px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wide ${cls} ${priority === 'Emergency' ? 'mic-listening' : ''}`}>
      {priority === 'Emergency' ? '! ' : ''}{priority}
    </span>
  )
}

const URGENCY_DOT = {
  Emergency: 'bg-red-600', High: 'bg-orange-500', Moderate: 'bg-amber-500', Low: 'bg-emerald-500',
}
function UrgencyDot({ priority }) {
  return (
    <span className="flex items-center gap-1 shrink-0" title={`${priority} urgency`}>
      <span className={`w-2 h-2 rounded-full ${URGENCY_DOT[priority] || 'bg-slate-400'}`} />
      <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase">{priority}</span>
    </span>
  )
}

function TriageCard({ triage, floating }) {
  const band = triage.confidence_band || 'low'
  const phrase = BAND_PHRASE[band]
  return (
    <div className={`bg-white dark:bg-slate-900 rounded-3xl border border-emerald-900/10 dark:border-slate-700 p-5 ${floating ? 'shadow-2xl shadow-emerald-900/15' : 'shadow-sm'} ${band === 'low' ? 'opacity-90' : ''}`}>
      <div className="flex items-center justify-between gap-3 mb-3">
        <p className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-[0.13em]">🩺 {phrase.prefix}</p>
        <PriorityPill priority={triage.priority} />
      </div>
      <p className="text-lg font-extrabold text-emerald-800 dark:text-emerald-400 leading-tight">{triage.department}</p>
      <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{triage.category}</p>
      <div className="flex items-center gap-2 mt-4">
        <span className="text-[11px] font-semibold text-slate-400 dark:text-slate-500">Confidence</span>
        <ConfidenceDots band={band} />
      </div>
      {phrase.note && <p className="mt-2.5 text-xs text-slate-500 dark:text-slate-400 leading-snug">{phrase.note}</p>}
      {triage.priority === 'Emergency' && (
        <div className="mt-3 flex items-start gap-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 rounded-2xl px-3.5 py-2.5">
          <span className="shrink-0">🚨</span>
          <p className="text-red-700 dark:text-red-400 text-xs font-bold leading-snug">Seek care immediately at the nearest facility.</p>
        </div>
      )}
    </div>
  )
}

function ConfidenceDots({ band }) {
  const filled = band === 'high' ? 3 : band === 'medium' ? 2 : 1
  return (
    <span className="inline-flex gap-1">
      {[0, 1, 2].map(i => (
        <span key={i} className={`w-2 h-2 rounded-full ${i < filled ? 'bg-emerald-600' : 'bg-slate-200 dark:bg-slate-700'}`} />
      ))}
    </span>
  )
}

// Mobile chips row under mic (Stitch mobile mock)
function TriageChipsRow({ triage }) {
  const band = triage.confidence_band || 'low'
  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      <span className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white dark:bg-slate-800 border border-emerald-900/10 dark:border-slate-700 text-xs font-bold text-slate-700 dark:text-slate-300 shadow-sm">
        <span className="w-2 h-2 rounded-full bg-emerald-600" />
        {band !== 'high' && <span className="text-slate-400 font-normal">maybe</span>} {triage.department}
      </span>
      <PriorityPill priority={triage.priority} />
    </div>
  )
}


// ─── Shared UI ────────────────────────────────────────────────
function Logo({ compact }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-9 h-9 rounded-2xl bg-emerald-700 flex items-center justify-center shadow-md shadow-emerald-700/25">
        <MicPlusIcon cls="w-5 h-5 text-white" />
      </div>
      <span className={`font-extrabold tracking-tight text-emerald-900 dark:text-white ${compact ? 'text-lg' : 'text-xl'}`}>
        VoiceMed<span className="text-emerald-600">AI</span>
      </span>
    </div>
  )
}

function PredictButton({ onClick, predicting, hasTriage }) {
  return (
    <button onClick={onClick} disabled={predicting}
      className="flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-emerald-700/10 dark:bg-emerald-500/15 text-emerald-800 dark:text-emerald-300 text-xs font-bold hover:bg-emerald-700/20 dark:hover:bg-emerald-500/25 disabled:opacity-50 transition-colors">
      {predicting
        ? <span className="w-3.5 h-3.5 rounded-full border-2 border-emerald-600/40 border-t-emerald-600 spinner" />
        : <span aria-hidden>🩺</span>}
      {hasTriage ? 'Re-check' : 'Predict now'}
    </button>
  )
}

function ThemeToggle({ dark, onToggle }) {
  return (
    <button onClick={onToggle} aria-label="Toggle theme"
      className="w-10 h-10 rounded-full bg-white dark:bg-slate-800 border border-emerald-900/10 dark:border-slate-700 text-slate-600 dark:text-slate-300 flex items-center justify-center transition-colors hover:bg-emerald-900/5 dark:hover:bg-slate-700">
      {dark ? '☀️' : '🌙'}
    </button>
  )
}

// ─── Helpers ──────────────────────────────────────────────────
function mapHistoryRow(r) {
  return {
    ...r,
    triage: r.triage_category ? {
      category: r.triage_category,
      department: r.triage_department,
      priority: r.triage_priority,
      confidence: r.triage_confidence,
      confidence_band: r.triage_confidence >= 0.44 ? 'high' : r.triage_confidence >= 0.20 ? 'medium' : 'low',
    } : null,
  }
}

function getMimeType() {
  if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) return 'audio/webm;codecs=opus'
  if (MediaRecorder.isTypeSupported('audio/webm')) return 'audio/webm'
  return ''
}

function fmtDate(iso) {
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function fmtTime(iso) {
  return new Date(iso).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

function fmtDayTime(iso) {
  return new Date(iso).toLocaleString(undefined, { weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

// Bucket conversations into Today / Yesterday / Earlier this week / Older,
// preserving the newest-first order the backend already returns.
function groupByDate(convos) {
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const DAY = 86400000
  const buckets = { Today: [], Yesterday: [], 'Earlier this week': [], Older: [] }
  for (const c of convos) {
    const t = new Date(c.last_at).getTime()
    if (t >= startOfToday) buckets.Today.push(c)
    else if (t >= startOfToday - DAY) buckets.Yesterday.push(c)
    else if (t >= startOfToday - 7 * DAY) buckets['Earlier this week'].push(c)
    else buckets.Older.push(c)
  }
  return Object.entries(buckets).filter(([, items]) => items.length > 0)
}

// ─── Icons ────────────────────────────────────────────────────
function MicIcon({ cls }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={cls}>
      <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm0 13a6 6 0 0 0 6-6H16a4 4 0 0 1-8 0H6a6 6 0 0 0 6 6zm-1 3v3h2v-3h-2z" />
    </svg>
  )
}

function MicPlusIcon({ cls }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={cls}>
      <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm0 13a6 6 0 0 0 6-6H16a4 4 0 0 1-8 0H6a6 6 0 0 0 6 6zm-1 3v3h2v-3h-2z" />
      <path d="M19 2h2v2h2v2h-2v2h-2V6h-2V4h2V2z" />
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

function SendIcon({ cls }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={cls}>
      <path d="M2.01 21 23 12 2.01 3 2 10l15 2-15 2z" />
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

function ChatIcon({ cls }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={cls}>
      <path d="M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z" />
    </svg>
  )
}

function HistoryIcon({ cls }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={cls}>
      <path d="M13 3a9 9 0 0 0-9 9H1l3.9 3.9L8.8 12H6a7 7 0 1 1 7 7v2a9 9 0 0 0 0-18zm-1 5v5l4.25 2.52.77-1.28-3.52-2.09V8H12z" />
    </svg>
  )
}
