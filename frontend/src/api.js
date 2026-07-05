/**
 * VoiceMedAI API client
 *
 * All requests automatically attach the Bearer token from localStorage
 * when available. Auth methods manage the token lifecycle.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const API_TIMEOUT_MS = 600_000

const TOKEN_KEY = 'voicemed_token'

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

export function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY) || null
}

export function setStoredToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
}

// ---------------------------------------------------------------------------
// Base fetch with automatic auth injection
// ---------------------------------------------------------------------------

async function apiFetch(path, options = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), API_TIMEOUT_MS)

  const token = getStoredToken()
  const headers = { ...(options.headers || {}) }
  if (token) headers['Authorization'] = `Bearer ${token}`

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
      signal: controller.signal,
    })
    return response
  } finally {
    clearTimeout(timer)
  }
}

// ---------------------------------------------------------------------------
// Auth endpoints
// ---------------------------------------------------------------------------

export async function registerUser(username, email, password, firstName, middleName, lastName) {
  const res = await apiFetch('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      username, email, password,
      first_name: firstName,
      middle_name: middleName || null,
      last_name: lastName,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Registration failed' }))
    throw new Error(_extractDetail(err, 'Registration failed'))
  }
  return res.json()
}

export async function loginUser(identifier, password) {
  const res = await apiFetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ identifier, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Login failed' }))
    throw new Error(_extractDetail(err, 'Login failed'))
  }
  const data = await res.json()
  setStoredToken(data.access_token)
  return data
}

/** Parse FastAPI detail — handles both string and Pydantic v2 array formats */
function _extractDetail(err, fallback) {
  if (!err || !err.detail) return fallback
  if (typeof err.detail === 'string') return err.detail
  if (Array.isArray(err.detail)) {
    return err.detail
      .map((e) => {
        const field = e.loc?.slice(1).join('.') || ''
        return field ? `${field}: ${e.msg}` : e.msg
      })
      .join('; ')
  }
  return fallback
}

/** Exchange a Google Identity Services credential for our JWT. */
export async function loginWithGoogle(credential) {
  const res = await apiFetch('/auth/google', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ credential }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Google sign-in failed' }))
    throw new Error(_extractDetail(err, 'Google sign-in failed'))
  }
  const data = await res.json()
  setStoredToken(data.access_token)
  return data
}

export async function logoutUser() {
  const token = getStoredToken()
  if (token) {
    // Best-effort server-side invalidation
    await apiFetch('/auth/logout-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    }).catch(() => {})
  }
  setStoredToken(null)
}

export async function getUserProfile() {
  const res = await apiFetch('/auth/me')
  if (res.status === 401) return null
  if (!res.ok) throw new Error('Failed to fetch profile')
  return res.json()
}

// ---------------------------------------------------------------------------
// Consultation endpoints
// ---------------------------------------------------------------------------

export async function transcribeAudio(blob) {
  const form = new FormData()
  const name = blob.type?.includes('wav') ? 'query.wav' : 'query.webm'
  form.append('audio', blob, name)
  const res = await apiFetch('/transcribe', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`transcribe_failed:${res.status}`)
  return res.json()
}

export async function reasonQuery(query) {
  const res = await apiFetch('/reason', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error(`reason_failed:${res.status}`)
  return res.json()
}

export async function speakText(text) {
  const res = await apiFetch('/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`speak_failed:${res.status}`)
  return res.blob()
}

export async function consultAudio(blob, conversationId = null) {
  const form = new FormData()
  const name = blob.type?.includes('wav') ? 'query.wav' : 'query.webm'
  form.append('audio', blob, name)
  if (conversationId) form.append('conversation_id', conversationId)
  const res = await apiFetch('/consult', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`consult_failed:${res.status}`)

  const rawTranscript = res.headers.get('X-VoiceMed-Transcript') || ''
  const rawGuidance = res.headers.get('X-VoiceMed-Guidance') || ''
  const transcript = decodeURIComponent(rawTranscript)
  const guidance = decodeURIComponent(rawGuidance)
  const escalate = res.headers.get('X-VoiceMed-Escalate') === 'true'
  const returnedConversationId = res.headers.get('X-VoiceMed-ConversationId') || conversationId

  let triage = null
  const rawTriage = res.headers.get('X-VoiceMed-Triage') || ''
  if (rawTriage) {
    try { triage = JSON.parse(decodeURIComponent(rawTriage)) } catch { /* ignore malformed */ }
  }

  const audioBlob = await res.blob()
  if (audioBlob.size < 100) throw new Error(`empty_audio:${audioBlob.size}`)

  return { blob: audioBlob, transcript, guidance, escalate, triage, conversationId: returnedConversationId }
}

/** Text chat — fast path, no audio. Returns { reply, triage, escalate, conversation_id } */
export async function consultText(message, conversationId = null) {
  const res = await apiFetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_id: conversationId }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Chat failed' }))
    throw new Error(_extractDetail(err, 'Chat failed'))
  }
  return res.json()
}

/** Force a triage prediction for the whole conversation (manual button).
 *  Returns { triage, detail } — triage is null when nothing could be predicted. */
export async function predictTriageNow(conversationId = null, message = null) {
  const res = await apiFetch('/triage', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_id: conversationId, message }),
  })
  if (!res.ok) throw new Error(`triage_failed:${res.status}`)
  return res.json()
}

export async function getConversationTurns(conversationId) {
  const res = await apiFetch(`/auth/conversations/${conversationId}`)
  if (!res.ok) throw new Error('Failed to load conversation')
  return res.json()
}

export async function updateUserVoice(voice) {
  const res = await apiFetch('/auth/voice', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ voice }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to update voice preference' }))
    throw new Error(err.detail || 'Failed to update voice preference')
  }
  return res.json()
}

