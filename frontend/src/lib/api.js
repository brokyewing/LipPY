/**
 * LipPY API client — tüm backend çağrıları buradan.
 *
 * Endpoint'ler:
 *   GET  /api/health               → sunucu canlılık
 *   GET  /api/models               → mevcut modeller
 *   POST /api/infer                → kısa video (sync)
 *   POST /api/infer_long           → uzun video (async, job_id döner)
 *   GET  /api/jobs/:id             → job status polling
 *   GET  /api/jobs/:id/stream      → SSE progress
 *   GET  /api/jobs/:id/subtitles.vtt → WebVTT altyazı
 *   WS   /ws/live                  → canlı webcam stream
 */

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const wsProto = (() => {
  if (typeof window === 'undefined') return 'ws'
  return window.location.protocol === 'https:' ? 'wss' : 'ws'
})()

const apiHost = API_BASE.replace(/^https?:\/\//, '')
export const WS_BASE = `${wsProto}://${apiHost}`

async function parseJSON(res) {
  try {
    return await res.json()
  } catch {
    return null
  }
}

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options)
  const data = await parseJSON(res)
  if (!res.ok) {
    throw new Error(data?.detail || data?.message || `HTTP ${res.status}`)
  }
  return data
}

// ─── Health / Models ────────────────────────────────────

export const getHealth = () => request('/api/health')

export const getModels = () => request('/api/models').then((d) => d?.models || [])

// ─── Short video (sync) ─────────────────────────────────

export async function inferShort({ file, model = 'video', audioFile = null, decodeMode = null }) {
  const form = new FormData()
  form.append('file', file)
  form.append('model', model)
  if (audioFile) {
    form.append('audio', audioFile)
  }
  if (decodeMode) {
    form.append('decode_mode', decodeMode)
  }
  return request('/api/infer', { method: 'POST', body: form })
}

// ─── Long video (async + SSE) ───────────────────────────

export async function startLongJob({ file, model = 'video' }) {
  const form = new FormData()
  form.append('file', file)
  form.append('model', model)
  return request('/api/infer_long', { method: 'POST', body: form })
}

export const getJob = (jobId) => request(`/api/jobs/${jobId}`)

/**
 * EventSource stream — progress + partial transcript.
 * Kapatmak için döndürülen close() fonksiyonunu çağır.
 */
export function streamJob(jobId, onEvent, onError) {
  const es = new EventSource(`${API_BASE}/api/jobs/${jobId}/stream`)
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      onEvent?.(data)
    } catch (err) {
      onError?.(err)
    }
  }
  es.onerror = (err) => {
    onError?.(err)
    es.close()
  }
  return () => es.close()
}

export const getSubtitlesUrl = (jobId) => `${API_BASE}/api/jobs/${jobId}/subtitles.vtt`

// ─── Live WebSocket ─────────────────────────────────────

/**
 * WebSocket bağlantısı aç — live mode için.
 * onMessage: {type: 'raw'|'corrected'|'error', text?, message?, timestamp?}
 */
export function openLiveSocket({ model = 'av-fusion', onOpen, onMessage, onClose, onError } = {}) {
  const ws = new WebSocket(`${WS_BASE}/ws/live?model=${encodeURIComponent(model)}`)
  ws.binaryType = 'arraybuffer'
  ws.onopen = () => onOpen?.()
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage?.(data)
    } catch (err) {
      onError?.(err)
    }
  }
  ws.onclose = (event) => onClose?.(event)
  ws.onerror = (err) => onError?.(err)
  return ws
}
