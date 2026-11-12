/**
 * LipPY API client — tüm backend çağrıları buradan.
 *
 * Endpoint'ler:
 *   GET  /api/health                 → sunucu canlılık
 *   GET  /api/models                 → mevcut modeller
 *   GET  /api/samples                → hazır Türkçe tez benchmark örnekleri
 *   POST /api/infer                  → kısa video (sync)
 *   POST /api/infer_sample           → hazır örnek klip çıkarımı (sync)
 *   POST /api/infer_long             → uzun video (async, job_id döner)
 *   POST /api/infer_long_demo        → uzun video SSE demo (async, job_id döner)
 *   GET  /api/jobs/:id               → job status polling
 *   GET  /api/jobs/:id/stream        → SSE progress
 *   GET  /api/jobs/:id/subtitles.vtt → WebVTT altyazı
 *   WS   /ws/live                    → canlı webcam stream
 */

function resolveApiBase() {
  const envBase = import.meta.env.VITE_API_BASE
  if (envBase) return envBase.replace(/\/$/, '')
  if (typeof window === 'undefined') return 'http://localhost:8000'
  if (window.location.port === '5173') return 'http://localhost:8000'
  const path = window.location.pathname
  const match = path.match(/^(\/demos\/[^/]+)/)
  return match ? match[1] : ''
}

export const API_BASE = resolveApiBase()

const wsProto = (() => {
  if (typeof window === 'undefined') return 'ws'
  return window.location.protocol === 'https:' ? 'wss' : 'ws'
})()

export const WS_BASE = (() => {
  if (API_BASE.startsWith('http://') || API_BASE.startsWith('https://')) {
    return `${wsProto}://${API_BASE.replace(/^https?:\/\//, '')}`
  }
  if (typeof window === 'undefined') return 'ws://localhost:8000'
  return `${wsProto}://${window.location.host}${API_BASE}`
})()

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

// ─── Health / Models / Samples ──────────────────────────

export const getHealth = () => request('/api/health')

export const getModels = () => request('/api/models').then((d) => d?.models || [])

export const getSamples = () => request('/api/samples')

export const getSampleVideoUrl = (sampleId) =>
  `${API_BASE}/api/samples/${encodeURIComponent(sampleId)}/video.mp4`

// ─── Short video (sync) & Sample Benchmark ──────────────

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

export async function inferSample({ sampleId, model = 'video', decodeMode = null }) {
  const form = new FormData()
  form.append('sample_id', sampleId)
  form.append('model', model)
  if (decodeMode) {
    form.append('decode_mode', decodeMode)
  }
  return request('/api/infer_sample', { method: 'POST', body: form })
}

// ─── Long video (async + SSE) ───────────────────────────

export async function startLongJob({ file, model = 'video' }) {
  const form = new FormData()
  form.append('file', file)
  form.append('model', model)
  return request('/api/infer_long', { method: 'POST', body: form })
}

export async function startLongDemoJob({ model = 'av-fusion' } = {}) {
  const form = new FormData()
  form.append('model', model)
  return request('/api/infer_long_demo', { method: 'POST', body: form })
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
      if (data.status === 'done' || data.status === 'error' || data.status === 'canceled') {
        es.close()
      }
    } catch (err) {
      onError?.(err)
    }
  }
  es.onerror = (err) => {
    if (es.readyState !== EventSource.CLOSED) {
      onError?.(err)
      es.close()
    }
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
