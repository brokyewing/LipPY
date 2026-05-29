import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Icon } from '../lib/icons.jsx'
import { openLiveSocket } from '../lib/api.js'

/**
 * Mode C — Canlı Webcam (WebSocket + live caption).
 *
 * Akış:
 *   1. Model seç (AV-Fusion=sesli / v1cc=sessiz)
 *   2. getUserMedia (webcam; ses model requires_audio'ya göre)
 *   3. WebSocket aç (model query param)
 *   4. Her ~3 sn BAĞIMSIZ tam WebM segment → WS'e binary gönder
 *      (fragmented chunk değil — backend her segmenti tek başına decode eder)
 *   5. Server segment → ROI + inference → {type:'raw', text}
 *
 * NOT: ~5sn gecikme normal (Whisper + decode + ROI). Sessiz modda video-only kalite.
 */
const SEGMENT_MS = 3000

export function LiveCaptureMode({ models, setToast }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const recorderRef = useRef(null)
  const wsRef = useRef(null)
  const stoppedRef = useRef(false)

  const [model, setModel] = useState('av-fusion')
  const [isRecording, setIsRecording] = useState(false)
  const [status, setStatus] = useState('idle') // idle | requesting | live | error
  const [rawCaption, setRawCaption] = useState('')
  const [history, setHistory] = useState([])
  const [error, setError] = useState(null)
  const [backendUnavailable, setBackendUnavailable] = useState(false)

  const availableModels = useMemo(() => models.filter((m) => m.available), [models])
  const selectedModel = models.find((m) => m.id === model)
  const useAudio = !!selectedModel?.requires_audio   // av-fusion → sesli, v1cc → sessiz

  // İlk available modeli seç
  useEffect(() => {
    if (models.length > 0 && !models.find((m) => m.id === model)) {
      const first = models.find((m) => m.available)
      if (first) setModel(first.id)
    }
  }, [models, model])

  const getSupportedMime = () => {
    const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
    for (const c of candidates) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(c)) return c
    }
    return null
  }

  const cleanup = useCallback(() => {
    stoppedRef.current = true
    try {
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
    } catch { /* ignore */ }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    wsRef.current?.close()
    streamRef.current = null
    recorderRef.current = null
    wsRef.current = null
  }, [])

  useEffect(() => () => cleanup(), [cleanup])

  const start = async () => {
    setError(null)
    setBackendUnavailable(false)
    setRawCaption('')
    setStatus('requesting')
    stoppedRef.current = false

    const mime = getSupportedMime()
    if (!mime) {
      setError('Tarayıcı WebM/MediaRecorder desteklemiyor')
      setStatus('error')
      return
    }

    // Webcam (ses model'e göre)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 25 }, facingMode: 'user' },
        audio: useAudio,
      })
      streamRef.current = stream
      videoRef.current.srcObject = stream
    } catch (e) {
      setError(`Webcam erişimi reddedildi: ${e.message}`)
      setStatus('error')
      return
    }

    // WebSocket (model query param)
    const ws = openLiveSocket({
      model,
      onOpen: () => {
        setStatus('live')
        setIsRecording(true)
        recordLoop(mime, ws)
      },
      onMessage: (data) => {
        if (data.type === 'raw') {
          setRawCaption(data.text)
          setHistory((h) => [...h, { text: data.text, time: data.timestamp || Date.now() / 1000 }])
        } else if (data.type === 'error') {
          setError(data.message)
        }
      },
      onClose: (event) => {
        setIsRecording(false)
        setStatus('idle')
        if (event.code === 1006 || event.code === 1011) setBackendUnavailable(true)
      },
      onError: () => {
        setBackendUnavailable(true)
        setStatus('error')
        cleanup()
      },
    })
    wsRef.current = ws
  }

  // Her SEGMENT_MS'de BAĞIMSIZ tam WebM segment kaydet + gönder (fragmented fix)
  const recordLoop = (mime, ws) => {
    if (stoppedRef.current || !streamRef.current || ws.readyState !== WebSocket.OPEN) return
    let rec
    try {
      rec = new MediaRecorder(streamRef.current, { mimeType: mime, videoBitsPerSecond: 2_000_000 })
    } catch (e) {
      setError(`MediaRecorder başlatılamadı: ${e.message}`)
      cleanup()
      return
    }
    const chunks = []
    rec.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data) }
    rec.onstop = async () => {
      if (chunks.length && ws.readyState === WebSocket.OPEN) {
        try {
          const blob = new Blob(chunks, { type: mime })
          const buf = await blob.arrayBuffer()
          ws.send(buf)   // tam bağımsız WebM segment (header dahil → decode edilebilir)
        } catch { /* ignore */ }
      }
      if (!stoppedRef.current) recordLoop(mime, ws)   // sonraki segment
    }
    recorderRef.current = rec
    rec.start()
    setTimeout(() => { try { if (rec.state === 'recording') rec.stop() } catch { /* ignore */ } }, SEGMENT_MS)
  }

  const stop = () => {
    cleanup()
    setIsRecording(false)
    setStatus('idle')
    setRawCaption('')
  }

  return (
    <div className="live-mode">
      <section className="card">
        <div className="card-header">
          <h2 className="card-title">Canlı Çekim</h2>
          {isRecording && (
            <span className="live-indicator"><span className="live-dot" /> LIVE</span>
          )}
        </div>

        <div className="card-body">
          {/* Model seçimi */}
          <div className="field" style={{ marginBottom: '1rem' }}>
            <label className="field-label" htmlFor="live-model">Model</label>
            <select
              id="live-model"
              className="select"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isRecording}
            >
              {models.length === 0 && <option value="av-fusion">yükleniyor…</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name}{!m.available ? ' — checkpoint yok' : ''}
                </option>
              ))}
            </select>
            <span className="field-hint">
              {useAudio
                ? '🎤 Sesli mod — mikrofon açık (AV-Fusion tam güç)'
                : '🔇 Sessiz mod — sadece video (video-only kalite)'}
              {' · ~5sn gecikme normaldir'}
            </span>
          </div>

          <div className="live-video-container">
            <video ref={videoRef} autoPlay muted playsInline className="live-video" />
            <div className="live-caption-overlay">
              {rawCaption ? (
                <div className="caption raw">{rawCaption}</div>
              ) : isRecording ? (
                <div className="caption pending">Dinleniyor… (~{SEGMENT_MS / 1000}sn segment)</div>
              ) : null}
            </div>
          </div>

          <div className="live-controls" style={{ marginTop: '1rem' }}>
            {!isRecording ? (
              <button
                type="button"
                className="btn btn-primary btn-full btn-lg"
                onClick={start}
                disabled={status === 'requesting' || availableModels.length === 0}
              >
                {status === 'requesting' ? (<><span className="spinner" /> Webcam izni…</>) : (<>{Icon.record} Kayıt Başlat</>)}
              </button>
            ) : (
              <button type="button" className="btn btn-secondary btn-full btn-lg" onClick={stop}>
                {Icon.stop} Durdur
              </button>
            )}
          </div>

          {error && (
            <div className="alert alert-error" style={{ marginTop: '1rem' }}>
              {Icon.alert}<div>{error}</div>
            </div>
          )}

          {backendUnavailable && !error && (
            <div className="alert alert-warning" style={{ marginTop: '1rem' }}>
              {Icon.alert}
              <div>
                <strong>Canlı servis bağlantısı kesildi.</strong><br />
                <code>/ws/live</code> WebSocket'ine ulaşılamadı (backend çalışıyor mu?).
              </div>
            </div>
          )}
        </div>
      </section>

      {history.length > 0 && (
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Geçmiş Altyazılar ({history.length})</h2>
            <button type="button" className="btn btn-ghost btn-icon" onClick={() => setHistory([])} title="Geçmişi temizle">
              {Icon.trash}
            </button>
          </div>
          <div className="card-body">
            <div className="live-history">
              {history.slice(-50).reverse().map((h, i) => (
                <div key={i} className="history-item">
                  <span className="history-time">{new Date(h.time * 1000).toLocaleTimeString()}</span>
                  <span className="history-text">{h.text}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
