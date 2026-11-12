import { useCallback, useEffect, useRef, useState } from 'react'
import { Icon } from '../lib/icons.jsx'
import {
  formatDuration,
  segmentsToSrt,
  segmentsToTxt,
  segmentsToVtt,
  downloadBlob,
} from '../lib/formatters.js'
import { startLongDemoJob, startLongJob, streamJob } from '../lib/api.js'
import { VideoDropzone } from '../components/VideoDropzone.jsx'

/**
 * Mode B — Uzun Video (20-25 dk, async job + SSE).
 *
 * Akış:
 *   1. Upload → startLongJob (veya startLongDemoJob) → jobId
 *   2. streamJob(jobId, onEvent) → faz faz progress
 *   3. Done → transcript segmentleri + video player + VTT/SRT/TXT altyazı
 */
export function LongVideoMode({ models, setToast }) {
  const [file, setFile] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [videoMeta, setVideoMeta] = useState(null)
  const [model, setModel] = useState('av-fusion')

  const [jobId, setJobId] = useState(null)
  const [job, setJob] = useState(null) // {status, phase, progress, ...}
  const [transcript, setTranscript] = useState([]) // [{start, end, text}]
  const [currentSegIdx, setCurrentSegIdx] = useState(null)
  const [backendUnavailable, setBackendUnavailable] = useState(false)

  const videoRef = useRef(null)
  const closeStreamRef = useRef(null)
  const timelineRef = useRef(null)

  useEffect(() => {
    if (models.length > 0 && !models.find((m) => m.id === model)) {
      const firstAvailable = models.find((m) => m.available)
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (firstAvailable) setModel(firstAvailable.id)
    }
  }, [models, model])

  // Video meta — loadedmetadata event ile güncelle
  useEffect(() => {
    if (!videoUrl || !videoRef.current) return
    const el = videoRef.current
    const onMeta = () => {
      setVideoMeta({
        duration: el.duration,
        width: el.videoWidth,
        height: el.videoHeight,
      })
    }
    el.addEventListener('loadedmetadata', onMeta)
    return () => el.removeEventListener('loadedmetadata', onMeta)
  }, [videoUrl])

  // Transcript-video senkron
  useEffect(() => {
    const v = videoRef.current
    if (!v || transcript.length === 0) return
    const onTime = () => {
      const t = v.currentTime
      const idx = transcript.findIndex((s) => t >= s.start && t <= s.end)
      if (idx !== currentSegIdx) setCurrentSegIdx(idx)
    }
    v.addEventListener('timeupdate', onTime)
    return () => v.removeEventListener('timeupdate', onTime)
  }, [transcript, currentSegIdx])

  // Otomatik scroll timeline
  useEffect(() => {
    if (currentSegIdx == null) return
    const el = timelineRef.current?.querySelector(`[data-seg="${currentSegIdx}"]`)
    el?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [currentSegIdx])

  const handleFile = useCallback((f) => {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setFile(f)
    setVideoUrl(URL.createObjectURL(f))
    setTranscript([])
    setJob(null)
    setJobId(null)
  }, [videoUrl])

  const clearVideo = () => {
    closeStreamRef.current?.()
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setFile(null)
    setVideoUrl(null)
    setVideoMeta(null)
    setTranscript([])
    setJob(null)
    setJobId(null)
  }

  const attachJobStream = (job_id) => {
    setJobId(job_id)
    setTranscript([])
    const close = streamJob(job_id, (ev) => {
      setJob(ev)
      if (ev.partial_segments) {
        setTranscript((t) => [...t, ...ev.partial_segments])
      }
      if (ev.status === 'done' && ev.transcript) {
        setTranscript(ev.transcript)
      }
    }, (err) => {
      console.error('SSE error', err)
    })
    closeStreamRef.current = close
  }

  const startJob = async () => {
    if (!file) {
      setToast({ type: 'error', message: 'Önce bir video yükleyin veya örnek SSE akışını başlatın' })
      return
    }
    setJob({ status: 'submitting' })
    try {
      const { job_id } = await startLongJob({ file, model })
      attachJobStream(job_id)
    } catch (e) {
      setJob(null)
      setJobId(null)
      if (e.message?.includes('404') || e.message?.toLowerCase().includes('not found')) {
        setBackendUnavailable(true)
      } else {
        setToast({ type: 'error', message: e.message })
      }
    }
  }

  const startDemoJob = async () => {
    setJob({ status: 'submitting' })
    try {
      const { job_id } = await startLongDemoJob({ model })
      attachJobStream(job_id)
    } catch (e) {
      setJob(null)
      setJobId(null)
      setToast({ type: 'error', message: e.message || 'Demo akışı başlatılamadı' })
    }
  }

  const cancelJob = () => {
    closeStreamRef.current?.()
    setJob(null)
    setJobId(null)
  }

  const exportAs = (format) => {
    if (transcript.length === 0) return
    const base = file?.name?.replace(/\.[^.]+$/, '') || 'lippy-altyazi'
    const content =
      format === 'srt' ? segmentsToSrt(transcript) :
      format === 'vtt' ? segmentsToVtt(transcript) :
      segmentsToTxt(transcript)
    const mime = format === 'txt' ? 'text/plain' : 'text/vtt'
    downloadBlob(content, `${base}.${format}`, mime)
  }

  const phases = [
    { id: 'upload', label: 'Yükleme' },
    { id: 'tracking', label: 'Landmark Takip + VAD' },
    { id: 'inference', label: 'Model Çıkarımı' },
  ]

  const isProcessing = job && job.status !== 'done' && job.status !== 'error' && job.status !== 'canceled'

  return (
    <div className="long-mode">
      <VideoDropzone
        file={file}
        videoUrl={videoUrl}
        videoMeta={videoMeta}
        videoRef={videoRef}
        onFile={handleFile}
        onClear={clearVideo}
        maxSizeMB={500}
        placeholder="Uzun video yükle (20-25 dk)"
        hint="MP4, WebM (max 500 MB)"
        disabled={isProcessing}
      />

      <section className="card">
        <div className="card-header">
          <h2 className="card-title">İşlem</h2>
          {jobId && (
            <code style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              job: {jobId.slice(0, 8)}…
            </code>
          )}
        </div>
        <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="field">
            <label className="field-label">Model</label>
            <select
              className="select"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isProcessing}
            >
              {models.length === 0 && <option value="video">yükleniyor…</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name}{!m.available ? ' — checkpoint yok' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Faz progress barları */}
          {job && (
            <div className="phases">
              {phases.map((p) => {
                const isActive = job.phase === p.id
                const isDone = job.status === 'done' || phases.findIndex((x) => x.id === job.phase) > phases.findIndex((x) => x.id === p.id)
                const pct = isDone ? 100 : isActive ? (job.progress ?? 0) : 0
                return (
                  <div key={p.id} className={`phase-row ${isActive ? 'active' : ''} ${isDone ? 'done' : ''}`}>
                    <div className="phase-label">
                      {isDone ? Icon.check2 : isActive ? <span className="spinner" /> : Icon.clock}
                      <span>{p.label}</span>
                    </div>
                    <div className="phase-bar">
                      <div className="phase-bar-fill" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="phase-pct">{Math.round(pct)}%</span>
                  </div>
                )
              })}
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {!isProcessing ? (
              <>
                <button
                  type="button"
                  className="btn btn-primary btn-full btn-lg"
                  onClick={startJob}
                  disabled={!file || models.filter((m) => m.available).length === 0}
                >
                  {Icon.play} İşlemi Başlat (Yüklenen Video)
                </button>
                <button
                  type="button"
                  className="btn btn-secondary btn-full"
                  onClick={startDemoJob}
                  disabled={models.filter((m) => m.available).length === 0}
                >
                  {Icon.play} Örnek Uzun Video Altyazı Akışını Başlat (4 Gerçek Klip SSE)
                </button>
              </>
            ) : (
              <button
                type="button"
                className="btn btn-secondary btn-full btn-lg"
                onClick={cancelJob}
              >
                {Icon.stop} İptal
              </button>
            )}
          </div>

          {backendUnavailable && (
            <div className="alert alert-warning">
              {Icon.alert}
              <div>
                <strong>Uzun video servisi henüz aktif değil.</strong>
                <br />
                <code>/api/infer_long</code> endpoint'i hazır olduğunda bu mod kullanılabilir.
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Transcript + export */}
      {transcript.length > 0 && (
        <section className="card" style={{ gridColumn: '1 / -1' }}>
          <div className="card-header">
            <h2 className="card-title">
              Altyazı ({transcript.length} segment)
            </h2>
            <div style={{ display: 'flex', gap: '0.25rem' }}>
              <button type="button" className="btn btn-ghost btn-icon" onClick={() => exportAs('vtt')} title=".vtt indir">
                {Icon.download} VTT
              </button>
              <button type="button" className="btn btn-ghost btn-icon" onClick={() => exportAs('srt')} title=".srt indir">
                {Icon.download} SRT
              </button>
              <button type="button" className="btn btn-ghost btn-icon" onClick={() => exportAs('txt')} title=".txt indir">
                {Icon.download} TXT
              </button>
            </div>
          </div>
          <div className="card-body">
            <div className="timeline" ref={timelineRef}>
              {transcript.map((s, i) => (
                <button
                  key={i}
                  data-seg={i}
                  type="button"
                  className={`timeline-segment ${currentSegIdx === i ? 'current' : ''}`}
                  onClick={() => {
                    if (videoRef.current) videoRef.current.currentTime = s.start
                  }}
                >
                  <span className="timeline-time">
                    {formatDuration(s.start)} → {formatDuration(s.end)}
                  </span>
                  <span className="timeline-text">{s.text}</span>
                </button>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
