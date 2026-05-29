import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Icon } from '../lib/icons.jsx'
import { formatMs } from '../lib/formatters.js'
import { inferShort } from '../lib/api.js'
import { VideoDropzone } from '../components/VideoDropzone.jsx'
import { ResultCard } from '../components/ResultCard.jsx'
import { MetricsGrid } from '../components/MetricsGrid.jsx'

// Decode modu etiketleri (kullanici dostu)
const DECODE_LABELS = {
  greedy: 'Greedy (hızlı, LM yok)',
  beam_lm: 'Beam + KenLM (bağlam)',
  attn: 'Attention decoder',
  greedy_lex: 'Greedy + Sözlük düzeltme',
  beam_lm_lex: 'Beam+KenLM + Sözlük',
  attn_lex: 'Attention + Sözlük',
}

/**
 * Mode A — Kısa Video (5-60 saniye klip, sync response).
 *
 * Props:
 *   models: [{id, name, available}]
 *   setToast(toast)
 */
export function ShortVideoMode({ models, setToast }) {
  const [file, setFile] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [videoMeta, setVideoMeta] = useState(null)
  const [model, setModel] = useState('video')
  const [decodeMode, setDecodeMode] = useState('greedy')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const videoRef = useRef(null)
  const audioInputRef = useRef(null)

  // Model otomatik seçimi (ilk available)
  useEffect(() => {
    if (models.length > 0 && !models.find((m) => m.id === model)) {
      const firstAvailable = models.find((m) => m.available)
      if (firstAvailable) setModel(firstAvailable.id)
    }
  }, [models, model])

  // Model değişince decode modunu o modelin default'una ayarla (en iyi mod gelsin)
  useEffect(() => {
    const sm = models.find((m) => m.id === model)
    if (sm?.default_decode) setDecodeMode(sm.default_decode)
  }, [model, models])

  // Video meta toplama
  useEffect(() => {
    if (!videoUrl || !videoRef.current) {
      setVideoMeta(null)
      return
    }
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

  const handleFile = useCallback((f) => {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setFile(f)
    setVideoUrl(URL.createObjectURL(f))
    setResult(null)
  }, [videoUrl])

  const clearVideo = () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setFile(null)
    setVideoUrl(null)
    setVideoMeta(null)
    setResult(null)
  }

  const handleInfer = async () => {
    if (!file) {
      setToast({ type: 'error', message: 'Önce bir video yükleyin' })
      return
    }
    setLoading(true)
    setResult(null)
    try {
      const sm = models.find((m) => m.id === model)
      const useDecode = sm?.decode_modes?.includes(decodeMode) ? decodeMode : null
      const data = await inferShort({ file, model, audioFile: needsAudio ? audioFile : null, decodeMode: useDecode })
      setResult(data)
      if (!data.success) {
        setToast({
          type: 'error',
          message: data.processing?.error || 'Tahmin başarısız',
        })
      }
    } catch (e) {
      setToast({
        type: 'error',
        message: e.message || 'Backend erişilemez. Backend çalışıyor mu?',
      })
    } finally {
      setLoading(false)
    }
  }

  const availableModels = useMemo(() => models.filter((m) => m.available), [models])
  const selectedModel = models.find((m) => m.id === model)
  const needsAudio = selectedModel?.requires_audio || model === 'av-fusion'

  return (
    <div className="mode-grid">
      <VideoDropzone
        file={file}
        videoUrl={videoUrl}
        videoMeta={videoMeta}
        videoRef={videoRef}
        onFile={handleFile}
        onClear={clearVideo}
        maxSizeMB={50}
        placeholder="Video yükle veya buraya sürükle"
        hint="MP4, WebM (5-60 saniye ideal, max 50 MB)"
      />

      <div className="right-panel">
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Tahmin</h2>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="field">
              <label className="field-label" htmlFor="model-select">Model</label>
              <select
                id="model-select"
                className="select"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {models.length === 0 && <option value="video">yükleniyor…</option>}
                {models.map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.name}{!m.available ? ' — checkpoint yok' : ''}
                  </option>
                ))}
              </select>
              {selectedModel && !selectedModel.available && (
                <span className="field-hint" style={{ color: 'var(--warning)' }}>
                  Bu model için checkpoint bulunamadı.
                </span>
              )}
            </div>

            {selectedModel?.decode_modes?.length > 0 && (
              <div className="field">
                <label className="field-label" htmlFor="decode-select">Decode Modu</label>
                <select
                  id="decode-select"
                  className="select"
                  value={decodeMode}
                  onChange={(e) => setDecodeMode(e.target.value)}
                >
                  {selectedModel.decode_modes.map((dm) => (
                    <option key={dm} value={dm}>{DECODE_LABELS[dm] || dm}</option>
                  ))}
                </select>
                <span className="field-hint">greedy / beam_lm / _lex (sözlük) kıyaslamak için değiştir</span>
              </div>
            )}

            {needsAudio && (
              <div className="field">
                <label className="field-label">Ses Dosyası (opsiyonel)</label>
                <input
                  ref={audioInputRef}
                  type="file"
                  accept="audio/*,.wav,.mp3,.m4a"
                  className="input"
                  onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
                />
                <span className="field-hint">
                  {audioFile
                    ? `${audioFile.name} (${(audioFile.size / 1024 / 1024).toFixed(1)} MB)`
                    : 'Boş bırakırsan video\'dan ses otomatik çıkarılır'
                  }
                </span>
              </div>
            )}

            <button
              type="button"
              className="btn btn-primary btn-full btn-lg"
              onClick={handleInfer}
              disabled={loading || !file || availableModels.length === 0}
            >
              {loading ? (<><span className="spinner" /> İşleniyor…</>) : (<>{Icon.play} Tahmin Yap</>)}
            </button>

            {loading && <div className="progress-indeterminate" />}
          </div>
        </section>

        <ResultCard
          rawText={result?.success ? result.text : null}
          rightMeta={result?.success ? formatMs(result.processing?.total_ms) : null}
          emptyHint={
            result && !result.success
              ? 'Tahmin başarısız.'
              : 'Video yükleyip tahmin başlat.'
          }
        />

        {result?.processing && <MetricsGrid processing={result.processing} />}
      </div>
    </div>
  )
}
