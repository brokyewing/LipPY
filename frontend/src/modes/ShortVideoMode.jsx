import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Icon } from '../lib/icons.jsx'
import { formatMs } from '../lib/formatters.js'
import { getSamples, getSampleVideoUrl, inferSample, inferShort } from '../lib/api.js'
import { VideoDropzone } from '../components/VideoDropzone.jsx'
import { ResultCard } from '../components/ResultCard.jsx'
import { MetricsGrid } from '../components/MetricsGrid.jsx'

// Decode modu etiketleri (kullanici dostu)
const DECODE_LABELS = {
  greedy: 'Greedy (hızlı, LM yok)',
  beam: 'Beam Search (k=10)',
  beam_lm: 'Beam + KenLM (5-gram bağlam)',
  attn: 'Attention decoder (Seq2Seq)',
  greedy_lex: 'Greedy + Sözlük düzeltme',
  beam_lm_lex: 'Beam+KenLM + Sözlük',
  attn_lex: 'Attention + Sözlük',
}

/**
 * Mode A — Kısa Video (5-60 saniye klip, sync response + hazır tez örnekleri).
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

  const [samples, setSamples] = useState([])
  const [selectedSampleId, setSelectedSampleId] = useState(null)

  const videoRef = useRef(null)
  const audioInputRef = useRef(null)

  // Hazır Türkçe tez örneklerini yükle
  useEffect(() => {
    let alive = true
    getSamples()
      .then((data) => {
        if (!alive) return
        setSamples(data?.samples || [])
      })
      .catch(() => {
        /* ignore */
      })
    return () => { alive = false }
  }, [])

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
    if (videoUrl?.startsWith('blob:')) URL.revokeObjectURL(videoUrl)
    setFile(f)
    setSelectedSampleId(null)
    setVideoUrl(URL.createObjectURL(f))
    setResult(null)
  }, [videoUrl])

  const clearVideo = () => {
    if (videoUrl?.startsWith('blob:')) URL.revokeObjectURL(videoUrl)
    setFile(null)
    setSelectedSampleId(null)
    setVideoUrl(null)
    setVideoMeta(null)
    setResult(null)
  }

  const runSampleInference = useCallback(async (sampleId, targetModel, targetDecode) => {
    setLoading(true)
    try {
      const sm = models.find((m) => m.id === targetModel)
      const useDecode = sm?.decode_modes?.includes(targetDecode) ? targetDecode : (sm?.default_decode || 'greedy')
      const data = await inferSample({
        sampleId,
        model: targetModel,
        decodeMode: useDecode,
      })
      setResult(data)
    } catch (e) {
      setToast({
        type: 'error',
        message: e.message || 'Örnek tahmin başarısız',
      })
    } finally {
      setLoading(false)
    }
  }, [models, setToast])

  // Seçili bir örnek varken model veya decodeMode değişirse otomatik güncelle
  useEffect(() => {
    if (selectedSampleId && !file) {
      runSampleInference(selectedSampleId, model, decodeMode)
    }
  }, [selectedSampleId, model, decodeMode, file, runSampleInference])

  const handleSelectSample = (sampleId) => {
    if (videoUrl?.startsWith('blob:')) URL.revokeObjectURL(videoUrl)
    setFile(null)
    setVideoMeta(null)
    setVideoUrl(getSampleVideoUrl(sampleId))
    setSelectedSampleId(sampleId)
  }

  const handleInfer = async () => {
    if (selectedSampleId && !file) {
      await runSampleInference(selectedSampleId, model, decodeMode)
      return
    }
    if (!file) {
      setToast({ type: 'error', message: 'Önce bir video yükleyin veya hazır tez örneklerinden birini seçin' })
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
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
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

        {samples.length > 0 && (
          <section className="card">
            <div className="card-header">
              <h2 className="card-title">Hazır Türkçe Tez Örnekleri (Klip Yüklemeden Dene)</h2>
              <span style={{ fontSize: '0.72rem', color: 'var(--accent-soft)' }}>
                4 Benchmark Klip
              </span>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <div className="sample-grid">
                {samples.map((s) => {
                  const active = selectedSampleId === s.id
                  return (
                    <button
                      key={s.id}
                      type="button"
                      className={`sample-btn ${active ? 'active' : ''}`}
                      onClick={() => handleSelectSample(s.id)}
                    >
                      <div className="sample-btn-top">
                        <strong>{s.title}</strong>
                        <span className="sample-duration">{s.duration_sec ?? s.duration_s} sn</span>
                      </div>
                      <span className="sample-speaker">{s.speaker}</span>
                      <code className="sample-gt">"{s.ground_truth}"</code>
                    </button>
                  )
                })}
              </div>
            </div>
          </section>
        )}
      </div>

      <div className="right-panel">
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Tahmin</h2>
            {selectedSampleId && (
              <span style={{ fontSize: '0.72rem', color: 'var(--success-soft)' }}>
                Örnek Klip Seçili ({selectedSampleId})
              </span>
            )}
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
              {selectedModel?.description && (
                <span className="field-hint">{selectedModel.description}</span>
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
                <span className="field-hint">greedy / beam_lm / attn çıktısını kıyaslamak için değiştir</span>
              </div>
            )}

            {needsAudio && !selectedSampleId && (
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
              disabled={loading || (!file && !selectedSampleId) || availableModels.length === 0}
            >
              {loading ? (<><span className="spinner" /> İşleniyor…</>) : (<>{Icon.play} Tahmin Yap</>)}
            </button>

            {loading && <div className="progress-indeterminate" />}
          </div>
        </section>

        <ResultCard
          rawText={result?.success ? result.text : null}
          groundTruth={result?.processing?.ground_truth || null}
          visemeNote={result?.processing?.viseme_note || null}
          comparison={result?.processing?.comparison || null}
          rightMeta={result?.success ? formatMs(result.processing?.total_ms) : null}
          emptyHint={
            result && !result.success
              ? 'Tahmin başarısız.'
              : 'Video yükleyip veya soldaki hazır tez örneklerinden birini seçip tahmin başlat.'
          }
        />

        {result?.processing && <MetricsGrid processing={result.processing} />}
      </div>
    </div>
  )
}
