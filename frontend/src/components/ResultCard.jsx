import { useState } from 'react'
import { Icon } from '../lib/icons.jsx'

/**
 * Sonuç kartı — CTC/Attention fonem çıktısını, Ground Truth referansını
 * ve Video vs AV-Fusion karşılaştırma tablosunu gösterir.
 *
 * Props:
 *   rawText: Seçili model ve decode modunun çıktısı
 *   groundTruth: (Opsiyonel) Tez benchmark referans cümlesi
 *   visemeNote: (Opsiyonel) Görsel fonem (viseme) karışıklığı açıklaması
 *   comparison: (Opsiyonel) { video_greedy, video_beam_lm, av_fusion }
 *   rightMeta: Header sağına küçük bilgi (süre vb.)
 *   emptyHint: Boş iken gösterilecek metin
 */
export function ResultCard({
  rawText,
  groundTruth = null,
  visemeNote = null,
  comparison = null,
  rightMeta,
  emptyHint = 'Video yükleyip veya soldaki hazır tez örneklerinden birini seçip tahmin başlat.',
}) {
  const [copied, setCopied] = useState(null)

  const copy = async (key, text) => {
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      setCopied(key)
      setTimeout(() => setCopied(null), 1400)
    } catch {
      /* ignore */
    }
  }

  return (
    <section className="card">
      <div className="card-header">
        <h2 className="card-title">Sonuç</h2>
        {rightMeta && (
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {rightMeta}
          </span>
        )}
      </div>

      <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
        <div>
          <div className="result-label">Seçili Model Çıktısı</div>
          <div className="result-box primary">
            {rawText ? (
              <>
                {rawText}
                <button
                  type="button"
                  className={`btn btn-ghost btn-icon result-copy ${copied === 'raw' ? 'copied' : ''}`}
                  onClick={() => copy('raw', rawText)}
                  aria-label="Kopyala"
                >
                  {copied === 'raw' ? Icon.check : Icon.copy}
                </button>
              </>
            ) : (
              <span className="result-empty">{emptyHint}</span>
            )}
          </div>
        </div>

        {groundTruth && (
          <div>
            <div className="result-label">Referans Metin (Ground Truth)</div>
            <div className="result-box" style={{ minHeight: 'auto', padding: '0.7rem 0.9rem', fontSize: '0.88rem' }}>
              {groundTruth}
            </div>
          </div>
        )}

        {visemeNote && (
          <div className="viseme-note">
            <strong>Viseme Analizi:</strong> {visemeNote}
          </div>
        )}

        {comparison && (
          <div className="comparison-table">
            <div className="result-label">Model & Decode Kıyaslaması (Tez Benchmark)</div>
            <div className="comparison-row">
              <span className="comparison-tag">Video · Greedy (%49.1 PER)</span>
              <code>{comparison.video?.greedy || comparison.video_greedy}</code>
            </div>
            <div className="comparison-row">
              <span className="comparison-tag">Video · Beam + KenLM (%78.7 WER)</span>
              <code>{comparison.video?.beam_lm || comparison.video_beam_lm}</code>
            </div>
            <div className="comparison-row highlight">
              <span className="comparison-tag">AV-Fusion · Dudak + Ses (%24.7 PER)</span>
              <code>{comparison['av-fusion']?.attn || comparison['av-fusion']?.beam_lm || comparison.av_fusion}</code>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
