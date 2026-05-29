import { useState } from 'react'
import { Icon } from '../lib/icons.jsx'

/**
 * Sonuç kartı — ham CTC fonem çıktısını gösterir.
 *
 * Props:
 *   rawText: Ham CTC çıktısı (boşluksuz fonem)
 *   rightMeta: Header sağına küçük bilgi (süre vb.)
 *   emptyHint: Boş iken gösterilecek metin
 */
export function ResultCard({ rawText, rightMeta, emptyHint = 'Video yükleyip tahmin başlat.' }) {
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

      <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        <div>
          <div className="result-label">Ham çıktı (fonem greedy)</div>
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
      </div>
    </section>
  )
}
