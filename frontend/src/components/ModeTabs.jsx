import { Icon } from '../lib/icons.jsx'

/**
 * 3-mode sekme seçici — Kısa / Uzun / Canlı.
 *
 * Props:
 *   mode: 'short' | 'long' | 'live'
 *   setMode(m)
 */
export function ModeTabs({ mode, setMode }) {
  const tabs = [
    { id: 'short', label: 'Kısa Video', desc: '5-60 sn klip, anlık sonuç', icon: Icon.videoFile },
    { id: 'long', label: 'Uzun Video', desc: '20 dk+ video, zaman damgalı altyazı', icon: Icon.film },
    { id: 'live', label: 'Canlı Çekim', desc: 'Webcam + anlık altyazı', icon: Icon.camera },
  ]

  return (
    <nav className="mode-tabs" aria-label="Tahmin modu">
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`mode-tab ${mode === t.id ? 'active' : ''}`}
          onClick={() => setMode(t.id)}
          aria-pressed={mode === t.id}
        >
          <span className="mode-tab-icon">{t.icon}</span>
          <span className="mode-tab-label">
            <strong>{t.label}</strong>
            <small>{t.desc}</small>
          </span>
        </button>
      ))}
    </nav>
  )
}
