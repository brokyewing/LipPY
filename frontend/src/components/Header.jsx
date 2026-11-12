import { Icon } from '../lib/icons.jsx'

export function Header({ health, theme, setTheme }) {
  const statusOk = health?.status === 'ok'
  const statusDown = health?.status === 'down'
  const modeTag = health?.mode === 'checkpoint'
    ? ' · Aktif Model (PyTorch)'
    : health?.mode === 'demo'
    ? ' · Tez Benchmark'
    : ''
  const statusLabel = statusOk
    ? `${health.device === 'cuda' ? 'GPU' : 'CPU'}${modeTag}`
    : statusDown
    ? 'Offline'
    : '...'

  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="brand">
          <div className="brand-logo">Lp</div>
          <div>
            <div className="brand-title">LipPY</div>
            <div className="brand-subtitle">Türkçe Görsel-İşitsel Dudak Okuma — Video (%49.1 PER) + AV-Fusion (%24.7 PER)</div>
          </div>
        </div>

        <div className="header-actions">
          <span
            className={`status-chip ${statusOk ? 'ok' : statusDown ? 'err' : ''}`}
            title={statusOk
              ? `Bağlandı: ${health.device?.toUpperCase()} (${health.mode || 'aktif'})`.trim()
              : 'Backend erişilemez'}
          >
            <span className="status-dot" />
            {statusLabel}
          </span>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label="Tema değiştir"
          >
            {theme === 'dark' ? Icon.sun : Icon.moon}
            {theme === 'dark' ? 'Açık' : 'Koyu'}
          </button>
        </div>
      </div>
    </header>
  )
}
