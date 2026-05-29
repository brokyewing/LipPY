import { useEffect, useState } from 'react'
import './App.css'

import { getHealth, getModels, API_BASE } from './lib/api.js'
import { Header } from './components/Header.jsx'
import { ModeTabs } from './components/ModeTabs.jsx'
import { Toast } from './components/Toast.jsx'
import { ShortVideoMode } from './modes/ShortVideoMode.jsx'
import { LongVideoMode } from './modes/LongVideoMode.jsx'
import { LiveCaptureMode } from './modes/LiveCaptureMode.jsx'

function App() {
  const [theme, setTheme] = useState(() =>
    localStorage.getItem('lippy-theme') ||
    (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
  )
  const [mode, setMode] = useState(() => localStorage.getItem('lippy-mode') || 'short')

  const [health, setHealth] = useState({ status: 'checking' })
  const [models, setModels] = useState([])
  const [toast, setToast] = useState(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('lippy-theme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem('lippy-mode', mode)
  }, [mode])

  // Backend health + models (tüm modeler paylaşıyor)
  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const [h, m] = await Promise.all([getHealth(), getModels()])
        if (!alive) return
        setHealth(h)
        setModels(m)
      } catch (e) {
        if (!alive) return
        setHealth({ status: 'down', error: String(e) })
      }
    }
    load()
    return () => { alive = false }
  }, [])

  return (
    <div className="app">
      <Header health={health} theme={theme} setTheme={setTheme} />

      <ModeTabs mode={mode} setMode={setMode} />

      <main className="app-main">
        {mode === 'short' && (
          <ShortVideoMode models={models} setToast={setToast} />
        )}
        {mode === 'long' && (
          <LongVideoMode models={models} setToast={setToast} />
        )}
        {mode === 'live' && (
          <LiveCaptureMode models={models} setToast={setToast} />
        )}
      </main>

      <footer className="app-footer">
        LipPY — Türkçe Dudak Okuma · Video + AV-Fusion (31 fonem CTC) &nbsp;·&nbsp;
        API: <code>{API_BASE}</code>
      </footer>

      <Toast toast={toast} onDismiss={() => setToast(null)} />
    </div>
  )
}

export default App
