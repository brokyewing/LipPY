import { useEffect } from 'react'
import { Icon } from '../lib/icons.jsx'

export function Toast({ toast, onDismiss, autoHideMs = 3500 }) {
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => onDismiss?.(), autoHideMs)
    return () => clearTimeout(t)
  }, [toast, onDismiss, autoHideMs])

  if (!toast) return null

  return (
    <div className="toast-container">
      <div className={`toast ${toast.type || ''}`}>
        {Icon.alert}
        <span>{toast.message}</span>
      </div>
    </div>
  )
}
