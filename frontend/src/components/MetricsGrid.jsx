import { formatMs } from '../lib/formatters.js'

/**
 * İnference metrikleri — kısa/uzun video için.
 *
 * Props:
 *   processing: {model_id, frames, roi_shape, roi_ms, decode_ms, total_ms, ...}
 *   extras: [{label, value}] — ek metrikler (ör. "Segment sayısı")
 */
export function MetricsGrid({ processing, extras = [] }) {
  if (!processing) return null

  const items = [
    { label: 'Model', value: processing.model_id || '—' },
    { label: 'Frame', value: processing.frames ?? '—' },
    {
      label: 'ROI',
      value: processing.roi_shape
        ? `${processing.roi_shape[1]}×${processing.roi_shape[2]}`
        : '—',
    },
    { label: 'ROI süresi', value: formatMs(processing.roi_ms) },
    { label: 'Decode', value: formatMs(processing.decode_ms) },
    { label: 'Toplam', value: formatMs(processing.total_ms) },
    ...extras,
  ]

  return (
    <section className="card">
      <div className="card-header">
        <h2 className="card-title">İşleme detayları</h2>
      </div>
      <div className="card-body">
        <div className="metrics-grid">
          {items.map((m, i) => (
            <div className="metric" key={`${m.label}-${i}`}>
              <div className="metric-label">{m.label}</div>
              <div className="metric-value">{m.value}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
