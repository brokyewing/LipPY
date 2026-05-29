/** Byte / süre / zaman format yardımcıları. */

export function formatBytes(bytes) {
  if (!bytes) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let v = bytes
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`
}

export function formatMs(ms) {
  if (ms == null) return '—'
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms} ms`
}

export function formatDuration(seconds) {
  if (seconds == null) return '—'
  const s = Math.floor(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  return `${m}:${String(sec).padStart(2, '0')}`
}

/** Saniye → "00:01:23.456" (VTT formatı) */
export function formatVttTime(seconds) {
  if (seconds == null) return '00:00:00.000'
  const ms = Math.floor((seconds - Math.floor(seconds)) * 1000)
  const s = Math.floor(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(ms).padStart(3, '0')}`
}

/** Segment listesini WebVTT string'ine dönüştür */
export function segmentsToVtt(segments) {
  let vtt = 'WEBVTT\n\n'
  segments.forEach((s, i) => {
    vtt += `${i + 1}\n`
    vtt += `${formatVttTime(s.start)} --> ${formatVttTime(s.end)}\n`
    vtt += `${s.text}\n\n`
  })
  return vtt
}

/** SRT formatı (`.srt` dosya) */
export function segmentsToSrt(segments) {
  let srt = ''
  segments.forEach((s, i) => {
    const tStart = formatVttTime(s.start).replace('.', ',')
    const tEnd = formatVttTime(s.end).replace('.', ',')
    srt += `${i + 1}\n${tStart} --> ${tEnd}\n${s.text}\n\n`
  })
  return srt
}

/** Plain text (her segment yeni satır) */
export function segmentsToTxt(segments) {
  return segments.map((s) => s.text).join('\n')
}

/** Download helper */
export function downloadBlob(content, filename, type = 'text/plain') {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
