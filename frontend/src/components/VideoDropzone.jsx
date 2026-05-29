import { useRef, useState } from 'react'
import { Icon } from '../lib/icons.jsx'
import { formatBytes } from '../lib/formatters.js'

/**
 * Drag & drop video alanı — kısa ve uzun mode için paylaşımlı.
 *
 * Props:
 *   file, onFile(file), onClear()
 *   videoUrl (oynatılacak blob url — eğer ayrı state tutulduysa)
 *   maxSizeMB (default 50)
 *   placeholder (dropzone boş iken gösterilecek metin)
 *   videoMeta (opsiyonel — dış state, videoRef'ten duration/width/height toplanmışsa)
 *   videoRef (dış ref — parent kontrol edebilsin)
 */
export function VideoDropzone({
  file,
  videoUrl,
  onFile,
  onClear,
  maxSizeMB = 50,
  placeholder = 'Video yükle veya buraya sürükle',
  hint,
  videoMeta,
  videoRef,
  disabled = false,
}) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState(null)

  const accept = (f) => {
    if (!f) return
    if (!f.type.startsWith('video/')) {
      setError('Lütfen bir video dosyası seçin')
      return
    }
    if (f.size > maxSizeMB * 1024 * 1024) {
      setError(`Video çok büyük (max ${maxSizeMB} MB)`)
      return
    }
    setError(null)
    onFile?.(f)
  }

  const handleChange = (e) => accept(e.target.files?.[0])
  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (disabled) return
    accept(e.dataTransfer.files?.[0])
  }

  return (
    <section className="card video-card">
      <div className="card-header">
        <h2 className="card-title">Video</h2>
        {file && !disabled && (
          <button
            type="button"
            className="btn btn-ghost btn-icon"
            onClick={onClear}
            aria-label="Temizle"
          >
            {Icon.trash}
          </button>
        )}
      </div>

      <div
        className={`dropzone ${videoUrl ? 'has-video' : ''} ${dragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !videoUrl && !disabled && inputRef.current?.click()}
        role={videoUrl ? undefined : 'button'}
        tabIndex={videoUrl ? undefined : 0}
      >
        {videoUrl ? (
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            playsInline
            className="dropzone-video"
          />
        ) : (
          <div className="dropzone-placeholder">
            {Icon.upload}
            <strong>{placeholder}</strong>
            <span>{hint || `MP4, WebM (max ${maxSizeMB} MB)`}</span>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        onChange={handleChange}
        className="file-input"
        disabled={disabled}
      />

      {!disabled && (
        <div className="video-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
          >
            {Icon.upload} {videoUrl ? 'Değiştir' : 'Dosya Seç'}
          </button>
        </div>
      )}

      {file && (
        <div className="video-meta">
          <span className="video-meta-item">
            <strong>{file.name}</strong>
          </span>
          <span className="video-meta-item">
            Boyut: <strong>{formatBytes(file.size)}</strong>
          </span>
          {videoMeta && (
            <>
              <span className="video-meta-item">
                Süre: <strong>{videoMeta.duration?.toFixed(1)}s</strong>
              </span>
              <span className="video-meta-item">
                Çözünürlük: <strong>{videoMeta.width}×{videoMeta.height}</strong>
              </span>
            </>
          )}
        </div>
      )}

      {error && (
        <div className="video-meta">
          <span className="video-meta-item" style={{ color: 'var(--error-soft)' }}>
            {error}
          </span>
        </div>
      )}
    </section>
  )
}
