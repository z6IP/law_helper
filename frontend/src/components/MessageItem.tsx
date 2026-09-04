import { useEffect, useMemo, useRef, useState } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { ChevronDown, FileText, Image } from 'lucide-react'
import { References } from './References'
import type { Attachment, SessionMessage } from '../types'

function renderMarkdown(src: string): string {
  const raw = marked.parse(src, { async: false }) as string
  return DOMPurify.sanitize(raw)
}

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'])

function getFileExt(name: string): string {
  const idx = name.lastIndexOf('.')
  return idx === -1 ? '' : name.slice(idx).toLowerCase()
}

function FileAttachment({ name }: { name: string }) {
  const ext = getFileExt(name)
  const isImage = IMAGE_EXTS.has(ext)
  return (
    <div className={`file-attachment-box ${isImage ? 'image' : 'file'}`}>
      {isImage ? <Image size={18} /> : <FileText size={18} />}
      <span className="file-attachment-name" title={name}>
        {name}
      </span>
    </div>
  )
}

function AttachmentItem({ attachment, onPreview }: { attachment: Attachment; onPreview?: (url: string, rect: DOMRect) => void }) {
  const [error, setError] = useState(false)
  const [loaded, setLoaded] = useState(false)

  // 切换会话复用组件时，重置加载状态，避免上一会话的失败/加载状态影响当前图片
  useEffect(() => {
    setError(false)
    setLoaded(false)
  }, [attachment.url])

  if (attachment.type === 'image' && !error) {
    return (
      <div className="image-attachment">
        <img
          src={attachment.url}
          alt={attachment.name}
          crossOrigin="use-credentials"
          onLoad={() => setLoaded(true)}
          onError={() => setError(true)}
          onClick={(e) => {
            onPreview?.(attachment.url, (e.currentTarget as HTMLElement).getBoundingClientRect())
          }}
          style={{ cursor: 'pointer', opacity: loaded ? 1 : 0 }}
        />
      </div>
    )
  }
  return <FileAttachment name={attachment.name} />
}

interface ImagePreviewProps {
  url: string
  sourceRect: DOMRect
  onClose: () => void
}

function ImagePreview({ url, sourceRect, onClose }: ImagePreviewProps) {
  const overlayRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [closing, setClosing] = useState(false)

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const target = computeTargetRect(sourceRect)
    const start = {
      x: sourceRect.left - target.left,
      y: sourceRect.top - target.top,
      scaleX: sourceRect.width / target.width,
      scaleY: sourceRect.height / target.height,
    }
    img.style.left = `${target.left}px`
    img.style.top = `${target.top}px`
    img.style.width = `${target.width}px`
    img.style.height = `${target.height}px`
    img.style.transform = `translate3d(${start.x}px, ${start.y}px, 0) scale3d(${start.scaleX}, ${start.scaleY}, 1)`

    const overlay = overlayRef.current
    if (overlay) {
      overlay.style.opacity = '0'
    }

    requestAnimationFrame(() => {
      img.style.transition = 'transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1)'
      img.style.transform = 'translate3d(0, 0, 0) scale3d(1, 1, 1)'
      if (overlay) {
        overlay.style.transition = 'opacity 0.35s ease'
        overlay.style.opacity = '1'
      }
    })
  }, [sourceRect])

  const handleClose = () => {
    if (closing) return
    setClosing(true)
    const img = imgRef.current
    if (img) {
      const target = computeTargetRect(sourceRect)
      const end = {
        x: sourceRect.left - target.left,
        y: sourceRect.top - target.top,
        scaleX: sourceRect.width / target.width,
        scaleY: sourceRect.height / target.height,
      }
      img.style.transition = 'transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1)'
      img.style.transform = `translate3d(${end.x}px, ${end.y}px, 0) scale3d(${end.scaleX}, ${end.scaleY}, 1)`
    }
    const overlay = overlayRef.current
    if (overlay) {
      overlay.style.transition = 'opacity 0.15s ease'
      overlay.style.opacity = '0'
    }
    setTimeout(() => {
      onClose()
    }, 350)
  }

  return (
    <div
      ref={overlayRef}
      className="image-preview-overlay"
      onClick={(e) => {
        e.stopPropagation()
        handleClose()
      }}
      role="button"
      aria-label="关闭预览"
      tabIndex={-1}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          e.stopPropagation()
          handleClose()
        }
      }}
    >
      <img ref={imgRef} src={url} alt="预览" crossOrigin="use-credentials" className="image-preview-floating" onClick={(e) => { e.stopPropagation(); handleClose() }} />
    </div>
  )
}

function computeTargetRect(sourceRect: DOMRect) {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const maxWidth = vw * 0.75
  const maxHeight = vh * 0.75
  const scale = Math.min(maxWidth / sourceRect.width, maxHeight / sourceRect.height)
  const width = sourceRect.width * scale
  const height = sourceRect.height * scale
  return {
    left: (vw - width) / 2,
    top: (vh - height) / 2,
    width,
    height,
  }
}

interface MessageItemProps {
  message: SessionMessage
  isCurrentLoading?: boolean
  thinkingLabel?: string
}

export function MessageItem({ message, isCurrentLoading, thinkingLabel }: MessageItemProps) {
  const [reasoningOpen, setReasoningOpen] = useState(false)
  interface PreviewInfo {
    url: string
    rect: DOMRect
  }
  
  const [preview, setPreview] = useState<PreviewInfo | null>(null)
  const isUser = message.role === 'user'

  useEffect(() => {
    setPreview(null)
  }, [message])

  const html = useMemo(() => {
    if (isUser) return ''
    return renderMarkdown(message.content)
  }, [message.content, isUser])

  if (isUser) {
    const hasAttachments = message.attachments && message.attachments.length > 0
    const hasFileNames = message.fileNames && message.fileNames.length > 0
    const hasText = message.content.trim().length > 0
    return (
      <>
        {preview && <ImagePreview url={preview.url} sourceRect={preview.rect} onClose={() => setPreview(null)} />}
        <div className="message message-user">
          <div className="message-user-content">
            {(hasAttachments || hasFileNames) && (
              <div className="message-user-attachments">
                {hasAttachments
                  ? message.attachments!.map((attachment, index) => (
                      <AttachmentItem key={`${attachment.url}-${index}`} attachment={attachment} onPreview={(url, rect) => setPreview({ url, rect })} />
                    ))
                  : message.fileNames!.map((name, index) => (
                      <FileAttachment key={`${name}-${index}`} name={name} />
                    ))}
              </div>
            )}
            {hasText && <div className="message-user-bubble">{message.content}</div>}
          </div>
        </div>
      </>
    )
  }

  const showReasoning = message.reasoning !== null || isCurrentLoading

  return (
    <div className="message message-assistant">
      <div className="message-assistant-content">
        {showReasoning && (
          <div className="reasoning-block">
            <button
              type="button"
              className="reasoning-toggle"
              onClick={() => setReasoningOpen((v) => !v)}
            >
              <ChevronDown size={14} className={reasoningOpen ? 'open' : ''} />
              {isCurrentLoading ? (
                <>
                  <span className="spinner" />
                  <span>{thinkingLabel || '思考中'}</span>
                </>
              ) : (
                <span>思考完成</span>
              )}
            </button>
            {reasoningOpen && (
              <div
                className="reasoning-body"
                dangerouslySetInnerHTML={{
                  __html: renderMarkdown(message.reasoning || ''),
                }}
              />
            )}
          </div>
        )}
        <div
          className="markdown-body"
          dangerouslySetInnerHTML={{ __html: html }}
        />
        {!isCurrentLoading && <References references={message.references || []} />}
      </div>
    </div>
  )
}
