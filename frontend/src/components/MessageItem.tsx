import { useMemo, useState } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { ChevronDown, FileText, Image } from 'lucide-react'
import { References } from './References'
import type { SessionMessage } from '../types'

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

interface MessageItemProps {
  message: SessionMessage
  isCurrentLoading?: boolean
  thinkingLabel?: string
}

export function MessageItem({ message, isCurrentLoading, thinkingLabel }: MessageItemProps) {
  const [reasoningOpen, setReasoningOpen] = useState(false)
  const isUser = message.role === 'user'

  const html = useMemo(() => {
    if (isUser) return ''
    return renderMarkdown(message.content)
  }, [message.content, isUser])

  if (isUser) {
    return (
      <div className="message message-user">
        <div className="message-user-bubble">
          {message.content}
          {message.fileNames && message.fileNames.length > 0 && (
            <div className="file-attachments">
              {message.fileNames.map((name, index) => (
                <FileAttachment key={`${name}-${index}`} name={name} />
              ))}
            </div>
          )}
        </div>
      </div>
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
