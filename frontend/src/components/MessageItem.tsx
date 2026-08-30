import { useMemo, useState } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { ChevronDown } from 'lucide-react'
import { References } from './References'
import type { SessionMessage } from '../types'

function renderMarkdown(src: string): string {
  const raw = marked.parse(src, { async: false }) as string
  return DOMPurify.sanitize(raw)
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
        <div className="message-user-bubble">{message.content}</div>
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
