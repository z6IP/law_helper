import { useEffect, useRef } from 'react'
import { MessageItem } from './MessageItem'
import { Welcome } from './Welcome'
import type { SessionMessage } from '../types'

interface MessageListProps {
  messages: SessionMessage[]
  loading?: boolean
  restoring?: boolean
  thinkingLabel?: string
}

export function MessageList({ messages, loading, restoring = false, thinkingLabel }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'auto' })
  }, [messages, loading])

  if (messages.length === 0) {
    if (restoring) {
      // 恢复历史会话期间不显示 Welcome 骨架，保持中间区域空白
      return <div className="message-list" />
    }
    return (
      <div className="message-list empty">
        <Welcome />
      </div>
    )
  }

  return (
    <div className="message-list">
      {messages.map((msg, idx) => (
        <MessageItem
          key={idx}
          message={msg}
          isCurrentLoading={loading && idx === messages.length - 1 && msg.role === 'assistant'}
          thinkingLabel={thinkingLabel}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
