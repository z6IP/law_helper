import { useEffect, useRef } from 'react'
import { MessageItem } from './MessageItem'
import { Welcome } from './Welcome'
import type { SessionMessage } from '../types'

interface MessageListProps {
  messages: SessionMessage[]
  loading?: boolean
  reasoningLoading?: boolean
  restoring?: boolean
  thinkingLabel?: string
  deepThinking?: boolean
}

export function MessageList({ messages, loading, reasoningLoading = false, restoring = false, thinkingLabel, deepThinking }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  // 记录用户是否在底部附近：用户主动向上滚动后不再强制跳转到底部
  const atBottomRef = useRef(true)
  const lastScrollTopRef = useRef(0)

  // 绑定滚动容器的 scroll 事件，实时更新 atBottomRef
  useEffect(() => {
    const el = bottomRef.current
    if (!el) return
    const container = el.closest('.messages-area') as HTMLElement | null
    if (!container) return

    const onScroll = () => {
      const distanceFromBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight
      const userIsScrollingUp = container.scrollTop < lastScrollTopRef.current - 2
      const nearBottom = distanceFromBottom <= 24

      if (userIsScrollingUp && !nearBottom) {
        atBottomRef.current = false
      } else if (nearBottom) {
        atBottomRef.current = true
      }

      lastScrollTopRef.current = container.scrollTop
    }
    container.addEventListener('scroll', onScroll, { passive: true })
    lastScrollTopRef.current = container.scrollTop
    onScroll() // 初始化一次

    return () => container.removeEventListener('scroll', onScroll)
  }, [messages.length])

  // 新消息/流式输出到达时，仅当用户停留在底部附近才自动滚动；
  // 用户已向上滚动查看历史时，绝不强制跳转，不阻碍用户操作
  useEffect(() => {
    if (document.body.getAttribute('data-preview-open') === 'true') return
    if (atBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'auto' })
    }
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
      {messages.map((msg, idx) => {
        const isLast = idx === messages.length - 1
        return (
          <MessageItem
            key={`msg-${idx}`}
            message={msg}
            isCurrentLoading={loading && isLast && msg.role === 'assistant'}
            reasoningLoading={reasoningLoading && isLast && msg.role === 'assistant'}
            thinkingLabel={isLast ? thinkingLabel : undefined}
            deepThinking={deepThinking}
          />
        )
      })}
      <div ref={bottomRef} />
    </div>
  )
}
