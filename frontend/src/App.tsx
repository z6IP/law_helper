import { useEffect, useRef, useState, useCallback } from 'react'
import { PanelLeftOpen } from 'lucide-react'
import { Sidebar } from './components/Sidebar'
import { MessageList } from './components/MessageList'
import { ChatInput } from './components/ChatInput'
import { ThemeToggle } from './components/ThemeToggle'
import { useSessions } from './hooks/useSessions'
import * as api from './api'
import type { Reference, SessionMessage } from './types'

const SIDEBAR_KEY = 'sidebarVisible'

function localISOString(): string {
  const d = new Date()
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 19)
}

function App() {
  const {
    listedSessions,
    currentSession,
    currentId,
    loaded,
    switchSession,
    createSession,
    removeSession,
    appendMessages,
    updateLastMessage,
    updateSessionTitle,
  } = useSessions()

  const [loading, setLoading] = useState(false)
  const [thinkingLabel, setThinkingLabel] = useState('思考中')
  const [sidebarVisible, setSidebarVisible] = useState(() => {
    const saved = localStorage.getItem(SIDEBAR_KEY)
    return saved ? saved === 'true' : false
  })
  const [isDark, setIsDark] = useState(false)
  const [animated, setAnimated] = useState(false)
  const [mounted, setMounted] = useState(false)
  const abortRef = useRef<(() => void) | null>(null)

  const applyTheme = useCallback((dark: boolean, withTransition: boolean) => {
    if (withTransition) {
      document.documentElement.classList.add('theme-transition')
      setTimeout(() => {
        document.documentElement.classList.remove('theme-transition')
      }, 260)
    }
    setIsDark(dark)
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
  }, [])

  // 初始化主题：index.html 已设置 data-theme，这里仅同步按钮状态
  useEffect(() => {
    const saved = localStorage.getItem('theme')
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    const initialDark = saved ? saved === 'dark' : prefersDark
    setIsDark(initialDark)
  }, [])

  // 监听系统主题变化
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = (e: MediaQueryListEvent) => {
      // 仅当用户未手动设置时跟随系统
      if (!localStorage.getItem('theme')) {
        applyTheme(e.matches, true)
      }
    }
    media.addEventListener('change', handler)
    return () => media.removeEventListener('change', handler)
  }, [applyTheme])

  const toggleTheme = useCallback(() => {
    setIsDark((prev) => {
      const next = !prev
      applyTheme(next, true)
      localStorage.setItem('theme', next ? 'dark' : 'light')
      return next
    })
  }, [applyTheme])

  const toggleSidebar = useCallback((visible: boolean) => {
    setSidebarVisible(visible)
    localStorage.setItem(SIDEBAR_KEY, String(visible))
  }, [])

  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current()
      }
    }
  }, [])

  // 首次渲染完成后再启用输入框位移动画，避免刷新时从底部闪到中间
  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 0)
    return () => clearTimeout(t)
  }, [])

  // 页面内容淡入，避免刷新时突然出现的刺眼感
  useEffect(() => {
    setMounted(true)
  }, [])

  // 切换会话时中止正在进行的流式回答
  useEffect(() => {
    if (abortRef.current) {
      abortRef.current()
      abortRef.current = null
      setLoading(false)
    }
  }, [currentId])

  const handleSend = useCallback(
    async (text: string) => {
      if (!currentSession) return
      const userMsg: SessionMessage = { role: 'user', content: text }

      await appendMessages(currentSession.id, [userMsg])

      const assistantMsg: SessionMessage = {
        role: 'assistant',
        content: '',
        references: [],
        reasoning: null,
      }
      await appendMessages(currentSession.id, [assistantMsg])

      setLoading(true)
      setThinkingLabel('思考中')

      let answerText = ''
      let reasoningText = ''
      let references: Reference[] = []
      let hasReasoning = false

      const finalize = async () => {
        setLoading(false)
        const finalTitle =
          currentSession.title === '新对话' ? text.slice(0, 18) || '新对话' : currentSession.title
        updateSessionTitle(currentSession.id, finalTitle)
        await api.saveSession({
          ...currentSession,
          messages: [
            ...currentSession.messages,
            userMsg,
            {
              role: 'assistant',
              content: answerText,
              references,
              reasoning: hasReasoning ? reasoningText : null,
            },
          ],
          title: finalTitle,
          updated_at: localISOString(),
        })
      }

      abortRef.current = api.streamChat(
        text,
        currentSession.messages,
        (event) => {
          const data = event.data as Record<string, unknown>
          switch (event.type) {
            case 'references':
              references = (data.references as Reference[]) || []
              updateLastMessage(currentSession.id, (msg) => ({
                ...msg,
                references,
              }))
              break
            case 'progress':
              setThinkingLabel((data.content as string) || '思考中')
              break
            case 'reasoning':
              hasReasoning = true
              reasoningText += (data.content as string) || ''
              updateLastMessage(currentSession.id, (msg) => ({
                ...msg,
                reasoning: reasoningText,
              }))
              break
            case 'delta':
              answerText += (data.content as string) || ''
              updateLastMessage(currentSession.id, (msg) => ({
                ...msg,
                content: answerText,
              }))
              break
          }
        },
        finalize,
        (err) => {
          answerText += `\n\n后端调用失败：${err.message}`
          updateLastMessage(currentSession.id, (msg) => ({
            ...msg,
            content: answerText,
          }))
          finalize()
        },
      )
    },
    [currentSession, appendMessages, updateLastMessage, updateSessionTitle],
  )

  if (!currentSession) {
    return null
  }

  const hashId = window.location.hash.replace(/^#/, '')
  const isRestoring = !loaded && Boolean(hashId)
  const hasMessages = currentSession.messages.length > 0 || isRestoring

  return (
    <div className={`app ${mounted ? 'mounted' : ''}`}>
      <Sidebar
        sessions={listedSessions}
        currentId={currentId}
        visible={sidebarVisible}
        loaded={loaded}
        onNew={() => {
          createSession()
        }}
        onClose={() => toggleSidebar(false)}
        onSelect={(id) => {
          switchSession(id)
        }}
        onDelete={removeSession}
      />
      <main className="main">
        {!sidebarVisible && (
          <button
            type="button"
            className="sidebar-open-btn"
            onClick={() => toggleSidebar(true)}
            aria-label="打开侧边栏"
          >
            <PanelLeftOpen size={20} />
          </button>
        )}
        <ThemeToggle isDark={isDark} onToggle={toggleTheme} />
        <div className={`chat-layout ${hasMessages ? 'with-messages' : 'empty'} ${animated ? 'animated' : ''}`}>
          <div className="messages-area">
            <MessageList
              messages={currentSession.messages}
              loading={loading}
              restoring={isRestoring}
              thinkingLabel={thinkingLabel}
            />
          </div>
          <div className="input-area">
            <ChatInput onSend={handleSend} disabled={loading} />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
