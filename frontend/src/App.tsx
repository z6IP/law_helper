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
// 输入框最大宽度 800px + 侧边栏宽度 260px，低于此宽度主内容会被挤压
const SIDEBAR_AUTO_THRESHOLD = 1060

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
  // 初始主题与 index.html 中内联脚本设置保持一致，避免首屏闪烁
  const [isDark, setIsDark] = useState(() => {
    return document.documentElement.getAttribute('data-theme') === 'dark'
  })
  const [animated, setAnimated] = useState(false)
  const [mounted, setMounted] = useState(false)
  const [openBtnVisible, setOpenBtnVisible] = useState(!sidebarVisible)
  const abortRef = useRef<(() => void) | null>(null)
  // 标记侧边栏是否因窗口变窄被系统自动收起，用于宽度恢复后自动展开
  const autoCollapsedRef = useRef(false)

  const applyTheme = useCallback((dark: boolean, withTransition: boolean) => {
    const apply = () => {
      document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
      // 同步覆盖 index.html 内联脚本注入的 body 背景 !important 样式
      // 否则切换后 body 背景会被锁定为初始主题色，视觉上看不到切换
      document.body.style.setProperty('background-color', dark ? '#1a1a1a' : '#ffffff', 'important')
      setIsDark(dark)
    }

    if (!withTransition) {
      apply()
      return
    }

    // 优先使用 View Transitions API + 圆形扩散（从主题按钮位置开始）
    if (document.startViewTransition) {
      // 标记切换方向：dark=true（亮→暗）走收缩动画，dark=false（暗→亮）走扩散动画
      if (dark) {
        document.documentElement.classList.add('theme-collapsing')
      } else {
        document.documentElement.classList.remove('theme-collapsing')
      }
      const btn = document.querySelector('.theme-toggle') as HTMLElement | null
      if (btn) {
        const rect = btn.getBoundingClientRect()
        const x = rect.left + rect.width / 2
        const y = rect.top + rect.height / 2
        const endRadius = Math.hypot(
          Math.max(x, window.innerWidth - x),
          Math.max(y, window.innerHeight - y)
        )
        document.documentElement.style.setProperty('--vt-x', `${x}px`)
        document.documentElement.style.setProperty('--vt-y', `${y}px`)
        document.documentElement.style.setProperty('--vt-r', `${endRadius}px`)
      }
      document.startViewTransition(apply)
      return
    }

    // 降级：全局元素过渡（覆盖所有使用主题变量的元素）
    document.documentElement.classList.add('theme-transition')
    apply()
    setTimeout(() => {
      document.documentElement.classList.remove('theme-transition')
    }, 260)
  }, [])

  // 初始化主题：index.html 已设置 data-theme，这里仅同步按钮状态并兜底读取 localStorage
  useEffect(() => {
    const saved = localStorage.getItem('theme')
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    const initialDark = saved ? saved === 'dark' : prefersDark
    setIsDark(initialDark)
    document.documentElement.setAttribute('data-theme', initialDark ? 'dark' : 'light')
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
    // 从 DOM 读取当前主题，避免在 setIsDark updater 内嵌套调用 applyTheme（内含 setState）
    const current = document.documentElement.getAttribute('data-theme') === 'dark'
    const next = !current
    applyTheme(next, true)
    localStorage.setItem('theme', next ? 'dark' : 'light')
  }, [applyTheme])

  const toggleSidebar = useCallback((visible: boolean) => {
    setSidebarVisible(visible)
    localStorage.setItem(SIDEBAR_KEY, String(visible))
  }, [])

  // 用户手动打开/关闭侧边栏时，取消系统自动收起状态
  const handleToggleSidebar = useCallback((visible: boolean) => {
    autoCollapsedRef.current = false
    toggleSidebar(visible)
  }, [toggleSidebar])

  // 侧边栏关闭动画完成（250ms）后再显示打开按钮，避免按钮在移动过程中提前出现
  useEffect(() => {
    if (sidebarVisible) {
      setOpenBtnVisible(false)
    } else {
      const timer = setTimeout(() => setOpenBtnVisible(true), 250)
      return () => clearTimeout(timer)
    }
  }, [sidebarVisible])

  // 窗口宽度低于阈值时自动收起侧边栏，宽度恢复后自动展开（仅处理系统自动收起的状态）
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth
      if (width <= SIDEBAR_AUTO_THRESHOLD && sidebarVisible && !autoCollapsedRef.current) {
        autoCollapsedRef.current = true
        toggleSidebar(false)
      } else if (width > SIDEBAR_AUTO_THRESHOLD && !sidebarVisible && autoCollapsedRef.current) {
        autoCollapsedRef.current = false
        toggleSidebar(true)
      }
    }
    window.addEventListener('resize', handleResize)
    handleResize()
    return () => window.removeEventListener('resize', handleResize)
  }, [sidebarVisible, toggleSidebar])

  // resize 时取消输入框位移动画，避免动画期间窗口尺寸变化导致闪屏
  useEffect(() => {
    const handleResize = () => setAnimated(false)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current()
      }
    }
  }, [])

  // 页面内容淡入，避免刷新时突然出现的刺眼感
  useEffect(() => {
    setMounted(true)
  }, [])

  // 切换会话时中止正在进行的流式回答，并关闭输入框位移动画
  //（从新建对话切到历史会话时，输入框应直接出现在底部，不要平滑滑过文字）
  useEffect(() => {
    if (abortRef.current) {
      abortRef.current()
      abortRef.current = null
      setLoading(false)
    }
    setAnimated(false)
  }, [currentId])

  const handleSend = useCallback(
    async (text: string) => {
      if (!currentSession) return
      const userMsg: SessionMessage = { role: 'user', content: text }

      // 当前会话为空时，启用输入框平滑下移动画；动画播放一次后关闭，
      // 避免 resize 时 transform 变化触发过渡导致输入框延迟移动
      if (currentSession.messages.length === 0) {
        setAnimated(true)
        setTimeout(() => setAnimated(false), 500)
      }

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
        const isFirstMessage = currentSession.title === '新对话'
        const finalTitle = isFirstMessage ? text.slice(0, 18) || '新对话' : currentSession.title
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
        // 首次发送消息后，会话从“新对话”变为历史会话，同步 URL hash 以便刷新可恢复
        if (isFirstMessage) {
          switchSession(currentSession.id)
        }
      }

      abortRef.current = api.streamChat(
        text,
        currentSession.messages,
        currentSession.id,
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
        onClose={() => handleToggleSidebar(false)}
        onSelect={(id) => {
          switchSession(id)
        }}
        onDelete={removeSession}
      />
      <main className="main">
        {openBtnVisible && (
          <button
            type="button"
            className="sidebar-open-btn"
            onClick={() => handleToggleSidebar(true)}
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
