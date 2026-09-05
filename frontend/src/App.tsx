import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { PanelLeftOpen } from 'lucide-react'
import { Sidebar } from './components/Sidebar'
import { MessageList } from './components/MessageList'
import { ChatInput } from './components/ChatInput'
import { ThemeToggle } from './components/ThemeToggle'
import { useSessions } from './hooks/useSessions'
import * as api from './api'
import type { Attachment, Reference } from './types'

const SIDEBAR_KEY = 'sidebarVisible'
// 输入框最大宽度 800px + 侧边栏宽度 260px，低于此宽度主内容会被挤压
const SIDEBAR_AUTO_THRESHOLD = 1060
// 拖拽上传限制
const MAX_DRAG_FILES = 3
const MAX_DRAG_FILE_SIZE = 5 * 1024 * 1024
const EMPTY_FILES: File[] = []



function App() {
  const {
    sessions,
    listedSessions,
    currentSession,
    currentId,
    loaded,
    switchSession,
    createSession,
    removeSession,
    appendMessages,
    updateLastMessage,
    updateMessageAt,
    updateSessionTitle,
  } = useSessions()

  // 后台生成任务按会话独立管理：切换会话/新建对话/刷新页面都不会中断生成
  const [runningIds, setRunningIds] = useState<Set<string>>(new Set())
  const [thinkingLabels, setThinkingLabels] = useState<Record<string, string>>({})
  const abortRefs = useRef<Map<string, () => void>>(new Map())
  const resumedRef = useRef<Set<string>>(new Set())
  const sessionsRef = useRef(sessions)
  // 材料窗口：sessionId -> { anchorIdx, text }，3 次问答（6 条消息）内有效
  const docTextsRef = useRef<Map<string, { anchorIdx: number; text: string }>>(new Map())

  const [sidebarVisible, setSidebarVisible] = useState(() => {
    const saved = localStorage.getItem(SIDEBAR_KEY)
    return saved ? saved === 'true' : false
  })
  // 主题状态以 DOM（<html data-theme>）为唯一事实来源，不在 App 中订阅，
  // 避免切换主题时整棵 React 树重渲染（2K 全屏下卡顿的主因）。
  const [animated, setAnimated] = useState(false)
  const [mounted, setMounted] = useState(false)
  const [openBtnVisible, setOpenBtnVisible] = useState(!sidebarVisible)
  // 全局拖拽上传遮罩
  const [isDragOver, setIsDragOver] = useState(false)
  const dragCounter = useRef(0)
  // 待发送的附件按会话隔离：sessionId -> File[]，切换会话不会误发
  const [pendingFilesBySession, setPendingFilesBySession] = useState<Record<string, File[]>>({})
  // 待发送的文本同样按会话隔离：sessionId -> string
  const [pendingTextBySession, setPendingTextBySession] = useState<Record<string, string>>({})

  // 当前会话的待发送附件/文本（只读派生），切回会话后仍能保留
  const pendingFiles = useMemo(
    () => (currentSession ? pendingFilesBySession[currentSession.id] || EMPTY_FILES : EMPTY_FILES),
    [currentSession, pendingFilesBySession],
  )
  const pendingText = useMemo(
    () => (currentSession ? pendingTextBySession[currentSession.id] || '' : ''),
    [currentSession, pendingTextBySession],
  )

  const setPendingFiles = useCallback(
    (sessionId: string, updater: File[] | ((prev: File[]) => File[])) => {
      setPendingFilesBySession((prev) => {
        const current = prev[sessionId] || []
        const next = typeof updater === 'function' ? (updater as (prev: File[]) => File[])(current) : updater
        return { ...prev, [sessionId]: next }
      })
    },
    [],
  )

  const clearPendingFiles = useCallback(
    (sessionId: string) => {
      setPendingFilesBySession((prev) => {
        if (!(sessionId in prev)) return prev
        const next = { ...prev }
        delete next[sessionId]
        return next
      })
    },
    [],
  )

  const setPendingText = useCallback(
    (sessionId: string, value: string) => {
      setPendingTextBySession((prev) => ({ ...prev, [sessionId]: value }))
    },
    [],
  )

  const clearPendingText = useCallback(
    (sessionId: string) => {
      setPendingTextBySession((prev) => {
        if (!(sessionId in prev)) return prev
        const next = { ...prev }
        delete next[sessionId]
        return next
      })
    },
    [],
  )

  // 标记侧边栏是否因窗口变窄被系统自动收起，用于宽度恢复后自动展开
  const autoCollapsedRef = useRef(false)

  useEffect(() => {
    sessionsRef.current = sessions
  }, [sessions])

  // 纯 DOM 主题变更：设置属性 + 同步 body 背景 + 通知 ThemeToggle 更新图标。
  // 注意：主题状态只允许存在于 DOM 与 ThemeToggle 内，切勿在此触发 App 树 setState。
  const applyThemeDOM = useCallback((dark: boolean) => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
    // 同步覆盖 index.html 内联脚本注入的 body 背景 !important 样式，
    // 颜色从 CSS 变量 --bg 读取，避免与 index.css 中的主题色重复定义。
    const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()
    document.body.style.setProperty('background-color', bg || (dark ? '#1a1a1a' : '#ffffff'), 'important')
    window.dispatchEvent(new CustomEvent('theme-change', { detail: { dark } }))
  }, [])

  // 主题动画连点锁：动画进行中忽略后续切换，避免多个全屏快照/overlay 叠加
  const themeAnimatingRef = useRef(false)

  const applyTheme = useCallback((dark: boolean, withTransition: boolean) => {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (!withTransition || reduceMotion) {
      applyThemeDOM(dark)
      return
    }

    if (themeAnimatingRef.current) return
    themeAnimatingRef.current = true
    const unlock = () => {
      themeAnimatingRef.current = false
    }

    const btn = document.querySelector('.theme-toggle') as HTMLElement | null
    let x = window.innerWidth - 40
    let y = 40
    if (btn) {
      const rect = btn.getBoundingClientRect()
      x = rect.left + rect.width / 2
      y = rect.top + rect.height / 2
    }
    const endRadius =
      Math.hypot(
        Math.max(x, window.innerWidth - x),
        Math.max(y, window.innerHeight - y)
      ) + 2

    // View Transitions API 需捕获两张全屏位图快照，开销与视口面积×devicePixelRatio 成正比。
    // 2560×1440 及以上/高分屏缩放（等效物理像素≈4K）下，捕获阶段可阻塞 50–80ms，
    // 表现为点击后明显延迟才开始动画。因此高分辨率下跳过 VT，统一走 overlay 方案。
    const physicalWidth = window.innerWidth * window.devicePixelRatio
    const supportsVT = typeof (document as Document & { startViewTransition?: unknown }).startViewTransition === 'function'
    const useVT = supportsVT && physicalWidth <= 2000

    if (useVT) {
      // 标记切换方向：dark=true（亮→暗）走收缩动画，dark=false（暗→亮）走扩散动画
      if (dark) {
        document.documentElement.classList.add('theme-collapsing')
      } else {
        document.documentElement.classList.remove('theme-collapsing')
      }
      document.documentElement.style.setProperty('--vt-x', `${x}px`)
      document.documentElement.style.setProperty('--vt-y', `${y}px`)
      document.documentElement.style.setProperty('--vt-r', `${endRadius}px`)
      // 回调内只做最轻量的 DOM 变更，快照捕获不被 React 渲染阻塞
      const transition = document.startViewTransition(() => {
        applyThemeDOM(dark)
      })
      transition.finished.then(unlock, unlock)
      return
    }

    // 通用方案（高分辨率优先 / 无 VT 浏览器降级）：用 mask 径向渐变揭露新主题页面。
    // overlay 为旧主题色全屏层，mask 挖出透明圆——圆内露出新主题真实页面，圆外保持旧主题色。
    // 圆半径 --reveal-r 由 JS requestAnimationFrame 插值驱动（不依赖 @property / CSS transition），
    // 兼容性好、不捕获位图，翻转主题在动画开始时执行（被 overlay 遮挡），重绘与动画并行。
    // - 切亮色（dark=false）：r 从 0→max，亮色页面从按钮位置扩散揭露。
    // - 切暗色（dark=true）：r 从 max→0，亮色区域向按钮位置聚拢，露出暗色页面。
    const isExpand = !dark // 亮色=扩散(r:0→max)，暗色=聚拢(r:max→0)
    const overlay = document.createElement('div')
    overlay.className = 'theme-reveal-overlay ' + (isExpand ? 'reveal-expand' : 'reveal-collapse')
    overlay.style.setProperty('--reveal-x', `${x}px`)
    overlay.style.setProperty('--reveal-y', `${y}px`)
    overlay.style.setProperty('--reveal-r', isExpand ? '0px' : `${endRadius}px`)
    // overlay 背景为旧主题色：切暗色时旧=亮(白)，切亮色时旧=暗
    overlay.style.background = dark ? '#ffffff' : '#1a1a1a'
    document.body.appendChild(overlay)

    let done = false
    const finish = () => {
      if (done) return
      done = true
      overlay.remove()
      unlock()
    }

    // 翻转到新主题（此时 overlay 圆外区域仍为旧主题色，翻转不可见）
    applyThemeDOM(dark)

    // JS 驱动的半径插值动画（ease-out，250ms），不依赖 @property
    const startR = isExpand ? 0 : endRadius
    const endR = isExpand ? endRadius : 0
    const duration = 250
    const startTime = performance.now()
    let rafId = 0
    const step = (now: number) => {
      if (done) return
      const progress = Math.min((now - startTime) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // ease-out cubic
      const r = startR + (endR - startR) * eased
      overlay.style.setProperty('--reveal-r', `${r}px`)
      if (progress < 1) {
        rafId = requestAnimationFrame(step)
      } else {
        finish()
      }
    }
    rafId = requestAnimationFrame(step)

    // 兜底：动画异常未完成时强制收尾并取消 rAF
    setTimeout(() => {
      if (!done) {
        cancelAnimationFrame(rafId)
        finish()
      }
    }, 320)
  }, [applyThemeDOM])

  // 初始化主题：index.html 已设置 data-theme，这里兜底读取 localStorage 并统一走 DOM 应用
  useEffect(() => {
    const saved = localStorage.getItem('theme')
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    const initialDark = saved ? saved === 'dark' : prefersDark
    applyThemeDOM(initialDark)
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    // 从 DOM 读取当前主题（DOM 是主题的唯一事实来源）
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

  // 页面卸载时中止所有在途流式订阅（后端后台任务不受影响，仍会继续生成并落库）
  useEffect(() => {
    const refs = abortRefs.current
    return () => {
      refs.forEach((fn) => fn())
    }
  }, [])

  // 页面内容淡入，避免刷新时突然出现的刺眼感
  useEffect(() => {
    setMounted(true)
  }, [])

  // 切换会话时不再中止生成：仅关闭输入框位移动画。
  // 后台生成按 sessionId 独立继续，写回对应会话，不受当前界面影响。
  useEffect(() => {
    setAnimated(false)
  }, [currentId])

  // 启动一次后台流式订阅，事件驱动更新会话（无论是否为当前会话）
  const runStream = useCallback(
    (sessionId: string, start: (h: api.StreamHandlers) => () => void) => {
      let answerText = ''
      let reasoningText = ''
      let references: Reference[] = []

      const finalize = () => {
        setRunningIds((prev) => {
          const next = new Set(prev)
          next.delete(sessionId)
          return next
        })
        setThinkingLabels((prev) => {
          const next = { ...prev }
          delete next[sessionId]
          return next
        })
        abortRefs.current.delete(sessionId)
        // 首条消息在回答完成（成功或失败）后，再生成总结性标题，避免发送时提前覆盖
        const session = sessionsRef.current.find((s) => s.id === sessionId)
        if (session && session.title === '新对话') {
          api
            .summarizeSession(sessionId, session.messages)
            .then((title) => updateSessionTitle(sessionId, title.slice(0, 18) || '新对话'))
            .catch(() => {})
        }
      }

      const onEvent = (event: api.StreamEvent) => {
        const data = event.data as Record<string, unknown>
        switch (event.type) {
          case 'references':
            references = (data.references as Reference[]) || []
            updateLastMessage(sessionId, (m) => ({ ...m, references }))
            break
          case 'progress':
            setThinkingLabels((prev) => ({
              ...prev,
              [sessionId]: (data.content as string) || '思考中',
            }))
            break
          case 'reasoning':
            reasoningText += (data.content as string) || ''
            updateLastMessage(sessionId, (m) => ({ ...m, reasoning: reasoningText }))
            break
          case 'delta':
            answerText += (data.content as string) || ''
            updateLastMessage(sessionId, (m) => ({ ...m, content: answerText }))
            break
        }
      }

      const onError = (err: Error) => {
        answerText += (answerText ? '\n\n' : '') + '后端调用失败：' + err.message
        updateLastMessage(sessionId, (m) => ({ ...m, content: answerText }))
        finalize()
      }

      const onDone = () => {
        finalize()
      }

      abortRefs.current.set(sessionId, start({ onEvent, onDone, onError }))
    },
    [updateLastMessage, updateSessionTitle],
  )

  const handleSend = useCallback(
    async (text: string, files?: File[]) => {
      if (!currentSession) return
      const sessionId = currentSession.id
      const isFirst = currentSession.title === '新对话'
      const hasFiles = files && files.length > 0
      // 首条消息保持"新对话"，等回答完成后再由 summarizeSession 统一生成最终标题，
      // 避免发送时立即覆盖、回答完成后再被二次覆盖的风险
      const sessionTitle = isFirst ? '新对话' : currentSession.title
      const history = currentSession.messages

      // 当前会话为空时，启用输入框平滑下移动画；动画播放一次后关闭
      if (currentSession.messages.length === 0) {
        setAnimated(true)
        setTimeout(() => setAnimated(false), 500)
      }

      // 1. 先生成本地预览 attachments，立即显示用户消息
      const localObjectUrls: string[] = []
      const localAttachments: Attachment[] = hasFiles
        ? files.map((file) => {
            const url = URL.createObjectURL(file)
            localObjectUrls.push(url)
            return {
              name: file.name,
              url,
              type: /\.(png|jpg|jpeg|webp|bmp|gif)$/i.test(file.name) ? 'image' : 'document',
            }
          })
        : []

      // 2. 显示用户消息（含本地预览）和助手占位
      await appendMessages(sessionId, [
        { role: 'user', content: text, fileNames: files?.map((f) => f.name), attachments: localAttachments },
      ])
      await appendMessages(sessionId, [
        { role: 'assistant', content: hasFiles ? '文件解析中...' : '', references: [], reasoning: null },
      ])

      // 首条消息发送即写入 URL hash：思考/回答过程中刷新，能恢复到当前生成中的会话，
      // 而不是回到空的新对话界面
      if (isFirst) {
        switchSession(sessionId)
      }

      // 发送成功后清空已确认的附件和文本
      clearPendingFiles(sessionId)
      clearPendingText(sessionId)

      // 3. 在后台异步上传文件并启动问答
      ;(async () => {
        let documentText: string | undefined
        let uploadedAttachments: Attachment[] = []
        if (hasFiles) {
          try {
            const results = await Promise.all(files.map((file) => api.uploadDocument(file)))
            documentText = results.map((r) => r.text).join('\n\n---\n\n')
            uploadedAttachments = results.map((r) => ({
              name: r.name,
              url: r.url,
              type: /\.(png|jpg|jpeg|webp|bmp|gif)$/i.test(r.url) ? 'image' : 'document',
            }))
            docTextsRef.current.set(sessionId, { anchorIdx: history.length, text: documentText })
            // 更新用户消息，把本地预览 URL 替换为持久化 URL，并释放本地 blob URL
            updateMessageAt(sessionId, history.length, (m) => {
              localObjectUrls.forEach((url) => URL.revokeObjectURL(url))
              return { ...m, attachments: uploadedAttachments }
            })
          } catch (err) {
            const message = err instanceof Error ? err.message : '文件上传失败'
            // 上传失败也要释放本地预览占用的 blob URL
            localObjectUrls.forEach((url) => URL.revokeObjectURL(url))
            updateLastMessage(sessionId, (m) => ({
              ...m,
              content: `文件解析失败：${message}`,
              references: [],
              reasoning: null,
            }))
            return
          }
        } else {
          const docEntry = docTextsRef.current.get(sessionId)
          if (docEntry && docEntry.anchorIdx >= history.length - 5) {
            documentText = docEntry.text
          }
        }

        // 清空占位内容，进入流式生成状态
        updateLastMessage(sessionId, (m) => ({ ...m, content: '', references: [], reasoning: null }))
        setRunningIds((prev) => new Set(prev).add(sessionId))
        setThinkingLabels((prev) => ({ ...prev, [sessionId]: '思考中' }))

        runStream(sessionId, (h) =>
          api.streamChat(
            sessionId,
            text,
            history,
            sessionTitle,
            h.onEvent,
            h.onDone,
            h.onError,
            documentText,
            files?.map((f) => f.name),
            uploadedAttachments,
          ),
        )
      })()
    },
    [
      currentSession,
      appendMessages,
      updateMessageAt,
      updateSessionTitle,
      switchSession,
      runStream,
      updateLastMessage,
      clearPendingFiles,
      clearPendingText,
    ],
  )

  // 全局文件拖拽上传：拖拽进入窗口时显示雾面遮罩，drop 后自动发送
  useEffect(() => {
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault()
      dragCounter.current += 1
      if (e.dataTransfer?.types.includes('Files')) {
        setIsDragOver(true)
      }
    }

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault()
      dragCounter.current = Math.max(0, dragCounter.current - 1)
      if (dragCounter.current === 0) {
        setIsDragOver(false)
      }
    }

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault()
    }

    const handleDrop = (e: DragEvent) => {
      e.preventDefault()
      dragCounter.current = 0
      setIsDragOver(false)

      const files = Array.from(e.dataTransfer?.files || [])
      if (files.length === 0 || !currentSession) return

      if (files.length > MAX_DRAG_FILES) {
        alert(`最多只能上传 ${MAX_DRAG_FILES} 个文件`)
        return
      }
      const oversized = files.find((f) => f.size > MAX_DRAG_FILE_SIZE)
      if (oversized) {
        alert('单个文件大小不能超过 5MB')
        return
      }

      // 将拖入的文件暂存到输入框，等待用户确认后发送
      if (!currentSession) return
      setPendingFiles(currentSession.id, (prev) => {
        const combined = [...prev, ...files]
        return combined.slice(0, MAX_DRAG_FILES)
      })
    }

    document.addEventListener('dragenter', handleDragEnter)
    document.addEventListener('dragleave', handleDragLeave)
    document.addEventListener('dragover', handleDragOver)
    document.addEventListener('drop', handleDrop)

    return () => {
      document.removeEventListener('dragenter', handleDragEnter)
      document.removeEventListener('dragleave', handleDragLeave)
      document.removeEventListener('dragover', handleDragOver)
      document.removeEventListener('drop', handleDrop)
    }
  }, [currentSession, handleSend])

  // 删除会话时同步清理本地材料窗口记录、待发送附件与文本
  const handleRemoveSession = useCallback(
    (id: string) => {
      docTextsRef.current.delete(id)
      clearPendingFiles(id)
      clearPendingText(id)
      removeSession(id)
    },
    [removeSession, clearPendingFiles, clearPendingText],
  )

  // 恢复加载：刷新/关闭标签页后，重新订阅仍在生成的会话，继续展示并写回
  useEffect(() => {
    if (!loaded) return
    let active = true
    api
      .listRunningJobs()
      .then((running) => {
        if (!active) return
        running.forEach((sid) => {
          if (resumedRef.current.has(sid)) return
          resumedRef.current.add(sid)

          const sess = sessionsRef.current.find((x) => x.id === sid)
          if (sess) {
            const last = sess.messages[sess.messages.length - 1]
            if (!last || last.role === 'user') {
              appendMessages(sid, [
                { role: 'assistant', content: '', references: [], reasoning: null },
              ])
            }
          }
          setRunningIds((prev) => new Set(prev).add(sid))
          setThinkingLabels((prev) => ({ ...prev, [sid]: '思考中' }))

          runStream(sid, (h) => api.resumeChat(sid, h.onEvent, h.onDone, h.onError))
        })
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [loaded, appendMessages, runStream])

  if (!currentSession) {
    return null
  }

  const hashId = window.location.hash.replace(/^#/, '')
  const isRestoring = !loaded && Boolean(hashId)
  const hasMessages = currentSession.messages.length > 0 || isRestoring
  const isCurrentLoading = currentId ? runningIds.has(currentId) : false
  const currentThinkingLabel = currentId ? thinkingLabels[currentId] || '思考中' : '思考中'

  return (
    <div className={`app ${mounted ? 'mounted' : ''}`}>
      {isDragOver && (
        <div className="drag-overlay">
          <div className="drag-overlay-card">
            <p className="drag-overlay-title">文件拖动即可上传</p>
            <p className="drag-overlay-subtitle">最多 3 个，每个 5MB</p>
          </div>
        </div>
      )}
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
        onDelete={handleRemoveSession}
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
        <ThemeToggle onToggle={toggleTheme} />
        <div className={`chat-layout ${hasMessages ? 'with-messages' : 'empty'} ${animated ? 'animated' : ''}`}>
          <div className="messages-area">
            <MessageList
              messages={currentSession.messages}
              loading={isCurrentLoading}
              restoring={isRestoring}
              thinkingLabel={currentThinkingLabel}
            />
          </div>
          <div className="input-area">
            <ChatInput
              onSend={handleSend}
              files={pendingFiles}
              onFilesChange={(files) => setPendingFiles(currentSession.id, files)}
              text={pendingText}
              onTextChange={(text) => setPendingText(currentSession.id, text)}
              disabled={isCurrentLoading}
            />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
