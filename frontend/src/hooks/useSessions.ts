import { useCallback, useEffect, useMemo, useState } from 'react'
import * as api from '../api'
import type { Session, SessionMessage } from '../types'

function localISOString(): string {
  const d = new Date()
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 19)
}

function newSession(): Session {
  return { id: crypto.randomUUID(), title: '新对话', updated_at: localISOString(), messages: [] }
}

function getHashId(): string | null {
  return window.location.hash.replace(/^#/, '') || null
}

function setHashId(id: string | null) {
  if (id) {
    window.location.hash = id
  } else {
    history.replaceState(null, '', window.location.pathname + window.location.search)
  }
}

export function useSessions() {
  const initialHash = getHashId()
  // 若 URL 已指向某历史会话，初始时不创建空的新会话占位；否则创建一个未保存的新会话
  const initialNewSession = initialHash ? null : newSession()
  const [sessions, setSessions] = useState<Session[]>(() =>
    initialHash ? [] : [initialNewSession!],
  )
  const [currentId, setCurrentId] = useState<string | null>(() => initialHash ?? initialNewSession?.id ?? null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let active = true
    const loadInitialHash = getHashId()
    api
      .listSessions()
      .then((stored) => {
        if (!active) return
        const storedIds = new Set(stored.map((s) => s.id))
        setSessions((prev) => {
          const existingIds = new Set(stored.map((s) => s.id))
          const kept = prev.filter((s) => !existingIds.has(s.id))
          return [...stored, ...kept]
        })
        // 刷新后若 URL hash 指向一个有效会话，则恢复它
        if (loadInitialHash && storedIds.has(loadInitialHash)) {
          setCurrentId(loadInitialHash)
        } else if (loadInitialHash) {
          // hash 对应会话不存在：可能是点击“新建对话”后未发送消息的临时会话，
          // 恢复为一个同 ID 的空会话，避免跳转到别的界面
          const recovered = { ...newSession(), id: loadInitialHash }
          setSessions((prev) => {
            const filtered = prev.filter((s) => s.id !== loadInitialHash)
            return [...filtered, recovered]
          })
          setCurrentId(loadInitialHash)
        }
        // 延迟标记 loaded，确保 sessions/currentId 同步后再渲染真实内容
        setTimeout(() => {
          if (!active) return
          setLoaded(true)
        }, 0)
      })
      .catch(() => {
        if (!active) return
        if (loadInitialHash) {
          // 后端列表获取失败时，保留 URL hash 并恢复为空会话，避免跳转别的界面
          const recovered = { ...newSession(), id: loadInitialHash }
          setSessions((prev) => {
            const filtered = prev.filter((s) => s.id !== loadInitialHash)
            return [...filtered, recovered]
          })
          setCurrentId(loadInitialHash)
        }
        setTimeout(() => {
          if (!active) return
          setLoaded(true)
        }, 0)
      })

    // 处理从浏览器 bfcache 恢复的情况：重新同步 URL hash 与当前会话
    const handlePageShow = (e: PageTransitionEvent) => {
      if (!e.persisted) return
      const restoredHash = getHashId()
      if (restoredHash) {
        setCurrentId(restoredHash)
      }
    }
    window.addEventListener('pageshow', handlePageShow)

    return () => {
      active = false
      window.removeEventListener('pageshow', handlePageShow)
    }
  }, [])

  const currentSession = useMemo(
    () => sessions.find((s) => s.id === currentId) || null,
    [sessions, currentId],
  )

  // 兜底：加载完成后若当前会话不存在（如删除后未补新会话），自动新建一个空会话
  useEffect(() => {
    if (loaded && !currentSession) {
      const ns = newSession()
      setSessions((prev) => [ns, ...prev])
      setCurrentId(ns.id)
    }
  }, [loaded, currentSession])

  const listedSessions = useMemo(
    () =>
      sessions
        .filter((s) => s.messages.length > 0)
        .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()),
    [sessions],
  )

  const switchSession = useCallback((id: string) => {
    setCurrentId(id)
    setHashId(id)
  }, [])

  const createSession = useCallback(() => {
    // 已有空会话时直接复用，避免堆积多个未使用的"新对话"
    const emptySession = sessions.find((s) => s.messages.length === 0)
    if (emptySession) {
      setCurrentId(emptySession.id)
      setHashId(null)
      return emptySession.id
    }
    const ns = newSession()
    setSessions((prev) => [ns, ...prev])
    setCurrentId(ns.id)
    // 主页/新建对话时 URL 保持干净，不带 hash
    setHashId(null)
    return ns.id
  }, [sessions])

  const removeSession = useCallback(
    async (id: string) => {
      try {
        await api.deleteSession(id)
      } catch {
        // 后端删除失败不阻塞前端状态
      }
      const remaining = sessions.filter((s) => s.id !== id)
      setSessions(remaining)
      if (id === currentId) {
        const candidates = remaining.filter((s) => s.messages.length > 0)
        if (candidates.length > 0) {
          setCurrentId(candidates[0].id)
          setHashId(candidates[0].id)
        } else {
          const ns = newSession()
          setSessions([ns])
          setCurrentId(ns.id)
          // 新建对话保持 URL 干净，不带 hash
          setHashId(null)
        }
      }
    },
    [currentId, sessions],
  )

  const appendMessages = useCallback(
    (sessionId: string, messages: SessionMessage[], title?: string) => {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== sessionId) return s
          const newTitle = title || s.title
          return {
            ...s,
            title: newTitle,
            messages: [...s.messages, ...messages],
            updated_at: localISOString(),
          }
        }),
      )
    },
    [],
  )

  const updateLastMessage = useCallback(
    (sessionId: string, updater: (msg: SessionMessage) => SessionMessage) => {
      setSessions((prev) => {
        const next = prev.map((s) => {
          if (s.id !== sessionId || s.messages.length === 0) return s
          const msgs = [...s.messages]
          msgs[msgs.length - 1] = updater(msgs[msgs.length - 1])
          return { ...s, messages: msgs }
        })
        return next
      })
    },
    [],
  )

  const updateSessionTitle = useCallback((sessionId: string, title: string) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === sessionId ? { ...s, title } : s)),
    )
  }, [])

  return {
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
    updateSessionTitle,
  }
}
