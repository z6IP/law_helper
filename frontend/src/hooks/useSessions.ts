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
  const initialSession = useMemo(() => newSession(), [])
  const hashId = getHashId()
  const [sessions, setSessions] = useState<Session[]>([initialSession])
  const [currentId, setCurrentId] = useState<string | null>(hashId || initialSession.id)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let active = true
    const initialHash = getHashId()
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
        // 刷新后若 URL hash 指向一个有效历史会话，则恢复它
        if (initialHash && storedIds.has(initialHash)) {
          setCurrentId(initialHash)
        } else if (initialHash) {
          // hash 无效时清空
          setHashId(null)
        }
        // 延迟标记 loaded，确保 sessions/currentId 同步后再渲染真实内容
        setTimeout(() => {
          if (!active) return
          setLoaded(true)
        }, 0)
      })
      .catch(() => {
        if (!active) return
        if (initialHash) setHashId(null)
        setTimeout(() => {
          if (!active) return
          setLoaded(true)
        }, 0)
      })
    return () => {
      active = false
    }
  }, [])

  const currentSession = useMemo(
    () => sessions.find((s) => s.id === currentId) || sessions[0],
    [sessions, currentId],
  )

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
    const ns = newSession()
    setSessions((prev) => [ns, ...prev])
    setCurrentId(ns.id)
    setHashId(ns.id)
    return ns.id
  }, [])

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
          setHashId(ns.id)
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
