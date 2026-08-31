import type { Reference, Session, SessionMessage } from './types'

const BASE = '/api/v1'

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`HTTP ${resp.status}: ${text}`)
  }
  return resp.json() as Promise<T>
}

export async function listSessions(): Promise<Session[]> {
  return fetchJson<Session[]>('/sessions')
}

export async function saveSession(session: Session): Promise<Session> {
  return fetchJson<Session>(`/sessions/${session.id}`, {
    method: 'PUT',
    body: JSON.stringify({
      title: session.title,
      messages: session.messages,
    }),
  })
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetchJson<{ status: string }>(`/sessions/${sessionId}`, {
    method: 'DELETE',
  })
}

export async function fetchLLMModel(): Promise<string> {
  const data = await fetchJson<{ llm_model: string }>('/settings/llm_model')
  return data.llm_model
}

export function streamChat(
  question: string,
  history: SessionMessage[],
  sessionId: string,
  onEvent: (event: { type: string; data?: unknown }) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): () => void {
  const payload = {
    question,
    session_id: sessionId,
    history: history.map((m) => ({ role: m.role, content: m.content })),
  }

  const abort = new AbortController()

  fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: abort.signal,
  })
    .then(async (resp) => {
      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${text}`)
      }
      if (!resp.body) {
        throw new Error('响应体为空')
      }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (!line.trim()) continue
          try {
            const data = JSON.parse(line)
            if (data.type === 'error') {
              onError(new Error(data.content || '服务端处理出错'))
              return
            }
            onEvent({ type: data.type, data })
          } catch {
            // 忽略无法解析的行
          }
        }
      }
      onDone()
    })
    .catch((err) => {
      if (err.name === 'AbortError') return
      onError(err)
    })

  return () => abort.abort()
}

export { type Reference, type Session, type SessionMessage }
