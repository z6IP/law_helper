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

export async function deleteSession(sessionId: string): Promise<void> {
  await fetchJson<{ status: string }>(`/sessions/${sessionId}`, { method: 'DELETE' })
}

export async function fetchLLMModel(): Promise<string> {
  const data = await fetchJson<{ llm_model: string }>('/settings/llm_model')
  return data.llm_model
}

export async function listRunningJobs(): Promise<string[]> {
  const data = await fetchJson<{ sessions: string[] }>('/chat/jobs/running')
  return data.sessions
}

export type StreamEvent = { type: string; data?: unknown }
export type StreamHandlers = {
  onEvent: (event: StreamEvent) => void
  onDone: () => void
  onError: (err: Error) => void
}

async function consumeNDJSON(
  resp: Response,
  onEvent: (event: StreamEvent) => void,
  onError: (err: Error) => void,
): Promise<void> {
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
}

function runStream(
  makeFetch: (abort: AbortController) => Promise<Response>,
  handlers: StreamHandlers,
): () => void {
  const abort = new AbortController()
  makeFetch(abort)
    .then((resp) => consumeNDJSON(resp, handlers.onEvent, handlers.onError))
    .then(handlers.onDone)
    .catch((err) => {
      if (err.name === 'AbortError') return
      handlers.onError(err)
    })
  return () => abort.abort()
}

export async function summarizeSession(sessionId: string, messages: SessionMessage[]): Promise<string> {
  const data = await fetchJson<{ title: string }>(`/sessions/${sessionId}/summarize`, {
    method: 'POST',
    body: JSON.stringify({ messages }),
  })
  return data.title
}

export async function uploadDocument(file: File): Promise<string> {
  const formData = new FormData()
  formData.append('file', file)
  const resp = await fetch(`${BASE}/chat/upload`, {
    method: 'POST',
    credentials: 'include',
    body: formData,
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`HTTP ${resp.status}: ${text}`)
  }
  const data = (await resp.json()) as { text: string }
  return data.text
}

export function streamChat(
  sessionId: string,
  question: string,
  history: SessionMessage[],
  title: string,
  onEvent: (event: StreamEvent) => void,
  onDone: () => void,
  onError: (err: Error) => void,
  documentText?: string,
): () => void {
  const payload = {
    question,
    session_id: sessionId,
    title,
    document_text: documentText,
    history: history.map((m) => ({
      role: m.role,
      content: m.content,
      references: m.references || [],
      reasoning: m.reasoning ?? null,
    })),
  }
  return runStream(
    (abort) =>
      fetch(`${BASE}/chat/stream`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abort.signal,
      }),
    { onEvent, onDone, onError },
  )
}

export function resumeChat(
  sessionId: string,
  onEvent: (event: StreamEvent) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): () => void {
  return runStream(
    (abort) =>
      fetch(`${BASE}/chat/jobs/${sessionId}/stream`, {
        credentials: 'include',
        signal: abort.signal,
      }),
    { onEvent, onDone, onError },
  )
}

export { type Reference, type Session, type SessionMessage }