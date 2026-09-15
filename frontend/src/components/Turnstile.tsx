import { useEffect, useRef } from 'react'

declare global {
  interface Window {
    turnstile?: {
      render: (el: HTMLElement, opts: Record<string, unknown>) => string
      remove: (id: string) => void
      reset: (id?: string) => void
    }
  }
}

const SCRIPT_SRC = 'https://challenges-china.cloudflare.com/turnstile/v0/api.js?render=explicit'
// 与后端 app/turnstile.py 的 EXPECTED_ACTION 保持一致
const ACTION = 'chat'

let scriptPromise: Promise<void> | null = null

function loadScript(): Promise<void> {
  if (window.turnstile) return Promise.resolve()
  if (scriptPromise) return scriptPromise
  scriptPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = SCRIPT_SRC
    script.async = true
    script.defer = true
    script.onload = () => resolve()
    script.onerror = () => {
      scriptPromise = null
      reject(new Error('Turnstile 脚本加载失败'))
    }
    document.head.appendChild(script)
  })
  return scriptPromise
}

interface TurnstileProps {
  siteKey: string
  onToken: (token: string) => void
  /** 脚本加载失败等致命错误回调（如 challenges 脚本被墙时） */
  onError?: (err: Error) => void
  /** 供外部调用 reset：一次性 token 消费后刷新，取得新 token */
  resetRef?: { current: (() => void) | null }
}

/**
 * 显式渲染 Turnstile（Managed 模式 + interaction-only 外观）：
 * 正常用户不可见，可疑流量才弹出交互挑战；token 单次有效，消费后需 reset。
 */
export function Turnstile({ siteKey, onToken, onError, resetRef }: TurnstileProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const widgetIdRef = useRef<string | null>(null)
  const onTokenRef = useRef(onToken)
  const onErrorRef = useRef(onError)
  onTokenRef.current = onToken
  onErrorRef.current = onError

  useEffect(() => {
    let cancelled = false
    loadScript()
      .then(() => {
        if (cancelled || !containerRef.current || !window.turnstile) return
        widgetIdRef.current = window.turnstile.render(containerRef.current, {
          sitekey: siteKey,
          action: ACTION,
          appearance: 'interaction-only',
          callback: (token: string) => onTokenRef.current(token),
          'expired-callback': () => onTokenRef.current(''),
          'error-callback': () => onTokenRef.current(''),
        })
        if (resetRef) {
          resetRef.current = () => window.turnstile?.reset(widgetIdRef.current ?? undefined)
        }
      })
      .catch((err) => {
        // 脚本加载失败必须上报：否则 token 永远不产生，用户被永久卡在发送拦截
        if (!cancelled) onErrorRef.current?.(err instanceof Error ? err : new Error(String(err)))
      })
    return () => {
      cancelled = true
      if (resetRef) resetRef.current = null
      if (widgetIdRef.current !== null && window.turnstile) {
        window.turnstile.remove(widgetIdRef.current)
        widgetIdRef.current = null
      }
    }
  }, [siteKey, resetRef])

  return <div ref={containerRef} className="turnstile-slot" />
}
