import { useEffect, useRef, useState } from 'react'

declare global {
  interface Window {
    turnstile?: {
      render: (el: HTMLElement, opts: Record<string, unknown>) => string
      remove: (id: string) => void
      reset: (id?: string) => void
    }
  }
}

// 后端未下发脚本源时的兜底默认值
const DEFAULT_SCRIPT_SRCS = ['https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit']
// 单次脚本加载超时，与后端 siteverify 的 timeout=10 对齐。
// 该域名在大陆网络的典型故障是「长时间挂起」而非立即报错，
// 只靠 onerror 会让用户永久停在「验证进行中」。
const LOAD_TIMEOUT_MS = 10_000
// 与后端 app/turnstile.py 的 EXPECTED_ACTION 保持一致
const ACTION = 'chat'

let scriptPromise: Promise<void> | null = null

/** 加载单个脚本源；超时或出错都算失败，由调用方决定是否回退到下一个源。 */
function loadOneScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    const timer = window.setTimeout(() => {
      script.remove()
      reject(new Error(`验证脚本加载超时（${src}）`))
    }, LOAD_TIMEOUT_MS)
    script.src = src
    script.async = true
    script.defer = true
    script.onload = () => {
      window.clearTimeout(timer)
      resolve()
    }
    script.onerror = () => {
      window.clearTimeout(timer)
      script.remove()
      reject(new Error(`验证脚本加载失败（${src}）`))
    }
    document.head.appendChild(script)
  })
}

/** 按顺序尝试各脚本源；全站只加载一次，失败后清空缓存以便用户手动重试。 */
function loadScript(srcs: string[]): Promise<void> {
  if (window.turnstile) return Promise.resolve()
  if (scriptPromise) return scriptPromise
  scriptPromise = (async () => {
    const errors: string[] = []
    for (const src of srcs) {
      // 前一个源可能在被判定超时后才真正加载完成，此时无需再试后续源
      if (window.turnstile) return
      try {
        await loadOneScript(src)
        return
      } catch (err) {
        errors.push(err instanceof Error ? err.message : String(err))
      }
    }
    throw new Error(errors.join('；') || '验证脚本加载失败')
  })()
  scriptPromise = scriptPromise.catch((err) => {
    scriptPromise = null
    throw err
  })
  return scriptPromise
}

interface TurnstileProps {
  siteKey: string
  /** 脚本地址候选列表（按顺序回退）；缺省用内置默认源 */
  scriptSrcs?: string[]
  onToken: (token: string) => void
  /** 脚本加载失败等致命错误回调（如 challenges 脚本被墙时） */
  onError?: (err: Error) => void
  /** widget 渲染成功回调：调用方据此把 UI 从「加载中」切到「就绪」 */
  onReady?: () => void
  /** 供外部调用 reset：一次性 token 消费后刷新，取得新 token */
  resetRef?: { current: (() => void) | null }
  /** 供外部触发整体重试：加载失败后重新加载脚本并渲染 widget */
  retryRef?: { current: (() => void) | null }
}

/**
 * 显式渲染 Turnstile（Managed 模式 + interaction-only 外观）：
 * 正常用户不可见，可疑流量才弹出交互挑战；token 单次有效，消费后需 reset。
 *
 * 脚本加载带超时与多源回退，并可通过 retryRef 手动重试——
 * 加载失败不再让调用方永久卡在「无 token 不能发送」的状态。
 */
export function Turnstile({
  siteKey,
  scriptSrcs,
  onToken,
  onError,
  onReady,
  resetRef,
  retryRef,
}: TurnstileProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const widgetIdRef = useRef<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  const onTokenRef = useRef(onToken)
  const onErrorRef = useRef(onError)
  const onReadyRef = useRef(onReady)
  onTokenRef.current = onToken
  onErrorRef.current = onError
  onReadyRef.current = onReady

  // 依赖指纹用 JSON 而非 join(',')：URL 的查询串允许出现逗号（如 ?render=explicit,onload），
  // 用逗号拼拆会把一个地址切成两个不存在的地址，导致脚本加载必然失败且重试无效。
  const srcsKey = JSON.stringify(
    scriptSrcs && scriptSrcs.length > 0 ? scriptSrcs : DEFAULT_SCRIPT_SRCS,
  )

  useEffect(() => {
    let cancelled = false
    if (retryRef) retryRef.current = () => setAttempt((n) => n + 1)
    loadScript(JSON.parse(srcsKey) as string[])
      .then(() => {
        if (cancelled) return
        const container = containerRef.current
        if (!container || !window.turnstile) return
        // 重试时清理上一次渲染残留，避免同一容器叠加多个 widget
        container.innerHTML = ''
        widgetIdRef.current = window.turnstile.render(container, {
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
        onReadyRef.current?.()
      })
      .catch((err) => {
        // 脚本加载失败必须上报：否则 token 永远不产生，用户被永久卡在发送拦截
        if (!cancelled) onErrorRef.current?.(err instanceof Error ? err : new Error(String(err)))
      })
    return () => {
      cancelled = true
      if (resetRef) resetRef.current = null
      if (retryRef) retryRef.current = null
      if (widgetIdRef.current !== null && window.turnstile) {
        window.turnstile.remove(widgetIdRef.current)
        widgetIdRef.current = null
      }
    }
  }, [siteKey, srcsKey, attempt, resetRef, retryRef])

  return <div ref={containerRef} className="turnstile-slot" />
}
