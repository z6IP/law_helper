import { useEffect, useState } from 'react'
import { Moon, Sun } from 'lucide-react'

interface ThemeToggleProps {
  onToggle: () => void
}

// 主题状态以 DOM（<html data-theme>）为唯一事实来源，本组件仅同步图标，
// 避免主题切换引起 App 整棵 React 树重渲染（卡顿主因）。
export function ThemeToggle({ onToggle }: ThemeToggleProps) {
  const [isDark, setIsDark] = useState(
    () => document.documentElement.getAttribute('data-theme') === 'dark',
  )

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ dark: boolean }>).detail
      if (detail && typeof detail.dark === 'boolean') {
        setIsDark(detail.dark)
      } else {
        setIsDark(document.documentElement.getAttribute('data-theme') === 'dark')
      }
    }
    window.addEventListener('theme-change', handler)
    return () => window.removeEventListener('theme-change', handler)
  }, [])

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onToggle}
      title={isDark ? '切换为亮色模式' : '切换为暗色模式'}
      aria-label={isDark ? '切换为亮色模式' : '切换为暗色模式'}
    >
      {isDark ? <Sun size={20} /> : <Moon size={20} />}
    </button>
  )
}
