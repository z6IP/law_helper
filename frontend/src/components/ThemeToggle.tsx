import { Moon, Sun } from 'lucide-react'

interface ThemeToggleProps {
  isDark: boolean
  onToggle: () => void
}

export function ThemeToggle({ isDark, onToggle }: ThemeToggleProps) {
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
