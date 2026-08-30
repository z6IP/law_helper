import { PanelLeftClose, Trash2 } from 'lucide-react'
import type { Session } from '../types'

interface SidebarProps {
  sessions: Session[]
  currentId: string | null
  visible: boolean
  loaded?: boolean
  onNew: () => void
  onClose: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

export function Sidebar({ sessions, currentId, visible, loaded = true, onNew, onClose, onSelect, onDelete }: SidebarProps) {
  return (
    <aside className={`sidebar ${visible ? '' : 'collapsed'}`}>
      <div className="sidebar-inner">
        <div className="sidebar-header">
          <button type="button" className="new-session-btn" onClick={onNew}>
            + 新建对话
          </button>
          <button
            type="button"
            className="sidebar-close"
            onClick={onClose}
            title="关闭侧边栏"
            aria-label="关闭侧边栏"
          >
            <PanelLeftClose size={18} />
          </button>
        </div>
        <div className="session-list">
          {!loaded && (
            <>
              <div className="session-skeleton" />
              <div className="session-skeleton" />
              <div className="session-skeleton" />
            </>
          )}
          {loaded && sessions.map((s) => {
            const isCurrent = s.id === currentId
            return (
              <div key={s.id} className={`session-item ${isCurrent ? 'current' : ''}`}>
                <button
                  type="button"
                  className="session-title"
                  onClick={() => onSelect(s.id)}
                  title={s.title}
                >
                  {s.title}
                </button>
                <button
                  type="button"
                  className="session-delete"
                  onClick={() => onDelete(s.id)}
                  title="删除会话"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </aside>
  )
}
