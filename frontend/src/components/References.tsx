import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import type { Reference } from '../types'

interface ReferencesProps {
  references: Reference[]
}

export function References({ references }: ReferencesProps) {
  const [open, setOpen] = useState(false)
  if (!references || references.length === 0) return null

  return (
    <div className="references">
      <button type="button" className="references-toggle" onClick={() => setOpen((v) => !v)}>
        <ChevronDown size={14} className={open ? 'open' : ''} />
        <span>查看引用法条</span>
      </button>
      {open && (
        <div className="references-list">
          {references.map((ref, idx) => {
            const header = `${ref.source ? `《${ref.source}》` : ''}${ref.article_no}${
              ref.section_header ? `（${ref.section_header}）` : ''
            }`
            return (
              <div key={idx} className="reference-item">
                <div className="reference-header">{header}</div>
                <div className="reference-text">{ref.text}</div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
