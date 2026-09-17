import { Fragment, memo, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { ChevronDown, FileText, Image } from 'lucide-react'
import { References } from './References'
import type { Attachment, SessionMessage } from '../types'

// 思考过程专用：仅将 ## / ### 标题转为 h2 / h3，其余按纯文本输出（配合 pre-wrap 保留换行）。
// 返回 React 元素数组（而非 HTML 字符串 + dangerouslySetInnerHTML），
// 让 React 只增量更新变化的 text node，避免流式期间每帧重建整个 DOM。
function renderPlain(src: string): ReactNode[] {
  const lines = src.split('\n')
  const nodes: ReactNode[] = []
  let prevIsHeading = false
  lines.forEach((line, i) => {
    if (line.startsWith('## ')) {
      nodes.push(<h2 key={i}>{line.slice(3)}</h2>)
      prevIsHeading = true
    } else if (line.startsWith('### ')) {
      nodes.push(<h3 key={i}>{line.slice(4)}</h3>)
      prevIsHeading = true
    } else {
      // 普通文本行：紧跟块级标题后时不补换行（标题自身已换行），
      // 避免 white-space: pre-wrap 下渲染出多余空行。
      const sep = nodes.length > 0 && !prevIsHeading ? '\n' : ''
      nodes.push(<Fragment key={i}>{sep}{line}</Fragment>)
      prevIsHeading = false
    }
  })
  return nodes
}

// ===== 正文轻量 Markdown：块级 + 行内，全部产出 React 节点（无 innerHTML、无 XSS）=====
// 大标题 ## -> h2(22px)；小标题 ### 或独占一行的 **xxx** -> h3(18px)；
// 列表 - / * -> ul，1. -> ol（连续行合并为一个列表）；其余非空行 -> p(16px)。
// React 自动转义文本，无 XSS 风险，且保留增量 diff 无 DOM 重建。
const INLINE_RE = /(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/g

// 行内解析：**加粗** / *斜体* / `代码`
function renderInline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = []
  INLINE_RE.lastIndex = 0
  let last = 0
  let n = 0
  let m: RegExpExecArray | null
  while ((m = INLINE_RE.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const token = m[0]
    const key = `${keyBase}-i${n++}`
    if (token.startsWith('**')) out.push(<strong key={key}>{token.slice(2, -2)}</strong>)
    else if (token.startsWith('`')) out.push(<code key={key}>{token.slice(1, -1)}</code>)
    else out.push(<em key={key}>{token.slice(1, -1)}</em>)
    last = m.index + token.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

function renderMarkdown(src: string): ReactNode[] {
  const lines = src.split('\n')
  const nodes: ReactNode[] = []
  let listBuf: { text: string; key: number }[] = []
  let listTag: 'ul' | 'ol' = 'ul'

  const flushList = () => {
    if (listBuf.length === 0) return
    const items = listBuf.map((it) => <li key={it.key}>{renderInline(it.text, `li-${it.key}`)}</li>)
    const Tag = listTag
    nodes.push(<Tag key={`${listTag}-${listBuf[0].key}`}>{items}</Tag>)
    listBuf = []
  }

  lines.forEach((line, i) => {
    if (line.startsWith('### ')) {
      flushList()
      nodes.push(<h3 key={i}>{renderInline(line.slice(4), `h3-${i}`)}</h3>)
      return
    }
    if (line.startsWith('## ')) {
      flushList()
      nodes.push(<h2 key={i}>{renderInline(line.slice(3), `h2-${i}`)}</h2>)
      return
    }
    // 独占一行的 **xxx** 视为小标题（模型常以其代替 ### 作小标题）
    const bold = /^\s*\*\*([^*]{1,40})\*\*\s*$/.exec(line)
    if (bold) {
      flushList()
      nodes.push(<h3 key={i}>{renderInline(bold[1], `bh-${i}`)}</h3>)
      return
    }
    const ul = /^\s*[-*]\s+(.*)$/.exec(line)
    const ol = /^\s*\d+[.、)]\s+(.*)$/.exec(line)
    if (ul || ol) {
      const tag: 'ul' | 'ol' = ul ? 'ul' : 'ol'
      if (listTag !== tag) flushList()
      listTag = tag
      listBuf.push({ text: (ul ? ul[1] : ol![1]).trim(), key: i })
      return
    }
    // 空行：仅作块间分隔，不产出节点
    if (line.trim() === '') {
      flushList()
      return
    }
    flushList()
    nodes.push(<p key={i}>{renderInline(line, `p-${i}`)}</p>)
  })
  flushList()
  return nodes
}


const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'])

function getFileExt(name: string): string {
  const idx = name.lastIndexOf('.')
  return idx === -1 ? '' : name.slice(idx).toLowerCase()
}

function FileAttachment({ name }: { name: string }) {
  const ext = getFileExt(name)
  const isImage = IMAGE_EXTS.has(ext)
  return (
    <div className={`file-attachment-box ${isImage ? 'image' : 'file'}`}>
      {isImage ? <Image size={18} /> : <FileText size={18} />}
      <span className="file-attachment-name" title={name}>
        {name}
      </span>
    </div>
  )
}

function AttachmentItem({ attachment, onPreview }: { attachment: Attachment; onPreview?: (url: string, rect: DOMRect) => void }) {
  const [error, setError] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)

  // 切换会话复用组件时，重置加载状态，避免上一会话的失败/加载状态影响当前图片
  useEffect(() => {
    setError(false)
    setLoaded(false)
    // 刷新后图片可能已在浏览器缓存中，onLoad 会在 React 挂载 img 前触发，
    // 导致 loaded 永远为 false、opacity 保持 0 显示黑屏。
    // 此处主动检查 img.complete，已加载则立即标记。
    if (imgRef.current?.complete) {
      setLoaded(true)
    }
  }, [attachment.url])

  if (attachment.type === 'image' && !error) {
    return (
      <div className="image-attachment">
        <img
          ref={imgRef}
          src={attachment.url}
          alt={attachment.name}
          crossOrigin="use-credentials"
          onLoad={() => setLoaded(true)}
          onError={() => setError(true)}
          onClick={(e) => {
            onPreview?.(attachment.url, (e.currentTarget as HTMLElement).getBoundingClientRect())
          }}
          style={{ cursor: 'pointer', opacity: loaded ? 1 : 0 }}
        />
      </div>
    )
  }
  return <FileAttachment name={attachment.name} />
}

interface ImagePreviewProps {
  url: string
  sourceRect: DOMRect
  onClose: () => void
}

function ImagePreview({ url, sourceRect, onClose }: ImagePreviewProps) {
  const overlayRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [closing, setClosing] = useState(false)

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    // 预览打开时标记全局状态，暂停消息列表自动滚动，避免流式输出干扰用户查看
    document.body.setAttribute('data-preview-open', 'true')
    const target = computeTargetRect(sourceRect)
    const start = {
      x: sourceRect.left - target.left,
      y: sourceRect.top - target.top,
      scaleX: sourceRect.width / target.width,
      scaleY: sourceRect.height / target.height,
    }
    img.style.left = `${target.left}px`
    img.style.top = `${target.top}px`
    img.style.width = `${target.width}px`
    img.style.height = `${target.height}px`
    img.style.transform = `translate3d(${start.x}px, ${start.y}px, 0) scale3d(${start.scaleX}, ${start.scaleY}, 1)`

    const overlay = overlayRef.current
    if (overlay) {
      overlay.style.opacity = '0'
    }

    requestAnimationFrame(() => {
      img.style.transition = 'transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1)'
      img.style.transform = 'translate3d(0, 0, 0) scale3d(1, 1, 1)'
      if (overlay) {
        overlay.style.transition = 'opacity 0.35s ease'
        overlay.style.opacity = '1'
      }
    })

    return () => {
      document.body.removeAttribute('data-preview-open')
    }
  }, [sourceRect])

  const handleClose = () => {
    if (closing) return
    setClosing(true)
    const img = imgRef.current
    if (img) {
      const target = computeTargetRect(sourceRect)
      const end = {
        x: sourceRect.left - target.left,
        y: sourceRect.top - target.top,
        scaleX: sourceRect.width / target.width,
        scaleY: sourceRect.height / target.height,
      }
      img.style.transition = 'transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1)'
      img.style.transform = `translate3d(${end.x}px, ${end.y}px, 0) scale3d(${end.scaleX}, ${end.scaleY}, 1)`
    }
    const overlay = overlayRef.current
    if (overlay) {
      overlay.style.transition = 'opacity 0.15s ease'
      overlay.style.opacity = '0'
    }
    setTimeout(() => {
      onClose()
    }, 350)
  }

  return (
    <div
      ref={overlayRef}
      className="image-preview-overlay"
      onClick={(e) => {
        e.stopPropagation()
        handleClose()
      }}
      role="button"
      aria-label="关闭预览"
      tabIndex={-1}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          e.stopPropagation()
          handleClose()
        }
      }}
    >
      <img ref={imgRef} src={url} alt="预览" crossOrigin="use-credentials" className="image-preview-floating" onClick={(e) => { e.stopPropagation(); handleClose() }} />
    </div>
  )
}

function computeTargetRect(sourceRect: DOMRect) {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const maxWidth = vw * 0.75
  const maxHeight = vh * 0.75
  const scale = Math.min(maxWidth / sourceRect.width, maxHeight / sourceRect.height)
  const width = sourceRect.width * scale
  const height = sourceRect.height * scale
  return {
    left: (vw - width) / 2,
    top: (vh - height) / 2,
    width,
    height,
  }
}

interface MessageItemProps {
  message: SessionMessage
  isCurrentLoading?: boolean
  reasoningLoading?: boolean
  thinkingLabel?: string
}

export const MessageItem = memo(function MessageItem({ message, isCurrentLoading, reasoningLoading, thinkingLabel }: MessageItemProps) {
  const [reasoningOpen, setReasoningOpen] = useState(false)
  interface PreviewInfo {
    url: string
    rect: DOMRect
  }
  
  const [preview, setPreview] = useState<PreviewInfo | null>(null)
  const isUser = message.role === 'user'

  // 正文用 renderMarkdown（块级 + 行内，标题/列表/加粗均产出 React 节点），
  // 思考过程用 renderPlain（纯文本 + h2/h3）；
  // 均不调用 marked.parse + DOMPurify，React 增量 diff 无切换闪屏、无 DOM 重建。

  if (isUser) {
    const hasAttachments = message.attachments && message.attachments.length > 0
    const hasFileNames = message.fileNames && message.fileNames.length > 0
    const hasText = message.content.trim().length > 0
    return (
      <>
        {preview && <ImagePreview url={preview.url} sourceRect={preview.rect} onClose={() => setPreview(null)} />}
        <div className="message message-user">
          <div className="message-user-content">
            {(hasAttachments || hasFileNames) && (
              <div className="message-user-attachments">
                {hasAttachments
                  ? message.attachments!.map((attachment, index) => (
                      <AttachmentItem key={`${attachment.url}-${index}`} attachment={attachment} onPreview={(url, rect) => setPreview({ url, rect })} />
                    ))
                  : message.fileNames!.map((name, index) => (
                      <FileAttachment key={`${name}-${index}`} name={name} />
                    ))}
              </div>
            )}
            {hasText && <div className="message-user-bubble">{message.content}</div>}
          </div>
        </div>
      </>
    )
  }

  const hasReasoning = message.reasoning !== null && message.reasoning !== ''
  // 只读消息自带的发送时快照，不读全局开关：生成中切换开关不影响本条
  const showReasoning = hasReasoning || (isCurrentLoading && !!message.deepThinking)

  return (
    <div className="message message-assistant">
      <div className="message-assistant-content">
        {showReasoning && (
          <div className="reasoning-block">
            <button
              type="button"
              className="reasoning-toggle"
              onClick={() => setReasoningOpen((v) => !v)}
            >
              <ChevronDown size={14} className={reasoningOpen ? 'open' : ''} />
              {reasoningLoading ? (
                <>
                  <span className="spinner" />
                  <span>{thinkingLabel || '思考中'}</span>
                </>
              ) : (
                <span>思考完成</span>
              )}
            </button>
            {reasoningOpen && (
              <div className="reasoning-body">{renderPlain(message.reasoning || '')}</div>
            )}
          </div>
        )}
        {/* 互斥兜底：推理块显示期间绝不同时渲染普通加载圈 */}
        {isCurrentLoading && !message.deepThinking && !hasReasoning && !message.content && (
          <div className="answer-loading">
            <span className="spinner" />
          </div>
        )}
        <div className="markdown-body">{renderMarkdown(message.content)}</div>
        {!isCurrentLoading && <References references={message.references || []} />}
      </div>
    </div>
  )
})
