import { useEffect, useRef, useState } from 'react'
import { ArrowUp, Plus, X } from 'lucide-react'

interface ChatInputProps {
  onSend: (text: string, files?: File[]) => void
  files?: File[]
  onFilesChange?: (files: File[]) => void
  text?: string
  onTextChange?: (text: string) => void
  disabled?: boolean
  placeholder?: string
}

const ACCEPT_TYPES = '.docx,.pdf,.png,.jpg,.jpeg,.webp,.bmp'
const MAX_FILE_SIZE = 5 * 1024 * 1024 // 5MB
const MAX_FILE_COUNT = 3

export function ChatInput({
  onSend,
  files: externalFiles,
  onFilesChange,
  text: externalText,
  onTextChange,
  disabled,
  placeholder = '请输入您的法律问题',
}: ChatInputProps) {
  const [internalText, setInternalText] = useState('')
  const [internalFiles, setInternalFiles] = useState<File[]>([])
  const isTextControlled = externalText !== undefined
  const text = isTextControlled ? externalText : internalText
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const objectUrlMapRef = useRef<Map<File, string>>(new Map())
  const isControlled = externalFiles !== undefined
  const files = isControlled ? externalFiles : internalFiles

  // 图片预览 URL 管理：文件从列表移除时立即释放，组件卸载时释放全部
  useEffect(() => {
    const currentFiles = new Set(files)
    const map = objectUrlMapRef.current
    map.forEach((url, file) => {
      if (!currentFiles.has(file)) {
        URL.revokeObjectURL(url)
        map.delete(file)
      }
    })
  }, [files])

  useEffect(() => {
    return () => {
      objectUrlMapRef.current.forEach((url) => URL.revokeObjectURL(url))
      objectUrlMapRef.current.clear()
    }
  }, [])

  const setFiles = (updater: React.SetStateAction<File[]>) => {
    if (isControlled) {
      if (!onFilesChange) return
      const next = typeof updater === 'function' ? (updater as (prev: File[]) => File[])(files) : updater
      onFilesChange(next)
    } else {
      setInternalFiles(updater)
    }
  }

  const submit = () => {
    const trimmed = text.trim()
    if ((!trimmed && files.length === 0) || disabled) return
    onSend(trimmed, files.length > 0 ? files : undefined)
    if (isTextControlled) {
      onTextChange?.('')
    } else {
      setInternalText('')
    }
    if (!isControlled) {
      setInternalFiles([])
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files || [])
    if (selected.length === 0) return

    const availableSlots = MAX_FILE_COUNT - files.length
    if (availableSlots <= 0) {
      alert(`最多只能上传 ${MAX_FILE_COUNT} 个文件`)
      return
    }

    const toAdd = selected.slice(0, availableSlots)
    const oversized = toAdd.find((f) => f.size > MAX_FILE_SIZE)
    if (oversized) {
      alert('单个文件大小不能超过 5MB')
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
      return
    }

    setFiles((prev) => [...prev, ...toAdd])
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const canSend = (text.trim() || files.length > 0) && !disabled
  const isImage = (file: File) => file.type.startsWith('image/')

  const getPreviewUrl = (file: File) => {
    const map = objectUrlMapRef.current
    if (!map.has(file)) {
      map.set(file, URL.createObjectURL(file))
    }
    return map.get(file)!
  }

  return (
    <div className="chat-input-bar">
      <div className="chat-input-wrap">
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPT_TYPES}
          multiple
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        {files.length > 0 && (
          <div className="file-chips">
            {files.map((file, index) => (
              <div
                key={`${file.name}-${index}`}
                className={`file-chip ${isImage(file) ? 'image-chip' : ''}`}
              >
                {isImage(file) ? (
                  <img
                    src={getPreviewUrl(file)}
                    alt={file.name}
                    className="file-thumb"
                  />
                ) : (
                  <span className="file-name" title={file.name}>
                    {file.name}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => removeFile(index)}
                  aria-label="移除文件"
                  className="file-remove-btn"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => {
              if (isTextControlled) {
                onTextChange?.(e.target.value)
              } else {
                setInternalText(e.target.value)
              }
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
          />
          <button
            type="button"
            className="attach-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || files.length >= MAX_FILE_COUNT}
            aria-label="上传文件"
            title="上传文件"
          >
            <Plus size={18} />
          </button>
          <button
            type="button"
            className="send-btn"
            onClick={submit}
            disabled={!canSend}
            aria-label="发送"
          >
            <ArrowUp size={18} />
          </button>
        </div>
      </div>
    </div>
  )
}
