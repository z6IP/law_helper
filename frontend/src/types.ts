export interface Reference {
  source: string
  article_no: string
  section_header: string
  text: string
  merged_from?: string[]
}

export interface Attachment {
  name: string
  type: 'image' | 'document'
  url: string
}

export interface SessionMessage {
  role: 'user' | 'assistant'
  content: string
  references?: Reference[]
  reasoning?: string | null
  fileNames?: string[]
  attachments?: Attachment[]
  /** 发送时的深度思考快照：渲染等待指示器用，生成中切换全局开关不影响本条 */
  deepThinking?: boolean
}

export interface Session {
  id: string
  title: string
  updated_at: string
  messages: SessionMessage[]
}

export interface ChatStreamEvent {
  type: 'references' | 'progress' | 'reasoning' | 'delta'
  references?: Reference[]
  content?: string
}
