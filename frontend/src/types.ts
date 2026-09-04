export interface Reference {
  source: string
  article_no: string
  section_header: string
  text: string
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
