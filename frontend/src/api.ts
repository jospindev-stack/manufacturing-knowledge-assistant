const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export type Source = {
  source: string
  page: number
  chunk_index: number
  similarity: number
}

export type ChatResponse = {
  answer: string
  grounded: boolean
  sources: Source[]
}

export type SearchResponse = {
  query: string
  top_k: number
  results: Array<Source & { text: string }>
}

export type EvaluationCase = {
  question: string
  expected_source: string
  expected_page?: number
}

export type EvaluationResponse = {
  total: number
  hits: number
  hit_rate: number
  mean_reciprocal_rank: number
  cases: Array<EvaluationCase & { hit: boolean; rank: number | null; top_similarity: number | null }>
}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export async function uploadDocument(file: File) {
  const form = new FormData()
  form.append('file', file)
  return parse(await fetch(`${API_BASE_URL}/api/documents/upload`, { method: 'POST', body: form }))
}

export async function askQuestion(question: string, topK: number): Promise<ChatResponse> {
  return parse(await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  }))
}

export async function semanticSearch(query: string, topK: number): Promise<SearchResponse> {
  const params = new URLSearchParams({ q: query, top_k: String(topK) })
  return parse(await fetch(`${API_BASE_URL}/api/search?${params}`))
}

export async function evaluateRetrieval(cases: EvaluationCase[], topK: number): Promise<EvaluationResponse> {
  return parse(await fetch(`${API_BASE_URL}/api/evaluation/retrieval`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cases, top_k: topK }),
  }))
}
