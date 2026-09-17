import { FormEvent, useState } from 'react'
import { askQuestion, evaluateRetrieval, EvaluationResponse, semanticSearch, SearchResponse, Source, uploadDocument } from './api'

type Tab = 'chat' | 'retrieval' | 'evaluation'

const starterEvaluation = JSON.stringify([
  { question: 'What should be inspected before restarting the machine?', expected_source: 'maintenance_manual.pdf', expected_page: 14 },
  { question: 'What PPE is required for this maintenance procedure?', expected_source: 'safety_procedure.pdf', expected_page: 3 }
], null, 2)

function Sources({ sources }: { sources: Source[] }) {
  if (!sources.length) return null
  return <div className="sources">{sources.map((source, i) => (
    <article className="source" key={`${source.source}-${source.page}-${source.chunk_index}`}>
      <strong>#{i + 1} {source.source}</strong>
      <span>Page {source.page} · Chunk {source.chunk_index}</span>
      <span>Similarity {(source.similarity * 100).toFixed(1)}%</span>
    </article>
  ))}</div>
}

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')
  const [topK, setTopK] = useState(5)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [grounded, setGrounded] = useState<boolean | null>(null)
  const [sources, setSources] = useState<Source[]>([])
  const [search, setSearch] = useState<SearchResponse | null>(null)
  const [evaluationText, setEvaluationText] = useState(starterEvaluation)
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null)
  const [status, setStatus] = useState('Ready')
  const [busy, setBusy] = useState(false)

  async function run<T>(label: string, action: () => Promise<T>, done: (value: T) => void) {
    setBusy(true); setStatus(label)
    try { done(await action()); setStatus('Done') }
    catch (error) { setStatus(error instanceof Error ? error.message : 'Request failed') }
    finally { setBusy(false) }
  }

  async function submitChat(event: FormEvent) {
    event.preventDefault()
    if (!question.trim()) return
    await run('Retrieving evidence and generating…', () => askQuestion(question, topK), data => {
      setAnswer(data.answer); setGrounded(data.grounded); setSources(data.sources)
    })
  }

  async function submitSearch(event: FormEvent) {
    event.preventDefault()
    if (!question.trim()) return
    await run('Searching vectors…', () => semanticSearch(question, topK), setSearch)
  }

  async function submitEvaluation() {
    try {
      const cases = JSON.parse(evaluationText)
      await run('Evaluating retrieval…', () => evaluateRetrieval(cases, topK), setEvaluation)
    } catch { setStatus('Evaluation JSON is invalid') }
  }

  return <main>
    <header className="hero">
      <div><p className="eyebrow">LOCAL-FIRST RAG · MANUFACTURING</p><h1>Manufacturing Knowledge Assistant</h1><p>Ask operational questions, inspect retrieved evidence, and measure retrieval quality.</p></div>
      <div className="status"><span className={busy ? 'pulse' : ''} />{status}</div>
    </header>

    <section className="upload panel">
      <div><h2>Knowledge base</h2><p>Index PDF, TXT or Markdown documentation.</p></div>
      <label className="uploadButton">Upload document<input type="file" accept=".pdf,.txt,.md" disabled={busy} onChange={event => {
        const file = event.target.files?.[0]
        if (file) run(`Indexing ${file.name}…`, () => uploadDocument(file), () => undefined)
      }} /></label>
    </section>

    <nav>{(['chat', 'retrieval', 'evaluation'] as Tab[]).map(item => <button className={tab === item ? 'active' : ''} onClick={() => setTab(item)} key={item}>{item}</button>)}</nav>

    <section className="workspace panel">
      <div className="controls"><label>Top-K <strong>{topK}</strong><input type="range" min="1" max="10" value={topK} onChange={e => setTopK(Number(e.target.value))} /></label></div>

      {tab !== 'evaluation' && <form onSubmit={tab === 'chat' ? submitChat : submitSearch} className="questionForm">
        <textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="e.g. What must be inspected before restarting the machine?" />
        <button disabled={busy || !question.trim()}>{tab === 'chat' ? 'Ask assistant' : 'Inspect retrieval'}</button>
      </form>}

      {tab === 'chat' && answer && <div className="result"><div className="resultTitle"><h2>Answer</h2><span className={grounded ? 'badge good' : 'badge'}>{grounded ? 'Grounded' : 'Insufficient evidence'}</span></div><p className="answer">{answer}</p><Sources sources={sources} /></div>}

      {tab === 'retrieval' && search && <div className="result"><h2>Retrieved chunks</h2>{search.results.map((item, i) => <article className="chunk" key={`${item.source}-${item.chunk_index}`}><div><strong>#{i + 1} {item.source} · p.{item.page}</strong><span>{(item.similarity * 100).toFixed(1)}%</span></div><p>{item.text}</p></article>)}</div>}

      {tab === 'evaluation' && <div className="evaluation"><p>Paste labeled questions to measure Hit Rate@K and Mean Reciprocal Rank.</p><textarea value={evaluationText} onChange={e => setEvaluationText(e.target.value)} /><button disabled={busy} onClick={submitEvaluation}>Run evaluation</button>{evaluation && <><div className="metrics"><div><span>Hit Rate@{topK}</span><strong>{(evaluation.hit_rate * 100).toFixed(1)}%</strong></div><div><span>MRR</span><strong>{evaluation.mean_reciprocal_rank.toFixed(3)}</strong></div><div><span>Hits</span><strong>{evaluation.hits}/{evaluation.total}</strong></div></div><div className="cases">{evaluation.cases.map((item, i) => <article key={i}><span className={item.hit ? 'dot hit' : 'dot'} /> <strong>{item.question}</strong><small>{item.expected_source}{item.expected_page ? ` · p.${item.expected_page}` : ''} · {item.hit ? `rank ${item.rank}` : 'miss'}</small></article>)}</div></> }</div>}
    </section>
  </main>
}
