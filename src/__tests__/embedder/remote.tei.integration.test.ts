// TEI live-endpoint integration tests — this fork's primary embedding mode
// (RAG_EMBEDDING_SERVER_URL). Skipped unless a TEI-compatible server answers
// GET /info, so environments without the deployment keep a green suite.

import { beforeAll, describe, expect, it } from 'vitest'
import { SemanticChunker } from '../../chunker/semantic-chunker.js'
import { estimateTokens, RemoteEmbedder } from '../../embedder/remote.js'

const SERVER_URL = process.env['RAG_EMBEDDING_SERVER_URL'] ?? 'http://127.0.0.1:18081'

async function teiInfo(): Promise<{ max_input_length?: unknown } | null> {
  try {
    const res = await fetch(`${SERVER_URL}/info`, { signal: AbortSignal.timeout(3000) })
    if (!res.ok) {
      return null
    }
    const parsed: unknown = await res.json()
    if (typeof parsed !== 'object' || parsed === null) {
      return null
    }
    return 'max_input_length' in parsed ? { max_input_length: parsed.max_input_length } : {}
  } catch {
    return null
  }
}

const info = await teiInfo()

describe.skipIf(info === null)('RemoteEmbedder against a live TEI endpoint', () => {
  let embedder: RemoteEmbedder

  beforeAll(async () => {
    embedder = new RemoteEmbedder({ serverUrl: SERVER_URL, batchSize: 16, timeoutMs: 30_000 })
    await embedder.initialize()
  }, 30_000)

  it('probes the dimension and the server input cap from /info', () => {
    expect(embedder.tokenLimit).toBe(info === null ? null : (info.max_input_length ?? null))
    expect(embedder.tokenLimit).not.toBeNull()
    expect(embedder.tokenLimit ?? 0).toBeGreaterThan(512)
  })

  it('embeds CJK and ASCII sentences with a stable dimension', async () => {
    const cjk = await embedder.embed('语义检索回归测试')
    const ascii = await embedder.embed('another sentence')
    expect(cjk.length).toBeGreaterThan(0)
    expect(ascii.length).toBe(cjk.length)
  })

  it('embedBatch preserves order across batches', async () => {
    const vectors = await embedder.embedBatch(['一', '二', '三', '四', '五'])
    expect(vectors).toHaveLength(5)
    expect(new Set(vectors.map((v) => v.length)).size).toBe(1)
  })

  it('containment keeps oversized ledger-like content under the server cap', async () => {
    // Digit/CJK-dense slab with no sentence boundaries: without containment
    // the semantic chunker emits one oversized unit and a strictly validating
    // TEI rejects it with HTTP 413 (the PENDING_EXPERIMENTS.md failure mode).
    const slab = '| e4327 | K178 | 121.5 | 0.19 | 通过 | 复利口径与诚实尺对齐 '.repeat(400)
    const cap = await embedder.getTokenLimit()
    expect(cap).not.toBeNull()
    expect(estimateTokens(slab)).toBeGreaterThan(cap ?? 0) // the case is real

    const chunks = await new SemanticChunker().chunkText(slab, embedder)
    expect(chunks.length).toBeGreaterThan(1)
    const lengths = await embedder.countTokens(chunks.map((chunk) => chunk.text))
    for (const length of lengths) {
      expect(length).toBeLessThanOrEqual(cap ?? 0)
    }
    // chunkText already embedded every unit against the live server — reaching
    // here means no HTTP 413 escaped containment.
  }, 120_000)
})
