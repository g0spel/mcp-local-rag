// RemoteEmbedder unit tests
//
// The fetch boundary is stubbed with `vi.stubGlobal` + `vi.unstubAllGlobals`
// per test (no transformers.js involvement — this class never imports it, so
// the shared-registry mock-isolation rule does not apply here).
//
// Pattern: `makeHealthyServer()` first satisfies the initialize() probe, so
// each test can then install its own failure/success script for the behavior
// under test without the probe consuming its call budget.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { createEmbedder } from '../../cli/common.js'
import {
  estimateTokens,
  RemoteEmbedder,
  RemoteEmbeddingError,
  RemoteEmbeddingRetryableError,
} from '../../embedder/remote.js'

const CONFIG = { serverUrl: 'http://test:8081', batchSize: 2, timeoutMs: 1_000 }
const DIM_2 = [[0.1, 0.2]]
const DIM_3 = [[0.1, 0.2, 0.3]]

function okEmbed(vectors: number[][]): Response {
  return new Response(JSON.stringify(vectors), { status: 200 })
}

function statusResponse(status: number, body = ''): Response {
  return new Response(body, { status })
}

/** A server that always answers with dim-2 vectors. */
function makeHealthyServer() {
  return vi.fn(async () => okEmbed(DIM_2))
}

async function makeInitializedEmbedder(): Promise<RemoteEmbedder> {
  vi.stubGlobal('fetch', makeHealthyServer())
  const e = new RemoteEmbedder(CONFIG)
  await e.initialize()
  return e
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('RemoteEmbedder', () => {
  it('initialize probes the dimension once; later embeds reuse it', async () => {
    const fetchMock = makeHealthyServer()
    vi.stubGlobal('fetch', fetchMock)
    const e = new RemoteEmbedder(CONFIG)
    await e.initialize()
    // The mock answers every route with embed-shaped JSON, so /info yields no
    // max_input_length and the cap stays unknown.
    expect(e.tokenLimit).toBeNull()
    await e.embed('hello')
    // Embed probe + /info probe + one embed = three requests total.
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('embedBatch slices by batchSize and preserves order', async () => {
    const bodies: string[][] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
        const body: { inputs?: string[] } = JSON.parse(String(init?.body))
        bodies.push(body.inputs ?? [])
        return okEmbed((body.inputs ?? []).map((t) => [t.length, 1]))
      })
    )
    const e = new RemoteEmbedder(CONFIG)
    await e.initialize()
    bodies.length = 0 // discard the probe
    const vectors = await e.embedBatch(['a', 'b', 'c'])
    expect(vectors).toEqual([
      [1, 1],
      [1, 1],
      [1, 1],
    ])
    expect(bodies).toEqual([['a', 'b'], ['c']])
  })

  it('retries network failures with backoff and eventually succeeds', async () => {
    const e = await makeInitializedEmbedder()
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('network down'))
      .mockRejectedValueOnce(new TypeError('network down'))
      .mockResolvedValue(okEmbed([[0.5, 0.6]]))
    vi.stubGlobal('fetch', fetchMock)
    const v = await e.embed('hello')
    expect(v).toEqual([0.5, 0.6])
    expect(fetchMock).toHaveBeenCalledTimes(3)
  }, 15_000)

  it('retries 429/5xx and throws RemoteEmbeddingRetryableError after exhaustion', async () => {
    const e = await makeInitializedEmbedder()
    const fetchMock = vi.fn(async () => statusResponse(500, 'boom'))
    vi.stubGlobal('fetch', fetchMock)
    await expect(e.embed('hello')).rejects.toThrow(RemoteEmbeddingRetryableError)
    expect(fetchMock).toHaveBeenCalledTimes(5) // 1 initial + 4 retries
  }, 40_000) // 重试序列（2s 起步 ×4 次 max 20s + jitter）+ mock 开销，15s 已不够

  it('does NOT retry client errors (401)', async () => {
    const e = await makeInitializedEmbedder()
    const fetchMock = vi.fn(async () => statusResponse(401, 'unauthorized'))
    vi.stubGlobal('fetch', fetchMock)
    await expect(e.embed('hello')).rejects.toThrow(RemoteEmbeddingError)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('fails fast when the response dimension drifts', async () => {
    const e = await makeInitializedEmbedder()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => okEmbed(DIM_3))
    )
    await expect(e.embed('hello')).rejects.toThrow(/dimension drifted/)
  })

  it('throws when the vector count does not match the input count', async () => {
    const e = await makeInitializedEmbedder()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => okEmbed([[0.1]]))
    )
    await expect(e.embedBatch(['a', 'b'])).rejects.toThrow(/1 vectors for 2 inputs/)
  })

  it('dispose clears initialization; next call re-probes', async () => {
    const fetchMock = makeHealthyServer()
    vi.stubGlobal('fetch', fetchMock)
    const e = new RemoteEmbedder(CONFIG)
    await e.initialize()
    await e.dispose()
    await e.initialize()
    // Each initialize: embed probe + /info probe.
    expect(fetchMock).toHaveBeenCalledTimes(4)
  })

  it('probes max_input_length from /info and exposes it via tokenLimit', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string | URL | Request) =>
        String(url).endsWith('/info')
          ? new Response(JSON.stringify({ max_input_length: 512 }), { status: 200 })
          : okEmbed(DIM_2)
      )
    )
    const e = new RemoteEmbedder(CONFIG)
    expect(e.tokenLimit).toBeNull() // before initialize
    await e.initialize()
    expect(e.tokenLimit).toBe(512)
    await expect(e.getTokenLimit()).resolves.toBe(512)
  })

  it('keeps tokenLimit null when /info omits max_input_length', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string | URL | Request) =>
        String(url).endsWith('/info')
          ? new Response(JSON.stringify({ model_id: 'x' }), { status: 200 })
          : okEmbed(DIM_2)
      )
    )
    const e = new RemoteEmbedder(CONFIG)
    await e.initialize()
    expect(e.tokenLimit).toBeNull()
  })

  it('keeps tokenLimit null when /info fails; initialize still succeeds', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string | URL | Request) => {
        if (String(url).endsWith('/info')) {
          return statusResponse(500)
        }
        return okEmbed(DIM_2)
      })
    )
    const e = new RemoteEmbedder(CONFIG)
    await e.initialize()
    expect(e.tokenLimit).toBeNull()
    await expect(e.embed('hello')).resolves.toEqual([0.1, 0.2])
  })

  it('countTokens returns conservative estimates without server round trips', async () => {
    const e = await makeInitializedEmbedder()
    const [cjk, ascii] = await e.countTokens(['一二三四五', 'abcde'])
    expect(cjk).toBe(7) // 5 × 1.4 = 7
    expect(ascii).toBe(4) // 5 × 0.7 = 3.5 → 4
  })
})

describe('estimateTokens', () => {
  it('counts surrogate pairs as one code point', () => {
    expect(estimateTokens('😀')).toBe(1)
  })

  it('applies the dense-script coefficient to CJK text', () => {
    expect(estimateTokens('汉字文本')).toBe(6) // 4 × 1.4 = 5.6 → 6
  })

  it('applies the other coefficient to digit-heavy table rows', () => {
    expect(estimateTokens('1234 | 5678')).toBe(8) // 11 × 0.7 = 7.7 → 8
  })
})

describe('factory selection (createEmbedder)', () => {
  const minimalConfig = {
    dbPath: '/tmp/x.lance',
    cacheDir: '/tmp/cache',
    modelName: 'BAAI/bge-m3',
  }

  it('returns RemoteEmbedder when RAG_EMBEDDING_SERVER_URL is set', async () => {
    vi.stubEnv('RAG_EMBEDDING_SERVER_URL', 'http://gpu-host:8081')
    const e = createEmbedder(minimalConfig)
    expect(e.constructor.name).toBe('RemoteEmbedder')
  })

  it('returns local Embedder when the env is unset', async () => {
    vi.stubEnv('RAG_EMBEDDING_SERVER_URL', '')
    const e = createEmbedder(minimalConfig)
    expect(e.constructor.name).toBe('Embedder')
  })
})
