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
    expect(e.tokenLimit).toBeNull()
    await e.embed('hello')
    // One probe + one embed = two POSTs total.
    expect(fetchMock).toHaveBeenCalledTimes(2)
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
    expect(fetchMock).toHaveBeenCalledTimes(4) // 1 initial + 3 retries
  }, 15_000)

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
    expect(fetchMock).toHaveBeenCalledTimes(2)
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
