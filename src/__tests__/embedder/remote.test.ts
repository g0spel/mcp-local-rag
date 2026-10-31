// RemoteEmbedder unit tests
//
// The HTTP boundary is stubbed with a real loopback `node:http` server whose
// handler is swapped per test (the transport under test uses `node:http`
// directly, so a `vi.stubGlobal('fetch')` mock would never be exercised).
//
// Pattern: a handler that always answers dim-2 vectors first satisfies the
// initialize() probe, so each test can then install its own failure/success
// script for the behavior under test without the probe consuming its call
// budget.

import { createServer, type Server } from 'node:http'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createEmbedder } from '../../cli/common.js'
import {
  estimateTokens,
  RemoteEmbedder,
  RemoteEmbeddingError,
  RemoteEmbeddingRetryableError,
} from '../../embedder/remote.js'

const DIM_2 = JSON.stringify([[0.1, 0.2]])
const DIM_3 = JSON.stringify([[0.1, 0.2, 0.3]])

interface StubRequest {
  method: string
  url: string
  body: string
}

type Handler = (
  req: StubRequest,
  respond: (status: number, body?: string) => void,
  destroySocket: () => void
) => void

interface Stub {
  url: string
  requests: StubRequest[]
  reset: () => void
  onSocketClose: (cb: () => void) => void
  setHandler: (handler: Handler) => void
  close: () => Promise<void>
}

const liveServers: Server[] = []

/** Start a loopback stub server; every request is recorded and routed through the (swappable) handler. */
async function startStub(initial: Handler): Promise<Stub> {
  const requests: StubRequest[] = []
  let handler: Handler = initial
  let closeCb: (() => void) | null = null
  const server = createServer((req, res) => {
    const chunks: Buffer[] = []
    req.on('data', (c: Buffer) => chunks.push(c))
    req.on('end', () => {
      const stubReq: StubRequest = {
        method: req.method ?? '',
        url: req.url ?? '',
        body: Buffer.concat(chunks).toString('utf8'),
      }
      requests.push(stubReq)
      handler(
        stubReq,
        (status, body = '') => {
          res.writeHead(status, { 'Content-Type': 'application/json' })
          res.end(body)
        },
        () => {
          res.socket?.destroy()
        }
      )
    })
  })
  server.on('connection', (socket) => {
    socket.on('close', () => closeCb?.())
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const addr = server.address()
  if (addr === null || typeof addr === 'string') {
    throw new Error('stub server did not report a port')
  }
  const url = `http://127.0.0.1:${addr.port}`
  return {
    url,
    requests,
    reset: () => {
      requests.length = 0
    },
    onSocketClose: (cb) => {
      closeCb = cb
    },
    setHandler: (h) => {
      handler = h
    },
    close: () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve())
        server.closeAllConnections()
      }),
  }
}

const okDim2: Handler = (_req, respond) => {
  respond(200, DIM_2)
}

/** A stub + embedder pair where initialize() has already succeeded on dim-2 vectors. */
async function makeInitializedEmbedder(): Promise<{ e: RemoteEmbedder; stub: Stub }> {
  const stub = await startStub(okDim2)
  const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
  await e.initialize()
  stub.reset()
  return { e, stub }
}

afterEach(async () => {
  await Promise.all(
    liveServers.splice(0).map(
      (s) =>
        new Promise<void>((r) => {
          s.close(() => r())
          s.closeAllConnections()
        })
    )
  )
  vi.unstubAllEnvs()
})

describe('RemoteEmbedder', () => {
  it('initialize probes the dimension once; later embeds reuse it', async () => {
    const stub = await startStub(okDim2)
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
    await e.initialize()
    // The stub answers every route with embed-shaped JSON, so /info yields no
    // max_input_length and the cap stays unknown.
    expect(e.tokenLimit).toBeNull()
    await e.embed('hello')
    // Embed probe + /info probe + one embed = three requests total.
    expect(stub.requests.length).toBe(3)
  })

  it('embedBatch slices by batchSize and preserves order', async () => {
    const bodies: string[][] = []
    const stub = await startStub((req, respond) => {
      const body: { inputs?: string[] } = req.body === '' ? {} : JSON.parse(req.body)
      const inputs = body.inputs ?? []
      bodies.push(inputs)
      respond(200, JSON.stringify(inputs.map((t) => [t.length, 1])))
    })
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
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
    const { e, stub } = await makeInitializedEmbedder()
    let remainingFailures = 2
    stub.setHandler((_req, respond, destroySocket) => {
      if (remainingFailures > 0) {
        remainingFailures--
        destroySocket() // simulate a dropped connection
        return
      }
      respond(200, JSON.stringify([[0.5, 0.6]]))
    })
    const v = await e.embed('hello')
    expect(v).toEqual([0.5, 0.6])
    expect(stub.requests.length).toBe(3)
  }, 15_000)

  it('retries 429/5xx and throws RemoteEmbeddingRetryableError after exhaustion', async () => {
    const { e, stub } = await makeInitializedEmbedder()
    stub.setHandler((_req, respond) => {
      respond(500, 'boom')
    })
    await expect(e.embed('hello')).rejects.toThrow(RemoteEmbeddingRetryableError)
    expect(stub.requests.length).toBe(5) // 1 initial + 4 retries
  }, 40_000) // 重试序列（2s 起步 ×4 次 max 20s + jitter）+ stub 开销，15s 已不够

  it('does NOT retry client errors (401)', async () => {
    const { e, stub } = await makeInitializedEmbedder()
    stub.setHandler((_req, respond) => {
      respond(401, 'unauthorized')
    })
    await expect(e.embed('hello')).rejects.toThrow(RemoteEmbeddingError)
    expect(stub.requests.length).toBe(1)
  })

  it('fails fast when the response dimension drifts', async () => {
    const { e, stub } = await makeInitializedEmbedder()
    stub.setHandler((_req, respond) => {
      respond(200, DIM_3)
    })
    await expect(e.embed('hello')).rejects.toThrow(/dimension drifted/)
  })

  it('throws when the vector count does not match the input count', async () => {
    const { e, stub } = await makeInitializedEmbedder()
    stub.setHandler((_req, respond) => {
      respond(200, JSON.stringify([[0.1]]))
    })
    await expect(e.embedBatch(['a', 'b'])).rejects.toThrow(/1 vectors for 2 inputs/)
  })

  it('dispose clears initialization; next call re-probes', async () => {
    const stub = await startStub(okDim2)
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
    await e.initialize()
    await e.dispose()
    await e.initialize()
    // Each initialize: embed probe + /info probe.
    expect(stub.requests.length).toBe(4)
  })

  it('probes max_input_length from /info and exposes it via tokenLimit', async () => {
    const stub = await startStub((req, respond) => {
      respond(200, req.url.endsWith('/info') ? JSON.stringify({ max_input_length: 512 }) : DIM_2)
    })
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
    expect(e.tokenLimit).toBeNull() // before initialize
    await e.initialize()
    expect(e.tokenLimit).toBe(512)
    await expect(e.getTokenLimit()).resolves.toBe(512)
  })

  it('keeps tokenLimit null when /info omits max_input_length', async () => {
    const stub = await startStub((req, respond) => {
      respond(200, req.url.endsWith('/info') ? JSON.stringify({ model_id: 'x' }) : DIM_2)
    })
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
    await e.initialize()
    expect(e.tokenLimit).toBeNull()
  })

  it('keeps tokenLimit null when /info fails; initialize still succeeds', async () => {
    const stub = await startStub((req, respond) => {
      if (req.url.endsWith('/info')) {
        respond(500)
        return
      }
      respond(200, DIM_2)
    })
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
    await e.initialize()
    expect(e.tokenLimit).toBeNull()
    await expect(e.embed('hello')).resolves.toEqual([0.1, 0.2])
  })

  it('closes the server-side connection after each response (no pooled residue)', async () => {
    const stub = await startStub(okDim2)
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
    await e.initialize()
    let closed = 0
    stub.onSocketClose(() => {
      closed++
    })
    await e.embed('hello')
    await e.embed('world')
    // initialize probes (2) + embeds (2); closed counts only sockets closed
    // after the hook was installed — the two embed sockets, one each.
    expect(stub.requests.length).toBe(4)
    expect(closed).toBe(2)
  })

  it('countTokens returns conservative estimates without server round trips', async () => {
    const { e, stub } = await makeInitializedEmbedder()
    const [cjk, ascii] = await e.countTokens(['一二三四五', 'abcde'])
    expect(cjk).toBe(7) // 5 × 1.4 = 7
    expect(ascii).toBe(4) // 5 × 0.7 = 3.5 → 4
    expect(stub.requests.length).toBe(0) // no requests beyond the (reset) initialize probes
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
