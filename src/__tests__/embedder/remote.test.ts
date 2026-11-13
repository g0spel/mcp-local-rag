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
import { privateMembers } from '../../__tests__/test-doubles.js'
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

  it('leaves zero open sockets after timeout-aborted requests against a black-hole server', async () => {
    // Regression for the chronic leak observed in production (22 idle
    // ESTABLISHED sockets accumulated over 11 days on 0.19.2's dedicated
    // transport, wedging the ssh tunnel's channel handling): a request that
    // hangs must not leave its socket open after the abort teardown.
    const sockets = new Set<import('node:net').Socket>()
    const hole = createServer((_req, res) => {
      // black hole: accept and read the request, never respond
      if (res.socket) {
        sockets.add(res.socket)
      }
    })
    hole.on('connection', (s) => sockets.add(s))
    liveServers.push(hole)
    await new Promise<void>((resolve) => hole.listen(0, '127.0.0.1', resolve))
    const addr = hole.address()
    if (addr === null || typeof addr === 'string') {
      throw new Error('hole server did not report a port')
    }
    const e = new RemoteEmbedder({
      serverUrl: `http://127.0.0.1:${addr.port}`,
      batchSize: 2,
      timeoutMs: 700, // fast abort for the test
    })
    // One embed → 5 attempts (1 + 4 retries), each hangs and aborts.
    await expect(e.embed('hang')).rejects.toThrow()
    // Give teardown a moment, then require every socket the server saw to be
    // closed from the client side (FIN received → destroyed or ended).
    await new Promise((r) => setTimeout(r, 400))
    const stillOpen = [...sockets].filter((s) => !s.destroyed && s.readyState !== 'closed')
    expect(stillOpen.length).toBe(0)
  }, 45_000)

  it('caps every attempt at 8s even when timeoutMs is configured larger', async () => {
    // Wedge #4: requests hung 5-8s but eventually 200'd, so no failure tripped
    // anything. The 8s hard ceiling aborts those attempts (retryable), and the
    // abort's RST frees the request's slot in the tunnel queue.
    const hole = createServer((_req, res) => {
      // black hole: read the request, never respond
      res.socket?.setKeepAlive?.(true)
    })
    liveServers.push(hole)
    await new Promise<void>((resolve) => hole.listen(0, '127.0.0.1', resolve))
    const addr = hole.address()
    if (addr === null || typeof addr === 'string') {
      throw new Error('hole server did not report a port')
    }
    const e = new RemoteEmbedder({
      serverUrl: `http://127.0.0.1:${addr.port}`,
      batchSize: 2,
      timeoutMs: 25_000, // far above the cap — the cap must win
    })
    const attempt = privateMembers<{
      postEmbedAttempt: (inputs: string[]) => Promise<number[][]>
    }>(e).postEmbedAttempt.bind(e)
    const t0 = Date.now()
    await expect(attempt(['x'])).rejects.toThrow()
    const elapsed = Date.now() - t0
    expect(elapsed).toBeGreaterThanOrEqual(7_000)
    expect(elapsed).toBeLessThan(11_000)
  }, 20_000)

  it('opens a circuit breaker during failure storms instead of hammering the server', async () => {
    // Wedge root cause (2026-11-12): during an outage, retry+auto-sync loops
    // kept creating hanging connections that re-wedged the freshly restarted
    // tunnel. The embedder must trip a breaker after consecutive failures and
    // stop issuing requests during the cool-down.
    let attempts = 0
    let isOutage: () => boolean = () => false
    const stub = await startStub((_req, respond) => {
      attempts++
      if (isOutage()) {
        respond(500, 'boom')
        return
      }
      respond(200, DIM_2)
    })
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 2, timeoutMs: 1_000 })
    await e.initialize() // healthy probe first
    stub.reset()
    isOutage = () => true
    attempts = 0
    await expect(e.embed('x')).rejects.toThrow() // full retry chain fails
    expect(attempts).toBe(5) // 1 + 4 retries → breaker trips
    // Immediately retrying must NOT hit the server (breaker open).
    await expect(e.embed('y')).rejects.toThrow(/circuit|cool/i)
    expect(attempts).toBe(5) // no new server hits
    // Known gap: the half-open recovery path (cooldown expiry → probe
    // success → breaker reset) is not asserted here — the 60s cooldown is
    // hardcoded and not worth parameterizing for the test.
  }, 60_000)

  it('caps concurrent upstream requests at 4 (and at 1 once degraded)', async () => {
    // Wedge #4: 16 concurrent MCP callers stacked onto the tunnel. The gate is
    // the single chokepoint, so every caller's own await provides backpressure.
    let concurrent = 0
    let maxConcurrent = 0
    let holdMs = 120
    const stub = await startStub((_req, respond) => {
      concurrent++
      maxConcurrent = Math.max(maxConcurrent, concurrent)
      setTimeout(() => {
        concurrent--
        respond(200, JSON.stringify([[0.1, 0.2]]))
      }, holdMs)
    })
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 1, timeoutMs: 8_000 })
    await e.initialize()
    maxConcurrent = 0
    // Normal state: 8 concurrent embeds, never more than 4 upstream at once.
    await Promise.all(Array.from({ length: 8 }, (_, i) => e.embed(`a${i}`)))
    expect(maxConcurrent).toBeLessThanOrEqual(4)
    // Degraded state: the window averages the last 8 attempts, so saturate it
    // with slow samples (8 × ~2.8s) before asserting the cap collapsed to 1.
    holdMs = 2_800
    for (let i = 0; i < 8; i++) {
      await e.embed(`slow-${i}`)
    }
    maxConcurrent = 0
    stub.reset()
    await Promise.all([e.embed('d1'), e.embed('d2'), e.embed('d3')])
    expect(maxConcurrent).toBe(1)
  }, 120_000)

  it('drains excess concurrency when the limit shrinks to 1 (degraded)', async () => {
    // The naive handover kept inFlight pinned at 4 after a 4→1 shrink, which
    // would have made the degraded cap a no-op (2026-11-13 review).
    let concurrent = 0
    let maxAfterFlip = 0
    // 1.2s keeps the latency window in the "hold" band (1-2.5s): fast samples
    // would flip degraded back off mid-test and re-admit the queue.
    const stub = await startStub((_req, respond) => {
      concurrent++
      setTimeout(() => {
        concurrent--
        respond(200, JSON.stringify([[0.1, 0.2]]))
      }, 1_200)
    })
    const e = new RemoteEmbedder({ serverUrl: stub.url, batchSize: 1, timeoutMs: 8_000 })
    await e.initialize()
    const priv = privateMembers<{ degraded: boolean; inFlight: number }>(e)
    // Fill the normal 4-slot gate, then shrink the limit mid-flight.
    const firstWave = Array.from({ length: 4 }, (_, i) => e.embed(`w${i}`))
    await new Promise((r) => setTimeout(r, 50))
    priv.degraded = true
    const secondWave = Array.from({ length: 3 }, (_, i) => e.embed(`s${i}`))
    // The already-running wave is not evicted; only AFTER it drains must new
    // admissions respect the shrunk limit of 1.
    await Promise.all(firstWave)
    maxAfterFlip = 0
    const watcher = setInterval(() => {
      maxAfterFlip = Math.max(maxAfterFlip, concurrent)
    }, 20)
    await Promise.all(secondWave)
    clearInterval(watcher)
    expect(maxAfterFlip).toBeLessThanOrEqual(1)
    expect(priv.inFlight).toBe(0)
  }, 60_000)

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
