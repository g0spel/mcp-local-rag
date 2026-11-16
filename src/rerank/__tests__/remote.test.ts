// HTTP reranker: the endpoint transport. A real loopback server stands in for
// TEI so the request body, timeout, retry and validation are exercised as they
// run in production rather than through a mocked HTTP layer.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { buildRerankRequestBody, rerankCandidatesViaUrl } from '../remote.js'
import type { RerankResult } from '../response.js'

function candidate(index: number): RerankResult {
  return {
    filePath: `/docs/${index}.md`,
    chunkIndex: index,
    text: `text ${index}`,
    score: 0.5,
    fileTitle: null,
    images: [],
  }
}

interface Recorded {
  body: unknown
  hits: { index: number; score: number }[]
}

const servers: Server[] = []

async function startServer(
  handler: (body: unknown, res: ServerResponse, attempt: number) => void
): Promise<{ url: string; recorded: Recorded[]; attempts: () => number }> {
  const recorded: Recorded[] = []
  let attempts = 0
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    attempts += 1
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => chunks.push(chunk))
    req.on('end', () => {
      const body: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'))
      recorded.push({ body, hits: [] })
      handler(body, res, attempts)
    })
  })
  servers.push(server)
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') {
    throw new Error('expected a bound TCP address')
  }
  return { url: `http://127.0.0.1:${address.port}`, recorded, attempts: () => attempts }
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(payload))
}

afterEach(() => {
  vi.restoreAllMocks()
  for (const server of servers.splice(0)) {
    server.close()
  }
})

describe('rerankCandidatesViaUrl', () => {
  it('sends the query in the request body, not in an argument', async () => {
    const { url, recorded } = await startServer((_body, res) => {
      sendJson(res, 200, [{ index: 0, score: 1 }])
    })

    await rerankCandidatesViaUrl({ baseUrl: url, timeoutMs: 2000 }, [candidate(0)], 'needle', 1)

    expect(recorded[0]?.body).toEqual({
      query: 'needle',
      texts: ['text 0'],
      raw_scores: true,
    })
  })

  it('orders by score, truncates to top and keeps the original objects', async () => {
    const { url } = await startServer((_body, res) => {
      sendJson(res, 200, [
        { index: 0, score: -8.5 },
        { index: 1, score: -0.9 },
        { index: 2, score: -4.0 },
      ])
    })

    const out = await rerankCandidatesViaUrl(
      { baseUrl: url, timeoutMs: 2000 },
      [candidate(0), candidate(1), candidate(2)],
      'q',
      2
    )

    expect(out.map((r) => r.chunkIndex)).toEqual([1, 2])
    expect(out[0]?.filePath).toBe('/docs/1.md')
    expect(out[0]?.score).toBe(-0.9)
  })

  it('returns the pre-rerank ordering on a non-retryable HTTP error', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { url, attempts } = await startServer((_body, res) => {
      sendJson(res, 400, { error: 'bad request' })
    })
    const candidates = [candidate(0), candidate(1)]

    const out = await rerankCandidatesViaUrl({ baseUrl: url, timeoutMs: 2000 }, candidates, 'q', 2)

    expect(out).toEqual(candidates)
    expect(attempts()).toBe(1)
  })

  it('retries 5xx and degrades to the original ordering once exhausted', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { url, attempts } = await startServer((_body, res) => {
      sendJson(res, 503, { error: 'unavailable' })
    })
    const candidates = [candidate(0)]

    const out = await rerankCandidatesViaUrl({ baseUrl: url, timeoutMs: 2000 }, candidates, 'q', 1)

    expect(out).toEqual(candidates)
    expect(attempts()).toBe(3)
  }, 15000)

  it('degrades when the endpoint does not answer within the budget', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { url, attempts } = await startServer(() => {
      // never respond
    })
    const candidates = [candidate(0)]

    const out = await rerankCandidatesViaUrl({ baseUrl: url, timeoutMs: 150 }, candidates, 'q', 1)

    expect(out).toEqual(candidates)
    expect(attempts()).toBe(3)
  }, 15000)

  it('rejects a duplicated or out-of-range index instead of remapping candidates', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    for (const hits of [
      [
        { index: 0, score: 1 },
        { index: 0, score: 2 },
      ],
      [{ index: 5, score: 1 }],
    ]) {
      const { url } = await startServer((_body, res) => sendJson(res, 200, hits))
      const candidates = [candidate(0), candidate(1)]
      const out = await rerankCandidatesViaUrl(
        { baseUrl: url, timeoutMs: 2000 },
        candidates,
        'q',
        2
      )
      expect(out).toEqual(candidates)
    }
  })

  it('sends nothing for an empty candidate list', async () => {
    const { url, attempts } = await startServer((_body, res) => sendJson(res, 200, []))
    const out = await rerankCandidatesViaUrl({ baseUrl: url, timeoutMs: 2000 }, [], 'q', 3)
    expect(out).toEqual([])
    expect(attempts()).toBe(0)
  })
})

describe('buildRerankRequestBody', () => {
  it('carries the query as a field, so an empty argv cannot drop it', () => {
    const body: unknown = JSON.parse(buildRerankRequestBody('为什么要统一', [candidate(0)]))
    expect(body).toEqual({ query: '为什么要统一', texts: ['text 0'], raw_scores: true })
  })
})
