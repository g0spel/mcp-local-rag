// Remote embedding server (TEI-compatible) backend.
//
// Activated by RAG_EMBEDDING_SERVER_URL: when set, the factory returns this
// class instead of the local Transformers.js pipeline, so embedding compute
// runs on a self-hosted server (e.g. HuggingFace TEI) while document content
// never leaves the deployment network.
//
// Wire protocol: POST {serverUrl}/embed  {"inputs": string[]} → number[][]
// (TEI-native shape; the server owns model choice, dtype, and truncation).

import { AppError } from '../utils/errors.js'

/** Remote embedding generation error. */
export class RemoteEmbeddingError extends AppError {
  constructor(message: string, options?: { cause?: Error }) {
    super(message, 'embedder', 'internal', options)
    this.name = 'RemoteEmbeddingError'
  }
}

/** Server responded with a retryable failure (429 / 5xx). */
export class RemoteEmbeddingRetryableError extends RemoteEmbeddingError {
  readonly status: number

  constructor(status: number, body: string, serverUrl: string) {
    super(
      `HTTP ${status} from embedding server ${serverUrl}${body ? `: ${body.slice(0, 200)}` : ''}`
    )
    this.name = 'RemoteEmbeddingRetryableError'
    this.status = status
  }
}

export interface RemoteEmbedderConfig {
  /** Base URL without trailing slash, e.g. http://gpu-host:8081 */
  serverUrl: string
  /** Texts per POST (default 16, mirroring the local Embedder) */
  batchSize: number
  /** Per-attempt HTTP timeout in ms (default 30_000) */
  timeoutMs?: number
}

// ============================================
// Retry (exponential backoff + full jitter)
// ============================================

const MAX_RETRIES = 3
const INITIAL_BACKOFF_MS = 500
const MAX_BACKOFF_MS = 8000

function isRetryableError(e: unknown): boolean {
  // Client errors (401/403/404/422) never succeed on retry; network-level
  // failures and 429/5xx do.
  if (e instanceof RemoteEmbeddingRetryableError) {
    return true
  }
  return !(e instanceof RemoteEmbeddingError)
}

function isRetryableStatus(status: number): boolean {
  return status === 429 || status >= 500
}

function sleep(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>()
  setTimeout(resolve, ms)
  return promise
}

/** Exponential backoff with full jitter: 500ms ×2^attempt, capped at 8s. */
function backoffMs(attempt: number): number {
  const cap = Math.min(MAX_BACKOFF_MS, INITIAL_BACKOFF_MS * 2 ** attempt)
  return Math.floor(Math.random() * cap)
}

async function withRetry<T>(fn: () => Promise<T>, context: string): Promise<T> {
  let lastError: unknown
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn()
    } catch (e: unknown) {
      lastError = e
      if (attempt === MAX_RETRIES || !isRetryableError(e)) {
        throw e
      }
      const wait = backoffMs(attempt)
      console.error(
        `RemoteEmbedder: ${context} failed (attempt ${attempt + 1}/${MAX_RETRIES + 1}), retrying in ${wait}ms`
      )
      await sleep(wait)
    }
  }
  throw lastError
}

async function withTimeout<T>(
  run: (signal: AbortSignal) => Promise<T>,
  timeoutMs: number
): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await run(controller.signal)
  } finally {
    clearTimeout(timer)
  }
}

// ============================================
// RemoteEmbedder
// ============================================

/**
 * Structural counterpart of the local `Embedder`: same public surface
 * (`initialize / embed / embedBatch / dispose / tokenLimit`), with embedding
 * compute delegated to a self-hosted server.
 *
 * Semantics intentionally aligned with the local class:
 * - `tokenLimit` is always `null` — the remote server owns truncation, which
 *   maps to the local class's "degraded mode" (no cap, no clamp).
 * - `dispose()` is a no-op — no in-process model resources.
 * - `device` / `dtype` / `cacheDir` / `modelPath` are server-side concerns.
 */
export class RemoteEmbedder {
  private readonly config: RemoteEmbedderConfig
  private dims: number | null = null
  private initialized: boolean = false

  constructor(config: RemoteEmbedderConfig) {
    this.config = config
  }

  /**
   * Probe the server and record the embedding dimension so later responses
   * can be validated (a dimension change means the server swapped models —
   * fail fast instead of writing mixed-dimension vectors).
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return
    }
    const [vector] = await this.postEmbed(['dimension probe'])
    if (!Array.isArray(vector) || vector.length === 0) {
      throw new RemoteEmbeddingError(
        `Remote embedding server returned an empty vector: ${this.config.serverUrl}`
      )
    }
    this.dims = vector.length
    this.initialized = true
    console.error(
      `RemoteEmbedder: connected to "${this.config.serverUrl}" (dimension ${this.dims})`
    )
  }

  async dispose(): Promise<void> {
    this.initialized = false
    this.dims = null
  }

  /** Remote servers decide their own truncation: no local cap, no clamp. */
  get tokenLimit(): null {
    return null
  }

  async embed(text: string): Promise<number[]> {
    const vectors = await this.embedBatch([text])
    const vector = vectors[0]
    if (vector === undefined) {
      throw new RemoteEmbeddingError(
        `Remote embedding server returned no vector for a single input: ${this.config.serverUrl}`
      )
    }
    return vector
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    await this.initialize()
    const vectors: number[][] = []
    for (let i = 0; i < texts.length; i += this.config.batchSize) {
      const batch = texts.slice(i, i + this.config.batchSize)
      const out = await withRetry(() => this.postEmbed(batch), `embedBatch(${batch.length} texts)`)
      this.validate(out, batch.length)
      vectors.push(...out)
    }
    return vectors
  }

  private validate(vectors: number[][], expectedCount: number): void {
    if (!Array.isArray(vectors) || vectors.length !== expectedCount) {
      throw new RemoteEmbeddingError(
        `Remote embedding server returned ${Array.isArray(vectors) ? vectors.length : 'non-array'} ` +
          `vectors for ${expectedCount} inputs: ${this.config.serverUrl}`
      )
    }
    if (this.dims !== null) {
      for (const v of vectors) {
        if (!Array.isArray(v) || v.length !== this.dims) {
          throw new RemoteEmbeddingError(
            `Remote embedding dimension drifted: expected ${this.dims}, got ` +
              `${Array.isArray(v) ? v.length : 'non-array'} (did the server switch models?): ` +
              this.config.serverUrl
          )
        }
      }
    }
  }

  private async postEmbed(inputs: string[]): Promise<number[][]> {
    return withTimeout(async (signal) => {
      const res = await fetch(`${this.config.serverUrl}/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs }),
        signal,
      })
      if (!res.ok) {
        const body = await res.text().catch(() => '')
        if (isRetryableStatus(res.status)) {
          throw new RemoteEmbeddingRetryableError(res.status, body, this.config.serverUrl)
        }
        throw new RemoteEmbeddingError(
          `HTTP ${res.status} from embedding server ${this.config.serverUrl}${body ? `: ${body.slice(0, 200)}` : ''}`
        )
      }
      const vectors: number[][] = await res.json()
      return vectors
    }, this.config.timeoutMs ?? 30_000)
  }
}
