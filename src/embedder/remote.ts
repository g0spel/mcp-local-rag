// Remote embedding server (TEI-compatible) backend.
//
// Activated by RAG_EMBEDDING_SERVER_URL: when set, the factory returns this
// class instead of the local Transformers.js pipeline, so embedding compute
// runs on a self-hosted server (e.g. HuggingFace TEI) while document content
// never leaves the deployment network.
//
// Wire protocol: POST {serverUrl}/embed  {"inputs": string[]} → number[][]
// (TEI-native shape; the server owns model choice, dtype, and truncation).
import { request as httpRequest } from 'node:http'
import { request as httpsRequest } from 'node:https'
import { AppError } from '../utils/errors.js'
import type { EmbeddingComputationIdentity } from './index.js'

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
// 隧道（frp 链路）存在 ~30s 的周期性失败窗口（约 90-110s 一个周期），窗口内 POST 全挂。
// 重试序列必须能跨越窗口：4 次重试 × max 20s 背靠背 ≈ 最长 ~70s 的覆盖跨度。
const MAX_RETRIES = 4
const INITIAL_BACKOFF_MS = 2000
const MAX_BACKOFF_MS = 20000

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
// One-shot HTTP transport (dedicated connection)
// ============================================

/**
 * Single-request HTTP(S) transport over a dedicated connection (`agent:
 * false`): the socket is created per request and closed when the response
 * completes.
 *
 * Why not `fetch`: undici classifies `Connection` as a forbidden header and
 * silently strips it, so the earlier `Connection: 'close'` header never
 * disabled keep-alive — pooled idle sockets stayed open for hours across the
 * ssh-forwarded tunnel and accumulated into a deadlock (one ssh channel per
 * pooled socket). A dedicated per-request connection cannot pool, so idle
 * channel accumulation is structurally impossible.
 */
function rawRequest(
  url: URL,
  init: { method: 'GET' | 'POST'; body?: string; signal: AbortSignal }
): Promise<{ status: number; text: string }> {
  const send = url.protocol === 'https:' ? httpsRequest : httpRequest
  return new Promise((resolve, reject) => {
    if (init.signal.aborted) {
      reject(abortError())
      return
    }
    const req = send(
      url,
      {
        method: init.method,
        agent: false,
        headers: init.body === undefined ? {} : { 'Content-Type': 'application/json' },
      },
      (res) => {
        const chunks: Buffer[] = []
        res.on('data', (chunk: Buffer) => chunks.push(chunk))
        res.on('error', reject)
        res.on('end', () => {
          resolve({ status: res.statusCode ?? 0, text: Buffer.concat(chunks).toString('utf8') })
        })
      }
    )
    req.on('error', reject)
    init.signal.addEventListener(
      'abort',
      () => {
        req.destroy()
        reject(abortError())
      },
      { once: true }
    )
    if (init.body !== undefined) {
      req.write(init.body)
    }
    req.end()
  })
}

/** Timeout-shaped abort error (retryable, matching the previous fetch behavior). */
function abortError(): Error {
  return Object.assign(new Error('aborted'), { name: 'AbortError' })
}

// ============================================
// Token estimation (containment support)
// ============================================

/**
 * Conservative CJK-aware token estimate for a server-side tokenizer we cannot
 * run locally. Calibrated against the deployment's own tokenizer (XLM-R /
 * bge-m3): worst observed densities were ~0.76 tokens/char CJK prose, 0.62 for
 * digit-heavy table rows and 0.46 for code; the experiment-ledger file that
 * motivated containment measures 0.615 overall. The coefficients deliberately
 * over-estimate (dense script x1.4, everything else x0.7) so containment
 * splits too early rather than letting an oversized chunk reach the server.
 */
export function estimateTokens(text: string): number {
  let dense = 0
  let total = 0
  for (const ch of text) {
    total += 1
    const code = ch.codePointAt(0) ?? 0
    if (
      (code >= 0x3040 && code <= 0x30ff) || // kana
      (code >= 0x3400 && code <= 0x4dbf) || // CJK extension A
      (code >= 0x4e00 && code <= 0x9fff) || // CJK unified ideographs
      (code >= 0xac00 && code <= 0xd7af) || // hangul syllables
      (code >= 0xf900 && code <= 0xfaff) // CJK compatibility ideographs
    ) {
      dense += 1
    }
  }
  return Math.ceil(dense * 1.4 + (total - dense) * 0.7)
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
 * - `tokenLimit` mirrors the server's own input cap, probed once from
 *   `GET /info` (`max_input_length`). When the server does not expose it —
 *   or a strictly validating server rejects oversized inputs without
 *   truncating — the cap stays `null` and containment stays off, exactly the
 *   previous behavior. `getTokenLimit`/`countTokens` implement the chunker's
 *   optional `EmbedderInterface` members so token containment activates.
 * - `dispose()` is a no-op — no in-process model resources.
 * - `device` / `dtype` / `cacheDir` / `modelPath` are server-side concerns.
 */
export class RemoteEmbedder {
  private readonly config: RemoteEmbedderConfig
  private dims: number | null = null
  private maxInputTokens: number | null = null
  private serverModelId: string | null = null
  private serverModelSha: string | null = null
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
    const server = await this.probeServerInfo()
    this.maxInputTokens = server.maxInputTokens
    this.serverModelId = server.modelId
    this.serverModelSha = server.modelSha
    this.initialized = true
    console.error(
      `RemoteEmbedder: connected to "${this.config.serverUrl}" (dimension ${this.dims}, model ${server.modelId ?? 'unknown'}, max input ${this.maxInputTokens ?? 'unknown'} tokens)`
    )
  }

  /** Best-effort `GET /info`: input cap + serving-model identity for reuse. */
  private async probeServerInfo(): Promise<{
    maxInputTokens: number | null
    modelId: string | null
    modelSha: string | null
  }> {
    const none = { maxInputTokens: null, modelId: null, modelSha: null }
    let info: unknown
    try {
      info = await withTimeout(async (signal) => {
        const res = await rawRequest(new URL(`${this.config.serverUrl}/info`), {
          method: 'GET',
          signal,
        })
        if (res.status < 200 || res.status >= 300) {
          return null
        }
        return JSON.parse(res.text)
      }, 10_000)
    } catch {
      return none
    }
    if (typeof info !== 'object' || info === null) {
      return none
    }
    const cap: unknown = 'max_input_length' in info ? info.max_input_length : undefined
    const modelId: unknown = 'model_id' in info ? info.model_id : undefined
    const modelSha: unknown = 'model_sha' in info ? info.model_sha : undefined
    return {
      maxInputTokens: typeof cap === 'number' && Number.isFinite(cap) && cap > 0 ? cap : null,
      modelId: typeof modelId === 'string' && modelId.length > 0 ? modelId : null,
      modelSha: typeof modelSha === 'string' && modelSha.length > 0 ? modelSha : null,
    }
  }

  async dispose(): Promise<void> {
    this.initialized = false
    this.dims = null
    this.maxInputTokens = null
    this.serverModelId = null
    this.serverModelSha = null
  }

  /** The server's own input cap when `/info` exposed it, else `null`. */
  get tokenLimit(): number | null {
    return this.maxInputTokens
  }

  /** The remote stack has no local model-load warnings to surface. */
  get modelWarnings(): readonly string[] {
    return []
  }

  /** Optional `EmbedderInterface` member: activates the chunker's containment. */
  async getTokenLimit(): Promise<number | null> {
    await this.initialize()
    return this.maxInputTokens
  }

  /**
   * `PersistentEmbeddingProvider` member: the serving model's identity from
   * `/info` (model id + sha + endpoint), so embedding snapshots taken against
   * one server are never reused against another. A server that did not expose
   * its model id cannot prove which model served — `null` disables snapshot
   * reuse (re-embed) rather than risking a shared fake fingerprint across
   * different models behind the same flaky `/info`.
   */
  async getComputationIdentity(): Promise<EmbeddingComputationIdentity | null> {
    await this.initialize()
    if (this.dims === null || this.serverModelId === null) {
      return null
    }
    return {
      fingerprint: `remote:${this.config.serverUrl}:${this.serverModelId}@${this.serverModelSha ?? 'unpinned'}`,
      dimension: this.dims,
    }
  }

  /**
   * Optional `EmbedderInterface` member: conservative local estimate (see
   * {@link estimateTokens}) — no tokenizer download, no extra server round trip.
   */
  async countTokens(texts: string[]): Promise<number[]> {
    return texts.map((text) => estimateTokens(text))
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
      const res = await rawRequest(new URL(`${this.config.serverUrl}/embed`), {
        method: 'POST',
        body: JSON.stringify({ inputs }),
        signal,
      })
      if (res.status < 200 || res.status >= 300) {
        if (isRetryableStatus(res.status)) {
          throw new RemoteEmbeddingRetryableError(res.status, res.text, this.config.serverUrl)
        }
        throw new RemoteEmbeddingError(
          `HTTP ${res.status} from embedding server ${this.config.serverUrl}${res.text ? `: ${res.text.slice(0, 200)}` : ''}`
        )
      }
      return JSON.parse(res.text)
    }, this.config.timeoutMs ?? 30_000)
  }
}
