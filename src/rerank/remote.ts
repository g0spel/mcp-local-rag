// HTTP reranker: POST {baseUrl}/rerank and take back one score per candidate.
//
// The command path (`RAG_RERANK_CMD`) spawns a process per query and injects the
// query through an argv template, which is easy to forget. This path talks to a
// TEI-compatible endpoint in-process: the query is a request-body field, so it
// cannot be dropped silently, and there is no child-process startup cost.
//
// Every failure degrades to the pre-rerank ordering, exactly like the command
// path: a reranker improves ranking and must never fail the request.

import { rawRequest, withTimeout } from '../embedder/remote.js'
import { isRecord } from '../utils/type-guards.js'
import type { RerankResult } from './response.js'

export interface RemoteRerankConfig {
  /** Base URL without the `/rerank` suffix (e.g. `http://127.0.0.1:18082`). */
  baseUrl: string
  /** Per-attempt budget; the whole call may take up to attempts × this. */
  timeoutMs: number
}

/** Attempts for network errors and 429/5xx; other statuses fail immediately. */
const MAX_ATTEMPTS = 3

interface Scored {
  index: number
  score: number
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function isRetryableStatus(status: number): boolean {
  return status === 429 || status >= 500
}

/**
 * `[{index, score}]` → validated hits, or null when the shape cannot be trusted.
 * Every index must name a candidate that was sent, exactly once: a duplicated or
 * out-of-range index would map one candidate onto another's position.
 */
function parseHits(text: string, candidateCount: number): Scored[] | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return null
  }
  if (!Array.isArray(parsed)) {
    return null
  }
  const seen = new Set<number>()
  const hits: Scored[] = []
  for (const item of parsed) {
    if (!isRecord(item)) {
      return null
    }
    const index = item['index']
    const score = item['score']
    if (typeof index !== 'number' || !Number.isInteger(index)) {
      return null
    }
    if (index < 0 || index >= candidateCount || seen.has(index)) {
      return null
    }
    if (typeof score !== 'number' || !Number.isFinite(score)) {
      return null
    }
    seen.add(index)
    hits.push({ index, score })
  }
  return hits
}

/** Request body for one call: the query is a field, never an argument. */
export function buildRerankRequestBody(query: string, candidates: RerankResult[]): string {
  return JSON.stringify({
    query,
    texts: candidates.map((candidate) => candidate.text),
    // TEI's default returns normalized relevance; raw logits keep the ordering
    // information the cross-encoder actually produced.
    raw_scores: true,
  })
}

type AttemptOutcome =
  | { kind: 'hits'; hits: Scored[] }
  | { kind: 'failure'; reason: string; retryable: boolean }

/** One POST attempt: hits to rank, or why it produced nothing usable. */
async function attemptOnce(
  url: URL,
  body: string,
  timeoutMs: number,
  candidateCount: number
): Promise<AttemptOutcome> {
  try {
    const response = await withTimeout(
      (signal) => rawRequest(url, { method: 'POST', body, signal }),
      timeoutMs
    )
    if (response.status < 200 || response.status >= 300) {
      return {
        kind: 'failure',
        reason: `the endpoint answered HTTP ${response.status}`,
        retryable: isRetryableStatus(response.status),
      }
    }
    const hits = parseHits(response.text, candidateCount)
    return hits === null
      ? {
          kind: 'failure',
          reason: 'the response was not a valid {index, score} set',
          retryable: false,
        }
      : { kind: 'hits', hits }
  } catch (error) {
    return {
      kind: 'failure',
      reason: error instanceof Error ? error.message : String(error),
      retryable: true,
    }
  }
}

/**
 * The reordered candidates, truncated to `top`, or `candidates` unchanged when
 * the endpoint could not answer with a usable result set. Never throws.
 */
export async function rerankCandidatesViaUrl(
  config: RemoteRerankConfig,
  candidates: RerankResult[],
  query: string,
  top: number
): Promise<RerankResult[]> {
  if (candidates.length === 0) {
    return candidates
  }
  const url = new URL(`${config.baseUrl.replace(/\/+$/u, '')}/rerank`)
  const body = buildRerankRequestBody(query, candidates)
  let reason = 'no attempt was made'

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const outcome = await attemptOnce(url, body, config.timeoutMs, candidates.length)
    if (outcome.kind === 'hits') {
      return rankByScore(candidates, outcome.hits, top)
    }
    reason = outcome.reason
    if (!outcome.retryable) {
      break
    }
    if (attempt < MAX_ATTEMPTS) {
      await sleep(500 * 2 ** (attempt - 1))
    }
  }

  console.error(`Rerank: ${reason}; returning the pre-rerank ordering`)
  return candidates
}

/**
 * Highest score first; the response's own order is never trusted.
 *
 * The rewritten `score` is the reranker's raw logit (higher is more relevant),
 * replacing the vector distance the search produced (lower is nearer). The
 * command transport replaces it the same way, so a result stream never mixes
 * both scales — but a consumer reading `score` as a distance would now be wrong.
 */
function rankByScore(candidates: RerankResult[], hits: Scored[], top: number): RerankResult[] {
  const ranked: RerankResult[] = []
  for (const hit of [...hits].sort((a, b) => b.score - a.score)) {
    const candidate = candidates[hit.index]
    if (candidate === undefined) {
      continue
    }
    ranked.push({ ...candidate, score: hit.score })
    if (ranked.length >= top) {
      break
    }
  }
  return ranked.length > 0 ? ranked : candidates
}
