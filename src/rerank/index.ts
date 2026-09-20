// External reranking: hand the search candidates to a configured command and
// take its response as an ordering over them.
//
// Every failure mode degrades to the ordering the caller already had, the way
// the FTS fallback in `VectorStore.searchWithKeywordBoost` does. A reranker is
// an improvement to ranking, so a broken one must not fail the request; a
// process that fails to start arrives as an `error` event, and leaving that
// unhandled would take the server process down.

import { spawn } from 'node:child_process'
import { errorCode } from '../utils/type-guards.js'
import { buildRerankArgv, parseRerankCommand } from './command.js'
import { matchRerankResponse, type RerankCandidate } from './response.js'

export type { RerankCandidate } from './response.js'

/** One rerank call: the candidates to order, and the command that orders them. */
export interface RerankRequest<T extends RerankCandidate> {
  /** Search candidates in their pre-rerank order. Returned as-is on failure. */
  candidates: T[]
  /** Query text, passed to the command as its own argv element. */
  query: string
  /** Result count the caller wants; the command is asked for at most this many. */
  top: number
  /** Configured command, parsed as a whitespace-separated argv vector. */
  command: string
  /** Per-call budget. The spawned process is killed when it elapses. */
  timeoutMs: number
}

type ChildRun = { ok: true; stdout: string } | { ok: false; reason: string }

/**
 * Stdin payload: `docs/schema/query-output.schema.json`. `images` is always
 * empty because attachments are hydrated after reranking, so the payload is
 * schema-valid and no image bytes are read for a candidate that gets trimmed.
 */
function buildRerankPayload(candidates: RerankCandidate[]): string {
  return JSON.stringify(
    candidates.map((candidate) => ({
      filePath: candidate.filePath,
      chunkIndex: candidate.chunkIndex,
      text: candidate.text,
      score: candidate.score,
      fileTitle: candidate.fileTitle,
      images: [],
    }))
  )
}

/**
 * A missing command, one without execute permission, and a `.cmd` / `.bat` shim
 * on Windows all land here. The operator would otherwise just see reranking not
 * happening, so the line says what the command has to be. Only the error's code
 * is reported: its message repeats the executable.
 */
function spawnFailureReason(error: unknown): string {
  return `the command could not be started (${errorCode(error) ?? 'unknown error'}); it must name a directly executable file, not a .cmd or .bat shim`
}

function runRerankCommand(
  executable: string,
  argv: string[],
  payload: string,
  timeoutMs: number
): Promise<ChildRun> {
  return new Promise<ChildRun>((resolve) => {
    let settled = false
    const stdoutChunks: string[] = []
    let timer: NodeJS.Timeout | undefined

    const finish = (result: ChildRun): void => {
      if (settled) {
        return
      }
      settled = true
      clearTimeout(timer)
      resolve(result)
    }

    let child: ReturnType<typeof spawn>
    try {
      // No shell on any platform: the query travels as its own argv element and
      // nothing can splice it back into a command string. `stderr` is ignored
      // because the child is untrusted and could otherwise write document text
      // into this server's log stream.
      child = spawn(executable, argv, { stdio: ['pipe', 'pipe', 'ignore'] })
    } catch (error) {
      finish({ ok: false, reason: spawnFailureReason(error) })
      return
    }

    // The budget itself ends the call. A child that traps the signal, or one
    // whose descendant holds the inherited stdout open, never emits `close`, so
    // waiting for the kill to land would leave the request pending for as long
    // as the child felt like running. Its stdout is dropped and the handle is
    // unreferenced so nothing it does afterwards reaches or holds this server.
    timer = setTimeout(() => {
      child.kill()
      child.stdout?.destroy()
      child.unref()
      finish({ ok: false, reason: `the command timed out after ${timeoutMs}ms and was killed` })
    }, timeoutMs)

    child.stdout?.setEncoding('utf8')
    child.stdout?.on('data', (chunk: string): void => {
      stdoutChunks.push(chunk)
    })
    // A child that exits before reading its input makes this write fail; the
    // exit itself is what gets reported.
    child.stdin?.on('error', (): void => {})
    child.stdin?.end(payload)

    child.on('error', (error): void => {
      finish({ ok: false, reason: spawnFailureReason(error) })
    })
    child.on('close', (code, signal): void => {
      if (code === 0) {
        finish({ ok: true, stdout: stdoutChunks.join('') })
        return
      }
      const outcome = code === null ? `was terminated by ${signal}` : `exited with code ${code}`
      finish({ ok: false, reason: `the command ${outcome}` })
    })
  })
}

function fallback<T>(candidates: T[], reason: string): T[] {
  console.error(`Rerank: ${reason}; returning the pre-rerank ordering`)
  return candidates
}

/**
 * Returns the `min(top, candidates.length)` best candidates in the order the
 * configured command gave, or every candidate unchanged when anything about
 * that run is not exactly what the contract allows. Never throws.
 *
 * The caller decides when reranking is worth a process at all, and trims the
 * fallback ordering itself.
 */
export async function rerankCandidates<T extends RerankCandidate>(
  request: RerankRequest<T>
): Promise<T[]> {
  const { candidates, query, top, command, timeoutMs } = request
  const parsed = parseRerankCommand(command)
  if (parsed === undefined) {
    return fallback(candidates, 'the configured command is empty')
  }

  // Asking for more than was sent would leave the accepted count ambiguous:
  // acceptance requires exactly `min(top, candidateCount)` items back.
  const expectedCount = Math.min(top, candidates.length)
  const run = await runRerankCommand(
    parsed.executable,
    buildRerankArgv(parsed, query, expectedCount),
    buildRerankPayload(candidates),
    timeoutMs
  )
  if (!run.ok) {
    return fallback(candidates, run.reason)
  }

  const matched = matchRerankResponse(run.stdout, candidates, expectedCount)
  return matched.ok ? matched.candidates : fallback(candidates, matched.reason)
}
