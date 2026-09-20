import { type FSWatcher, watch } from 'node:fs'

/**
 * Same hygiene rule as the document scanner (utils/scan.ts): dependency trees
 * and VCS metadata never carry document changes worth reconciling. Keep the
 * two in sync when the scanner's rule changes.
 */
const HYGIENE = /(^|\/)(node_modules|\.git)(\\|\/|$)/

export interface AutoSyncWatchOptions {
  baseDirs: readonly string[]
  /** Drift signal — the caller routes it into its own throttle + reconciliation. */
  onDriftSignal: () => void
  onError?: (message: string) => void
}

/**
 * Watch the document roots so any write outside the hygiene-excluded trees
 * fires a drift signal. The signal carries no payload: dedup, throttling and
 * the fingerprint reconciliation all live in the caller (maybeAutoSync), so
 * event storms collapse into at most one reconciliation per throttle window.
 *
 * Watchers are fire-and-forget for the stdio server lifecycle: the process
 * dies with the omp session and the OS reclaims the inotify handles. A root
 * that cannot be watched (missing, permission) degrades to query-triggered
 * reconciliation only — reported, never thrown.
 */
export function startAutoSyncWatch(options: AutoSyncWatchOptions): FSWatcher[] {
  const watchers: FSWatcher[] = []
  for (const dir of options.baseDirs) {
    try {
      const watcher = watch(dir, { recursive: true }, (_event, filename) => {
        if (filename === null) {
          // Directory-level or rename events without a name: signal anyway —
          // missing a real change costs more than a wasted fingerprint pass,
          // and the caller's throttle bounds that cost.
          options.onDriftSignal()
          return
        }
        if (!HYGIENE.test(`${dir}/${filename}`)) {
          options.onDriftSignal()
        }
      })
      watcher.on('error', (error: unknown) => {
        options.onError?.(
          `Auto-sync watch on ${dir} failed: ${error instanceof Error ? error.message : error}`
        )
      })
      watchers.push(watcher)
    } catch (error: unknown) {
      options.onError?.(
        `Auto-sync watch unavailable on ${dir}: ${error instanceof Error ? error.message : error}`
      )
    }
  }
  return watchers
}
