// Cross-process sync mutex: a lockfile beside the database keyed by the
// holder's PID. The in-process mutation guard (server-side) cannot see other
// processes — the CLI `sync`, the MCP `sync_start`, and every auto-sync share
// one LanceDB, so without this file two reconciliations can race on it.
//
// A lock whose holder PID is no longer alive is stale by definition and is
// stolen, so a killed process (crash, OOM, interrupt) can never deadlock the
// queue: the next sync simply takes over. PID reuse is accepted as a residual
// risk (large PID space, long recycle horizon).
import { open, readFile, unlink } from 'node:fs/promises'
import { resolve } from 'node:path'
import { errorCode } from './type-guards.js'

function lockPath(dbPath: string): string {
  return resolve(dbPath, '..', 'sync.lock')
}

/** Try to take the cross-process sync lock. `false` = another live holder. */
export async function acquireSyncLock(dbPath: string): Promise<boolean> {
  const path = lockPath(dbPath)
  try {
    const owner = Number.parseInt((await readFile(path, 'utf8')).split(' ')[0] ?? '', 10)
    if (Number.isFinite(owner)) {
      try {
        process.kill(owner, 0) // liveness probe: signal 0 is a no-op
        return false // holder still alive — real contention
      } catch {
        // ESRCH: holder is gone — zombie lock, steal it below.
      }
    }
    await unlink(path).catch(() => {})
  } catch {
    // No lock file — free to take.
  }
  try {
    const handle = await open(path, 'wx')
    await handle.writeFile(`${process.pid} ${new Date().toISOString()}\n`)
    await handle.close()
    return true
  } catch (error) {
    // EEXIST: lost a concurrent create race — real contention, report it.
    // Any other error (unwritable/missing lock location, test scaffolds) is a
    // degraded environment: the lock is best-effort mutual exclusion, so we
    // yield rather than brick sync on an unusable lock location.
    if (errorCode(error) === 'EEXIST') {
      return false
    }
    return true
  }
}

/** Drop the lock after a finished run. Missing file is fine. */
export async function releaseSyncLock(dbPath: string): Promise<void> {
  await unlink(lockPath(dbPath)).catch(() => {})
}
