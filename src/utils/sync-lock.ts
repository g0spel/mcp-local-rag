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

/**
 * Read-only contention check: a live foreign holder exists. Used to reject a
 * `sync_start` (and skip its state-file writes) BEFORE registering a job, so a
 * cross-process loser never overwrites the holder's running/terminal state in
 * the shared sync-state file. Does not take or steal the lock; a missing or
 * stale lock reports free (the real acquire remains authoritative).
 */
export async function isSyncLockHeld(dbPath: string): Promise<boolean> {
  const owner = await readLockOwner(dbPath)
  if (owner === null || owner === process.pid) {
    return false
  }
  return isProcessAlive(owner)
}

/** The holder PID recorded in the lock file, or null when absent/unreadable. */
async function readLockOwner(dbPath: string): Promise<number | null> {
  try {
    const owner = Number.parseInt(
      (await readFile(lockPath(dbPath), 'utf8')).split(' ')[0] ?? '',
      10
    )
    return Number.isFinite(owner) ? owner : null
  } catch {
    return null // no lock file
  }
}

/**
 * Liveness via signal 0. EPERM means the target exists but is not ours to
 * signal (e.g. another uid's process) — that still proves it is alive, so a
 * live holder is never mistaken for a stale lock and stolen from.
 */
function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0) // no-op probe
    return true
  } catch (error) {
    return errorCode(error) === 'EPERM'
  }
}

/** Try to take the cross-process sync lock. `false` = another live holder. */
export async function acquireSyncLock(dbPath: string): Promise<boolean> {
  const path = lockPath(dbPath)
  try {
    const owner = await readLockOwner(dbPath)
    if (owner !== null && owner !== process.pid && isProcessAlive(owner)) {
      return false // holder still alive — real contention
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
