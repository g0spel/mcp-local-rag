import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it, vi } from 'vitest'
import { startAutoSyncWatch } from '../auto-sync-watch.js'

describe('startAutoSyncWatch', () => {
  const dirs: string[] = []

  function makeRoot(): string {
    const dir = mkdtempSync(join(tmpdir(), 'auto-sync-watch-'))
    dirs.push(dir)
    return dir
  }

  afterAll(() => {
    for (const dir of dirs) {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('fires the drift signal when a file is written', async () => {
    const root = makeRoot()
    let signals = 0
    const watchers = startAutoSyncWatch({
      baseDirs: [root],
      onDriftSignal: () => {
        signals += 1
      },
    })
    try {
      writeFileSync(join(root, 'doc.md'), 'hello')
      // fs.watch events come from the kernel (inotify) — no fake-timer path exists,
      // so this must await the real platform event delivery.
      await vi.waitFor(() => expect(signals).toBeGreaterThan(0))
    } finally {
      for (const w of watchers) {
        w.close()
      }
    }
  })

  it('fires when a nested file changes', async () => {
    const root = makeRoot()
    const sub = join(root, 'notes')
    mkdirSync(sub)
    let signals = 0
    const watchers = startAutoSyncWatch({
      baseDirs: [root],
      onDriftSignal: () => {
        signals += 1
      },
    })
    try {
      writeFileSync(join(sub, 'deep.md'), 'nested')
      await vi.waitFor(() => expect(signals).toBeGreaterThan(0))
    } finally {
      for (const w of watchers) {
        w.close()
      }
    }
  })

  it('ignores writes under node_modules and .git', async () => {
    const root = makeRoot()
    for (const dir of ['node_modules/pkg', '.git']) {
      mkdirSync(join(root, dir), { recursive: true })
    }
    let signals = 0
    const watchers = startAutoSyncWatch({
      baseDirs: [root],
      onDriftSignal: () => {
        signals += 1
      },
    })
    try {
      writeFileSync(join(root, 'node_modules', 'pkg', 'index.js'), 'x')
      writeFileSync(join(root, '.git', 'HEAD'), 'ref')
      // Negative assertion against a kernel event stream: absence can only be
      // observed by waiting out the platform's real delivery window.
      await new Promise((resolve) => setTimeout(resolve, 600))
      expect(signals).toBe(0)
    } finally {
      for (const w of watchers) {
        w.close()
      }
    }
  })

  it('degrades to a report when a root cannot be watched', () => {
    const missing = join(makeRoot(), 'does-not-exist')
    const errors: string[] = []
    const watchers = startAutoSyncWatch({
      baseDirs: [missing],
      onDriftSignal: () => {},
      onError: (message) => {
        errors.push(message)
      },
    })
    expect(watchers).toHaveLength(0)
    expect(errors).toHaveLength(1)
    expect(errors[0]).toContain('does-not-exist')
  })
})
