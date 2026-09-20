// Response matching: the child's stdout is untrusted, so it is accepted only as
// an ordering over candidates the server already holds. Nothing from the child
// reaches the caller except that order.

import { describe, expect, it } from 'vitest'
import { matchRerankResponse } from '../response.js'

interface TestCandidate {
  id: string
  filePath: string
  chunkIndex: number
  text: string
  score: number
  fileTitle: string | null
}

function candidate(id: string, chunkIndex: number): TestCandidate {
  return {
    id,
    filePath: `/docs/${id}.md`,
    chunkIndex,
    text: `text of ${id}`,
    score: 0.5,
    fileTitle: id,
  }
}

const candidates = [candidate('a', 0), candidate('b', 1), candidate('c', 2)]

function childItem(source: TestCandidate, overrides: Record<string, unknown> = {}) {
  return {
    filePath: source.filePath,
    chunkIndex: source.chunkIndex,
    text: 'rewritten by the child',
    score: 0.01,
    fileTitle: 'rewritten by the child',
    images: [],
    ...overrides,
  }
}

function stdoutOf(items: unknown[]): string {
  return JSON.stringify(items)
}

describe('matchRerankResponse', () => {
  it("should return the server's own candidates in the child's order", () => {
    const result = matchRerankResponse(
      stdoutOf([childItem(candidates[2]), childItem(candidates[0]), childItem(candidates[1])]),
      candidates,
      3
    )

    expect(result).toEqual({ ok: true, candidates: [candidates[2], candidates[0], candidates[1]] })
  })

  it('should accept a response trimmed to the expected count', () => {
    const result = matchRerankResponse(
      stdoutOf([childItem(candidates[1]), childItem(candidates[2])]),
      candidates,
      2
    )

    expect(result).toEqual({ ok: true, candidates: [candidates[1], candidates[2]] })
  })

  it('should tolerate surrounding whitespace around the JSON array', () => {
    const result = matchRerankResponse(
      `\n  ${stdoutOf([childItem(candidates[0])])}\n`,
      [candidates[0]],
      1
    )

    expect(result).toEqual({ ok: true, candidates: [candidates[0]] })
  })

  it('should reject an unknown identifier', () => {
    const unknown = childItem(candidates[0], { filePath: '/docs/elsewhere.md' })
    const result = matchRerankResponse(
      stdoutOf([unknown, childItem(candidates[1]), childItem(candidates[2])]),
      candidates,
      3
    )

    expect(result.ok).toBe(false)
  })

  it('should reject an identifier whose chunkIndex belongs to no candidate', () => {
    const result = matchRerankResponse(
      stdoutOf([
        childItem(candidates[0], { chunkIndex: 99 }),
        childItem(candidates[1]),
        childItem(candidates[2]),
      ]),
      candidates,
      3
    )

    expect(result.ok).toBe(false)
  })

  it('should reject a duplicated identifier', () => {
    const result = matchRerankResponse(
      stdoutOf([childItem(candidates[0]), childItem(candidates[0]), childItem(candidates[1])]),
      candidates,
      3
    )

    expect(result.ok).toBe(false)
  })

  it('should reject a dropped candidate', () => {
    const result = matchRerankResponse(
      stdoutOf([childItem(candidates[0]), childItem(candidates[1])]),
      candidates,
      3
    )

    expect(result.ok).toBe(false)
  })

  it('should reject an extra item', () => {
    const result = matchRerankResponse(
      stdoutOf([
        childItem(candidates[0]),
        childItem(candidates[1]),
        childItem(candidates[2]),
        childItem(candidates[2]),
      ]),
      candidates,
      2
    )

    expect(result.ok).toBe(false)
  })

  it('should reject stdout that does not parse as JSON', () => {
    expect(matchRerankResponse('not json at all', candidates, 3).ok).toBe(false)
    expect(matchRerankResponse('', candidates, 3).ok).toBe(false)
  })

  it('should reject JSON that is not an array', () => {
    expect(matchRerankResponse('{"results":[]}', candidates, 3).ok).toBe(false)
    expect(matchRerankResponse('null', candidates, 3).ok).toBe(false)
    expect(matchRerankResponse('"a string"', candidates, 3).ok).toBe(false)
  })

  it('should reject an item whose identifier fields have the wrong type', () => {
    expect(
      matchRerankResponse(stdoutOf([childItem(candidates[0], { chunkIndex: '0' })]), candidates, 1)
        .ok
    ).toBe(false)
    expect(
      matchRerankResponse(stdoutOf([childItem(candidates[0], { filePath: 7 })]), candidates, 1).ok
    ).toBe(false)
    expect(matchRerankResponse(stdoutOf(['just a string']), candidates, 1).ok).toBe(false)
    expect(matchRerankResponse(stdoutOf([null]), candidates, 1).ok).toBe(false)
  })

  it('should carry neither document text nor candidate paths in its rejection reason', () => {
    const result = matchRerankResponse(
      stdoutOf([childItem(candidates[0]), childItem(candidates[0]), childItem(candidates[1])]),
      candidates,
      3
    )

    expect(result.ok).toBe(false)
    const reason = result.ok ? '' : result.reason
    expect(reason.length).toBeGreaterThan(0)
    for (const item of candidates) {
      expect(reason).not.toContain(item.text)
      expect(reason).not.toContain(item.filePath)
    }
  })
})
