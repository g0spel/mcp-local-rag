// Command parsing: the configured string becomes an argv vector, and the
// server's own flags are appended as separate elements so query text can never
// be read as command syntax.

import { describe, expect, it } from 'vitest'
import { buildRerankArgv, parseRerankCommand } from '../command.js'

describe('parseRerankCommand', () => {
  it('should split the configured command into an executable and its arguments', () => {
    expect(parseRerankCommand('jev-reranker --score-field score --score-order asc')).toEqual({
      executable: 'jev-reranker',
      args: ['--score-field', 'score', '--score-order', 'asc'],
    })
  })

  it('should return an executable with no arguments for a bare command', () => {
    expect(parseRerankCommand('jev-reranker')).toEqual({ executable: 'jev-reranker', args: [] })
  })

  it('should ignore surrounding and repeated whitespace', () => {
    expect(parseRerankCommand('  jev-reranker \t --top-field\n score  ')).toEqual({
      executable: 'jev-reranker',
      args: ['--top-field', 'score'],
    })
  })

  it('should keep quote characters as part of the token, since no shell interprets them', () => {
    expect(parseRerankCommand(`jev-reranker --label 'two`)).toEqual({
      executable: 'jev-reranker',
      args: ['--label', `'two`],
    })
  })

  it('should return undefined for a command with no token', () => {
    expect(parseRerankCommand('')).toBeUndefined()
    expect(parseRerankCommand('   \t ')).toBeUndefined()
  })
})

describe('buildRerankArgv', () => {
  it('should append --query and --top after the configured arguments as separate elements', () => {
    expect(
      buildRerankArgv({ executable: 'jev-reranker', args: ['--score-field', 'score'] }, 'cats', 3)
    ).toEqual(['--score-field', 'score', '--query', 'cats', '--top', '3'])
  })

  it('should keep a query containing shell metacharacters as one element', () => {
    const query = 'cats; rm -rf / && echo "$(whoami)" | tee /tmp/x'
    const argv = buildRerankArgv({ executable: 'jev-reranker', args: [] }, query, 1)

    expect(argv).toEqual(['--query', query, '--top', '1'])
  })

  it('should keep a query that is empty or only whitespace as one element', () => {
    expect(buildRerankArgv({ executable: 'r', args: [] }, '   ', 2)).toEqual([
      '--query',
      '   ',
      '--top',
      '2',
    ])
  })
})
