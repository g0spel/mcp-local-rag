import { JSDOM } from 'jsdom'
import { describe, expect, it } from 'vitest'
import { sourceContextForRange } from '../../utils/source-context.js'
import { convertDocxDocumentToText } from '../docx-parser.js'
import { textSourceMap } from '../text-source-map.js'

describe('source context', () => {
  it('locates repeated Markdown text and reports every intersecting heading path', () => {
    const text = '# Archive\r\n\r\n## A\r\nSame quote.\r\n\r\n## B\r\nSame quote.\r\n'
    const map = textSourceMap(text, true)
    const start = text.lastIndexOf('Same quote.')
    expect(sourceContextForRange(map, start, start + 11)).toEqual({
      startLine: 7,
      endLine: 7,
      headingPaths: [['Archive', 'B']],
    })
    expect(
      sourceContextForRange(map, text.indexOf('Same quote.'), text.length).headingPaths
    ).toEqual([
      ['Archive', 'A'],
      ['Archive', 'B'],
    ])
  })

  it('ignores fenced code/frontmatter and recognizes setext headings', () => {
    const text = '---\ntitle: Test\n---\n# Real\n```md\n# Fake\n```\nChild\n-----\nBody'
    expect(textSourceMap(text, true).headings.map((h) => h.text)).toEqual(['Real', 'Child'])
  })

  it('preserves DOCX headings at the exact serialized offset without inventing source lines', () => {
    const doc = new JSDOM('<h1>Guide</h1><p>First.</p><h2>Setup</h2><p>Second.</p>').window.document
    const parsed = convertDocxDocumentToText(doc)
    const start = parsed.content.indexOf('Second.')
    if (!parsed.sourceMap) {
      throw new Error('Missing DOCX source map')
    }
    expect(sourceContextForRange(parsed.sourceMap, start, start + 7)).toEqual({
      headingPaths: [['Guide', 'Setup']],
    })
  })
  it.each(['- item two', '* item', '+ item', '1. item', '1) item', '> quote', '---'])(
    'does not treat block syntax %s followed by a rule as a setext heading',
    (block) => {
      expect(textSourceMap(`${block}\n---\n# Real`, true).headings.map((h) => h.text)).toEqual([
        'Real',
      ])
    }
  )

  it.each(['\n', '\r\n'])('ignores empty frontmatter with line ending %j', (newline) => {
    expect(
      textSourceMap(['---', '---', '# Real', '', '---'].join(newline), true).headings.map(
        (h) => h.text
      )
    ).toEqual(['Real'])
  })
})
