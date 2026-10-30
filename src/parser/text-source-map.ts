import type { HeadingAnchor, SourceMap } from '../utils/source-context.js'

/** Markdown headings outside fenced code and YAML frontmatter. */
function markdownHeadings(text: string): HeadingAnchor[] {
  // Replace excluded blocks with spaces to retain exact source offsets.
  const prose = text
    .replace(/^---\r?\n(?:[^\n]*\n)*?(?:---|\.\.\.)[ \t]*(?:\r?\n|$)/, (match) =>
      match.replace(/[^\n]/g, ' ')
    )
    .replace(
      /^ {0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:^ {0,3}\1[~`]*[ \t]*\r?$|(?![\s\S]))/gm,
      (match) => match.replace(/[^\n]/g, ' ')
    )
  const headings: HeadingAnchor[] = []
  const pattern = /^ {0,3}(#{1,6})(?:[\t ]+|$)(.*)$|^([^\n]+)\r?\n {0,3}(=+|-+)[ \t]*\r?$/gm
  for (const match of prose.matchAll(pattern)) {
    const atx = match[1]
    const title = atx ? (match[2] ?? '').replace(/\s+#+\s*$/, '').trim() : (match[3] ?? '').trim()
    const setextLevel = match[4]?.startsWith('=') ? 1 : 2
    const blockSyntax = /^(?:>|[-+*]\s|\d+[.)]\s|[-*_]+$)/.test(title)
    if (title && !/^ {4}/.test(match[0]) && (atx || !blockSyntax)) {
      headings.push({
        offset: match.index,
        level: atx ? atx.length : setextLevel,
        text: title,
      })
    }
  }
  return headings
}

export function textSourceMap(text: string, markdown: boolean): SourceMap {
  const lineStarts = [0]
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') {
      lineStarts.push(i + 1)
    }
  }
  return { headings: markdown ? markdownHeadings(text) : [], lineStarts }
}
