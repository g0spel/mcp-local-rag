import { isRecord } from './type-guards.js'

/** Display-only context. Paths are ordered sections intersecting the chunk, not alternatives. */
export interface SourceContext {
  headingPaths: string[][]
  /** One-based inclusive lines in the original MD/TXT file, never extracted PDF/DOCX text. */
  startLine?: number
  endLine?: number
  /** One-based physical PDF pages, not printed page labels. */
  startPage?: number
  endPage?: number
}

export interface HeadingAnchor {
  offset: number
  level: number
  text: string
}

/** Offsets refer to the final text supplied to the chunker. */
export interface SourceMap {
  headings: HeadingAnchor[]
  lineStarts?: number[]
  pages?: { start: number; end: number; page: number }[]
}

function lineAt(starts: number[], offset: number): number {
  let low = 0
  let high = starts.length
  while (low < high) {
    const mid = (low + high) >>> 1
    if ((starts[mid] ?? 0) <= offset) {
      low = mid + 1
    } else {
      high = mid
    }
  }
  return low
}

export function sourceContextForRange(map: SourceMap, start: number, end: number): SourceContext {
  const stack: HeadingAnchor[] = []
  const headingPaths: string[][] = []
  for (const [index, heading] of map.headings.entries()) {
    if (heading.offset >= end) {
      break
    }
    while (stack.length && (stack.at(-1)?.level ?? 0) >= heading.level) {
      stack.pop()
    }
    stack.push(heading)
    if ((map.headings[index + 1]?.offset ?? Number.POSITIVE_INFINITY) > start) {
      headingPaths.push(stack.map((h) => h.text))
    }
  }
  const pages = map.pages?.filter((page) => page.start < end && page.end > start)
  const firstPage = pages?.[0]
  const lastPage = pages?.at(-1)
  return {
    headingPaths,
    ...(map.lineStarts
      ? { startLine: lineAt(map.lineStarts, start), endLine: lineAt(map.lineStarts, end - 1) }
      : {}),
    ...(firstPage && lastPage ? { startPage: firstPage.page, endPage: lastPage.page } : {}),
  }
}

/** Legacy rows have no context. Invalid stored metadata is omitted rather than guessed. */
export function parseSourceContext(value: unknown): SourceContext | undefined {
  if (typeof value !== 'string' || !value) {
    return undefined
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(value)
  } catch {
    return undefined
  }
  if (!isRecord(parsed) || !Array.isArray(parsed['headingPaths'])) {
    return undefined
  }
  const paths = parsed['headingPaths']
  if (
    !paths.every(
      (path) =>
        Array.isArray(path) && path.length > 0 && path.every((part) => typeof part === 'string')
    )
  ) {
    return undefined
  }
  if (
    !validRange(parsed['startLine'], parsed['endLine']) ||
    !validRange(parsed['startPage'], parsed['endPage'])
  ) {
    return undefined
  }
  return {
    headingPaths: paths,
    ...(typeof parsed['startLine'] === 'number' && typeof parsed['endLine'] === 'number'
      ? { startLine: parsed['startLine'], endLine: parsed['endLine'] }
      : {}),
    ...(typeof parsed['startPage'] === 'number' && typeof parsed['endPage'] === 'number'
      ? { startPage: parsed['startPage'], endPage: parsed['endPage'] }
      : {}),
  }
}

function validRange(start: unknown, end: unknown): boolean {
  if (start === undefined && end === undefined) {
    return true
  }
  return (
    typeof start === 'number' &&
    typeof end === 'number' &&
    Number.isInteger(start) &&
    Number.isInteger(end) &&
    start >= 1 &&
    end >= start
  )
}
