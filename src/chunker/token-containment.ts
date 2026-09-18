// Reduces oversized sentence units and groups to pieces that measure at or below
// the embedder's resolved token cap, cutting only at grapheme boundaries.

/** Measures true, unclamped token lengths for each input text. */
export type TokenCounter = (texts: string[]) => Promise<number[]>

/** The unit shape containment reads. `text` must be the exact slice `[sourceStart, sourceEnd)`. */
export interface ContainmentUnit {
  text: string
  sourceStart: number
  sourceEnd: number
}

export interface ContainmentBudget {
  /** Resolved token cap: a piece measuring at or below it fits the model's window. */
  cap: number
  countTokens: TokenCounter
}

export interface RunContainmentBudget<T> extends ContainmentBudget {
  /**
   * Joins a run exactly as the chunk text is built, so the measured text is the
   * text that will be embedded.
   */
  joinUnits: (units: readonly T[]) => string
}

/**
 * Cuts land on grapheme boundaries: a string index addresses a UTF-16 code unit,
 * so an unconstrained cut can leave a lone surrogate ('\u{20000}'.length === 2)
 * or split a base character from its combining marks.
 */
const graphemeSegmenter = new Intl.Segmenter('und', { granularity: 'grapheme' })

/** Grapheme boundary offsets in `text`, terminated by `text.length`. */
function graphemeBoundaries(text: string): number[] {
  const boundaries: number[] = []
  for (const segment of graphemeSegmenter.segment(text)) {
    boundaries.push(segment.index)
  }
  boundaries.push(text.length)
  return boundaries
}

async function measure(text: string, budget: ContainmentBudget): Promise<number> {
  const [tokens] = await budget.countTokens([text])
  if (tokens === undefined) {
    throw new Error('Token counter returned no measurement for the requested text')
  }
  return tokens
}

/**
 * Divide one sentence unit into pieces that each measure at or below the cap.
 *
 * Halving, not a search for the longest fitting prefix: token count is not
 * monotonic in prefix length — the prefixes of `playingx` measure 3, 3, 4, 3,
 * 4, 4, 3, 4 here. A single grapheme over the cap is emitted anyway, leaving
 * the embedder's clamp to truncate it.
 */
export async function splitUnitToFit<T extends ContainmentUnit>(
  unit: T,
  budget: ContainmentBudget
): Promise<T[]> {
  const boundaries = graphemeBoundaries(unit.text)
  const graphemeCount = boundaries.length - 1
  const pieces: T[] = []
  let start = 0
  let nextCandidate = budget.cap

  while (start < graphemeCount) {
    const pieceStart = boundaries[start] ?? 0
    const measureCandidate = (size: number): Promise<number> =>
      measure(unit.text.slice(pieceStart, boundaries[start + size] ?? unit.text.length), budget)

    let candidate = Math.min(graphemeCount - start, nextCandidate)
    let tokens = await measureCandidate(candidate)
    while (tokens > budget.cap && candidate > 1) {
      candidate = Math.floor(candidate / 2)
      tokens = await measureCandidate(candidate)
    }
    nextCandidate = candidate * 2
    const pieceEnd = boundaries[start + candidate] ?? unit.text.length
    pieces.push({
      ...unit,
      text: unit.text.slice(pieceStart, pieceEnd),
      sourceStart: unit.sourceStart + pieceStart,
      sourceEnd: unit.sourceStart + pieceEnd,
    })
    start += candidate
  }

  return pieces
}

/**
 * Divide a group into consecutive runs of whole units whose joined text measures
 * at or below the cap.
 *
 * A one-unit run is final even above the cap: {@link splitUnitToFit} already
 * reduced that unit as far as a valid cut allows. Each division strictly reduces
 * the units per run, so progress does not depend on the text.
 */
export async function splitUnitsIntoFittingRuns<T>(
  units: readonly T[],
  budget: RunContainmentBudget<T>
): Promise<T[][]> {
  if (units.length === 0) {
    return []
  }
  if (units.length === 1) {
    return [[...units]]
  }
  if ((await measure(budget.joinUnits(units), budget)) <= budget.cap) {
    return [[...units]]
  }

  const mid = Math.ceil(units.length / 2)
  const head = await splitUnitsIntoFittingRuns(units.slice(0, mid), budget)
  const tail = await splitUnitsIntoFittingRuns(units.slice(mid), budget)
  return [...head, ...tail]
}
