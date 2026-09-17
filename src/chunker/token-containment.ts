// Measured token containment for chunker output
// Purpose: reduce oversized sentence units and groups to pieces that measure at or
// below the embedder's resolved token cap, using measurement rather than estimates
// and cutting only at grapheme boundaries.

/** Measures true, unclamped token lengths for each input text. */
export type TokenCounter = (texts: string[]) => Promise<number[]>

/** The unit shape containment reads: its text and the exact source span holding it. */
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
 * Cut candidates are grapheme boundaries because a JavaScript string index
 * addresses a UTF-16 code unit: an unconstrained cut can leave a lone surrogate
 * ('\u{20000}'.length === 2) or separate a base character from its combining
 * marks, which corrupts exactly the dense scripts containment exists for.
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

interface PrefixSearch {
  /** Remainder being divided. */
  text: string
  /** Grapheme boundaries of `text`, terminated by `text.length`. */
  boundaries: readonly number[]
  /** Measured length of the whole remainder, already known to exceed the cap. */
  measuredTokens: number
}

/**
 * Number of leading graphemes to emit as the next piece.
 *
 * The search shrinks only and never looks for the longest fitting prefix: token
 * count is not monotonic in prefix length (the prefixes of `playingx` measure
 * 3, 3, 4, 3, 4, 4, 3, 4 with this repo's default tokenizer), so a longest-prefix
 * search has no sound precondition. Halving reaches one grapheme in O(log n)
 * measurements, which makes termination structural instead of tokenizer-dependent.
 *
 * A single grapheme is returned without measuring: when a piece holds one
 * cluster there is no smaller valid cut, so a measurement above the cap is a
 * recorded fact about that piece rather than a condition to retry.
 */
async function fittingPrefixGraphemes(
  search: PrefixSearch,
  budget: ContainmentBudget
): Promise<number> {
  const graphemeCount = search.boundaries.length - 1
  let candidate = Math.max(1, Math.floor((graphemeCount * budget.cap) / search.measuredTokens))
  while (candidate > 1) {
    const end = search.boundaries[candidate] ?? search.text.length
    if ((await measure(search.text.slice(0, end), budget)) <= budget.cap) {
      return candidate
    }
    candidate = Math.floor(candidate / 2)
  }
  return 1
}

/**
 * Stage A: divide one sentence unit into pieces that each measure at or below the
 * cap, in source coordinates.
 *
 * Offsets are `unit.sourceStart + pieceStart`, which holds because a unit's stored
 * text is exactly its source slice. Every other unit property is carried onto each
 * piece, so an oversized atomic unit yields atomic pieces.
 *
 * The one exception to the cap is a piece holding a single grapheme cluster that
 * still measures above it: it is emitted as-is, the embedder's clamp truncates it,
 * and its warning is the observable signal.
 */
export async function splitUnitToFit<T extends ContainmentUnit>(
  unit: T,
  budget: ContainmentBudget
): Promise<T[]> {
  const boundaries = graphemeBoundaries(unit.text)
  const pieces: T[] = []
  let startBoundary = 0

  while (startBoundary < boundaries.length - 1) {
    const pieceStart = boundaries[startBoundary] ?? 0
    const remainder = unit.text.slice(pieceStart)
    const measuredTokens = await measure(remainder, budget)
    const takenGraphemes =
      measuredTokens <= budget.cap
        ? boundaries.length - 1 - startBoundary
        : await fittingPrefixGraphemes(
            {
              text: remainder,
              boundaries: boundaries.slice(startBoundary).map((offset) => offset - pieceStart),
              measuredTokens,
            },
            budget
          )
    const endBoundary = startBoundary + takenGraphemes
    const pieceEnd = boundaries[endBoundary] ?? unit.text.length
    pieces.push({
      ...unit,
      text: unit.text.slice(pieceStart, pieceEnd),
      sourceStart: unit.sourceStart + pieceStart,
      sourceEnd: unit.sourceStart + pieceEnd,
    })
    startBoundary = endBoundary
  }

  return pieces
}

/**
 * Stage C: divide a group into consecutive runs of whole units whose joined text
 * measures at or below the cap.
 *
 * A one-unit run is final and is returned as measured, even above the cap, because
 * stage A already reduced that unit as far as a valid Unicode cut allows; this
 * function never re-enters unit-internal splitting. Each division strictly reduces
 * the number of units per run, so progress does not depend on how dense the text is.
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
