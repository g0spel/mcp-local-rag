// Grapheme-bounded prefix search and measured run splitting, with injected
// counters for the dense and non-monotonic ratios no shipped tokenizer offers.

import { describe, expect, it } from 'vitest'
import type { SentenceUnit } from '../sentence-splitter.js'
import {
  splitUnitsIntoFittingRuns,
  splitUnitToFit,
  type TokenCounter,
} from '../token-containment.js'

// Fixtures are built from code points so no combining mark or ZWJ is invisible in
// this source.
function codePoints(...points: readonly number[]): string {
  return String.fromCodePoint(...points)
}

/** U+1F468 ZWJ U+1F469 ZWJ U+1F467: one grapheme cluster of 8 UTF-16 code units. */
const FAMILY_EMOJI = codePoints(0x1f468, 0x200d, 0x1f469, 0x200d, 0x1f467)

// Independent oracle for grapheme boundaries: the assertions must not reuse the
// module's own segmentation to decide where a cut was allowed.
const testSegmenter = new Intl.Segmenter('und', { granularity: 'grapheme' })

function graphemeBoundariesOf(text: string): number[] {
  const boundaries = [...testSegmenter.segment(text)].map((segment) => segment.index)
  boundaries.push(text.length)
  return boundaries
}

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/

/** One token per UTF-16 code unit. */
const codeUnitCounter: TokenCounter = (texts) => Promise.resolve(texts.map((t) => t.length))

/** Three tokens per code point: the densest ratio the design has to survive. */
const denseCounter: TokenCounter = (texts) => Promise.resolve(texts.map((t) => [...t].length * 3))

/**
 * Non-monotonic in prefix length: length 6 measures 5 while the longer length 8
 * measures 4, so a search assuming monotonicity would reject a fitting prefix.
 */
const nonMonotonicCounter: TokenCounter = (texts) =>
  Promise.resolve(texts.map((t) => Math.ceil(t.length / 2) + (t.length % 3 === 0 ? 2 : 0)))

/** Four tokens for every 'X', one for every other character. */
const weightedCounter: TokenCounter = (texts) =>
  Promise.resolve(
    texts.map((t) => [...t].reduce((sum, character) => sum + (character === 'X' ? 4 : 1), 0))
  )

function unitOf(text: string, sourceStart = 0, atomic = false): SentenceUnit {
  return { text, atomic, sourceStart, sourceEnd: sourceStart + text.length }
}

function reassemble(pieces: readonly SentenceUnit[]): string {
  return pieces.map((piece) => piece.text).join('')
}

describe('splitUnitToFit', () => {
  it('should return the unit unchanged when it measures at or below the cap', async () => {
    const unit = unitOf('abcdefgh', 12, true)

    const pieces = await splitUnitToFit(unit, { cap: 8, countTokens: codeUnitCounter })

    expect(pieces).toEqual([unit])
  })

  it('should keep the full-size body and the short tail of a split unit', async () => {
    // 12 code units measure 12 against a cap of 10, so the first candidate is 10
    // graphemes, which fits and leaves a 2-character tail.
    const unit = unitOf('abcdefghijkl')

    const pieces = await splitUnitToFit(unit, { cap: 10, countTokens: codeUnitCounter })

    expect(pieces.map((piece) => piece.text)).toEqual(['abcdefghij', 'kl'])
    expect(reassemble(pieces)).toBe(unit.text)
  })

  it('should preserve every other unit property on each piece', async () => {
    const unit = unitOf('abcdefghijkl', 5, true)

    const pieces = await splitUnitToFit(unit, { cap: 10, countTokens: codeUnitCounter })

    expect(pieces).toHaveLength(2)
    for (const piece of pieces) {
      expect(piece.atomic).toBe(true)
    }
  })

  it('should give each piece the exact source slice as its offsets', async () => {
    const source = 'prefix abcdefghijklmnopqrst suffix'
    const unit = unitOf('abcdefghijklmnopqrst', 7)

    const pieces = await splitUnitToFit(unit, { cap: 7, countTokens: codeUnitCounter })

    expect(pieces.length).toBeGreaterThan(1)
    for (const piece of pieces) {
      expect(piece.text).toBe(source.slice(piece.sourceStart, piece.sourceEnd))
    }
    expect(pieces[0]?.sourceStart).toBe(unit.sourceStart)
    expect(pieces[pieces.length - 1]?.sourceEnd).toBe(unit.sourceEnd)
    for (let i = 1; i < pieces.length; i += 1) {
      expect(pieces[i]?.sourceStart).toBe(pieces[i - 1]?.sourceEnd)
    }
  })

  it('should terminate and keep every piece within the cap under a dense counter', async () => {
    const unit = unitOf('a'.repeat(40))

    const pieces = await splitUnitToFit(unit, { cap: 3, countTokens: denseCounter })

    expect(pieces).toHaveLength(40)
    expect(reassemble(pieces)).toBe(unit.text)
    for (const piece of pieces) {
      expect(await denseCounter([piece.text])).toEqual([3])
    }
  })

  it('should keep every piece within the cap under a non-monotonic counter', async () => {
    const unit = unitOf('playingxplayingx')

    // The oracle itself is non-monotonic: the shorter prefix measures more.
    expect(await nonMonotonicCounter([unit.text.slice(0, 6)])).toEqual([5])
    expect(await nonMonotonicCounter([unit.text.slice(0, 8)])).toEqual([4])

    const pieces = await splitUnitToFit(unit, { cap: 4, countTokens: nonMonotonicCounter })

    expect(reassemble(pieces)).toBe(unit.text)
    for (const piece of pieces) {
      const [measured] = await nonMonotonicCounter([piece.text])
      expect(measured).toBeLessThanOrEqual(4)
    }
  })

  it('should halve a candidate the density underestimated until it fits', async () => {
    // 16 graphemes measure 28 tokens, so the density puts the first candidate at
    // 5, and 'XXXXa' measures 17. Where halving lands is not the contract; a
    // measured non-empty prefix within the cap is.
    const unit = unitOf(`${'X'.repeat(4)}${'a'.repeat(12)}`)

    const pieces = await splitUnitToFit(unit, { cap: 10, countTokens: weightedCounter })

    expect(pieces.length).toBeGreaterThan(1)
    for (const piece of pieces) {
      const [measured] = await weightedCounter([piece.text])
      expect(measured).toBeLessThanOrEqual(10)
    }
    const firstPiece = pieces[0]
    expect(firstPiece).toBeDefined()
    expect([...(firstPiece?.text ?? '')].length).toBeLessThan(5)
    expect(reassemble(pieces)).toBe(unit.text)
  })

  it('measures a bounded multiple of the unit, not the remainder per piece', async () => {
    // The bound is about linearity, not a specific constant: measuring the
    // remainder per piece costs length / cap times the unit's length, which was
    // about 100 times over for 100,000 characters at a cap of 510.
    const unit = unitOf('\u6f22'.repeat(20000))
    let measuredCharacters = 0
    const countingCounter: TokenCounter = (texts) => {
      for (const text of texts) {
        measuredCharacters += text.length
      }
      return codeUnitCounter(texts)
    }

    const pieces = await splitUnitToFit(unit, { cap: 500, countTokens: countingCounter })

    expect(pieces.length).toBeGreaterThan(1)
    expect(reassemble(pieces)).toBe(unit.text)
    expect(measuredCharacters).toBeLessThan(unit.text.length * 5)
  })

  // Cuts must stay near the cap whatever the density profile.
  const DENSE = '\u{20000}'
  const densityProfiles = [
    { profile: 'uniform sparse', text: 'a'.repeat(400) },
    { profile: 'uniform dense', text: DENSE.repeat(40) },
    { profile: 'dense head', text: `${DENSE}${'a'.repeat(1000)}` },
    { profile: 'dense middle', text: `${'a'.repeat(100)}${DENSE}${'a'.repeat(1000)}` },
    { profile: 'dense tail', text: `${'a'.repeat(1000)}${DENSE}` },
  ]

  it.each(densityProfiles)(
    'should size cuts from nearby text for a $profile unit',
    async ({ text }) => {
      const cap = 100
      const skewedCounter: TokenCounter = (texts) =>
        Promise.resolve(
          texts.map((piece) =>
            [...piece].reduce((sum, character) => sum + (character === DENSE ? 100000 : 1), 0)
          )
        )
      const unit = unitOf(text)
      const graphemes = graphemeBoundariesOf(text).length - 1
      const overCap = [...text].filter((character) => character === DENSE).length

      const pieces = await splitUnitToFit(unit, { cap, countTokens: skewedCounter })

      expect(reassemble(pieces)).toBe(text)
      for (const piece of pieces) {
        const [measured = 0] = await skewedCounter([piece.text])
        if (graphemeBoundariesOf(piece.text).length - 1 > 1) {
          expect(measured).toBeLessThanOrEqual(cap)
        }
      }
      // Pieces stay proportional to the text's size over the cap, plus the
      // graphemes that exceed it alone. Sizing every cut from the unit's
      // average instead leaves one piece per grapheme.
      expect(pieces.length).toBeLessThanOrEqual(Math.ceil(graphemes / cap) * 3 + 3 + overCap)
    }
  )

  it('should emit an oversized single grapheme cluster as one piece', async () => {
    const unit = unitOf(`a${FAMILY_EMOJI}b`)

    const pieces = await splitUnitToFit(unit, { cap: 3, countTokens: denseCounter })

    expect(pieces.map((piece) => piece.text)).toEqual(['a', FAMILY_EMOJI, 'b'])
    // The exception is recorded, not retried: the cluster stays whole above the cap.
    expect(await denseCounter([FAMILY_EMOJI])).toEqual([15])
    expect(reassemble(pieces)).toBe(unit.text)
  })

  it('should emit a unit that is one oversized grapheme as a single piece', async () => {
    const unit = unitOf(FAMILY_EMOJI)

    const pieces = await splitUnitToFit(unit, { cap: 1, countTokens: denseCounter })

    expect(pieces).toEqual([unit])
  })

  describe('Unicode integrity', () => {
    // Each cap makes the ratio's first candidate land inside a cluster when cut
    // positions are code-unit indices, so a cut that ignores grapheme boundaries
    // fails these cases instead of passing by coincidence.
    const cases = [
      {
        name: 'astral CJK',
        text: codePoints(0x20000, 0x20001, 0x20002, 0x20003, 0x20004, 0x20005),
        cap: 7,
      },
      { name: 'ZWJ emoji', text: FAMILY_EMOJI.repeat(3), cap: 9 },
      { name: 'Thai combining sequences', text: codePoints(0x0e01, 0x0e49).repeat(5), cap: 7 },
      { name: 'Devanagari vowel signs', text: codePoints(0x0915, 0x093f).repeat(5), cap: 7 },
      {
        name: 'Devanagari conjuncts',
        text: codePoints(0x0915, 0x094d, 0x0937, 0x093f).repeat(3),
        cap: 7,
      },
    ]

    const unitStart = 3

    it.each(cases)('should keep $name intact across every piece', async ({ text, cap }) => {
      const unit = unitOf(text, unitStart)
      const boundaries = graphemeBoundariesOf(text)

      const pieces = await splitUnitToFit(unit, { cap, countTokens: codeUnitCounter })

      expect(pieces.length).toBeGreaterThan(1)
      expect(reassemble(pieces)).toBe(text)
      for (const piece of pieces) {
        expect(piece.text).not.toMatch(LONE_SURROGATE)
        expect(piece.text).toBe(
          text.slice(piece.sourceStart - unitStart, piece.sourceEnd - unitStart)
        )
        expect(boundaries).toContain(piece.sourceStart - unitStart)
        expect(boundaries).toContain(piece.sourceEnd - unitStart)
        const [measured] = await codeUnitCounter([piece.text])
        expect(measured).toBeLessThanOrEqual(cap)
      }
    })
  })
})

describe('splitUnitsIntoFittingRuns', () => {
  const joinUnits = (units: readonly SentenceUnit[]): string =>
    units.map((unit) => unit.text).join(' ')

  function unitsOf(texts: readonly string[]): SentenceUnit[] {
    const units: SentenceUnit[] = []
    let cursor = 0
    for (const text of texts) {
      units.push(unitOf(text, cursor))
      cursor += text.length + 1
    }
    return units
  }

  it('should return no runs for an empty group', async () => {
    const runs = await splitUnitsIntoFittingRuns([], {
      cap: 10,
      countTokens: codeUnitCounter,
      joinUnits,
    })

    expect(runs).toEqual([])
  })

  it('should return one run when the joined group measures at or below the cap', async () => {
    const units = unitsOf(['abc', 'def'])

    const runs = await splitUnitsIntoFittingRuns(units, {
      cap: 10,
      countTokens: codeUnitCounter,
      joinUnits,
    })

    expect(runs).toEqual([units])
  })

  it('should split an oversized group into consecutive runs of whole units', async () => {
    const units = unitsOf(['aaaa', 'bbbb', 'cccc', 'dddd'])

    const runs = await splitUnitsIntoFittingRuns(units, {
      cap: 9,
      countTokens: codeUnitCounter,
      joinUnits,
    })

    expect(runs).toEqual([
      [units[0], units[1]],
      [units[2], units[3]],
    ])
    for (const run of runs) {
      const [measured] = await codeUnitCounter([joinUnits(run)])
      expect(measured).toBeLessThanOrEqual(9)
    }
  })

  it('should keep a one-unit run final even when it exceeds the cap', async () => {
    const units = unitsOf(['aaaaaaaa', 'bb'])

    const runs = await splitUnitsIntoFittingRuns(units, {
      cap: 4,
      countTokens: codeUnitCounter,
      joinUnits,
    })

    expect(runs).toEqual([[units[0]], [units[1]]])
    expect(await codeUnitCounter([joinUnits(runs[0] ?? [])])).toEqual([8])
  })

  it('should terminate under a dense counter with one unit per run', async () => {
    const units = unitsOf(['a', 'b', 'c', 'd', 'e'])

    const runs = await splitUnitsIntoFittingRuns(units, {
      cap: 3,
      countTokens: denseCounter,
      joinUnits,
    })

    expect(runs).toEqual(units.map((unit) => [unit]))
  })
})
