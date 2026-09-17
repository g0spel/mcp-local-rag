// Measured containment against the real tokenizer (AC-008, AC-009).
//
// The rest of the containment criteria run on injected counters
// (token-containment.test.ts, semantic-chunker.test.ts), which is cheaper and
// stronger for structural properties. This case exists for the one thing a
// counter cannot show: that the chunker's measurement is wired to the model
// that will embed the text. It loads the cached default model and asserts the
// overflow **before** containment as well as the containment after, so it fails
// if that wiring is removed rather than passing on a fixture no tokenizer
// considers long.
//
// No module factory is registered for `@huggingface/transformers`: mocking it
// would leak across files (`isolate: false`, see lazy-initialization.test.ts)
// and would lose the real-model coverage this case exists for.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { getTestDevice, testModelCacheDir } from '../../__tests__/test-device.js'
import { expectDefined } from '../../__tests__/test-doubles.js'
import { Embedder } from '../../embedder/index.js'
import { SemanticChunker } from '../semantic-chunker.js'
import { splitIntoSentenceUnits } from '../sentence-splitter.js'

/**
 * A CJK paragraph with no sentence terminator, so the splitter keeps it as one
 * unit that overflows the cap on its own. Ideographs only: the default English
 * tokenizer splits them per character, while hiragana falls outside its
 * vocabulary and collapses to a single `[UNK]`, which would leave nothing to
 * contain. Repeating a 61-character phrase keeps every character well under
 * `isGarbageChunk`'s 80% repetition bound, so the fixture is stored content.
 */
const DENSE_PHRASE =
  '检索增强生成系统在处理密集文字时会遇到位置窗口限制的问题因此需要以真实词元度量为依据的容纳策略而不是基于字符数量的粗略估计'
const CJK_SENTENCE =
  '本项目的分块流程先按语义边界切分文本再度量真实词元数量然后仅对超出模型位置窗口的部分进行切分以保证内容完整可检索。'
const CJK_DOCUMENT = `${DENSE_PHRASE.repeat(10)}${CJK_SENTENCE.repeat(3)}`

const TRUNCATION_WARNING = /input exceeds the model token limit/

function defaultModelEmbedder(): Embedder {
  return new Embedder({
    modelPath: 'Xenova/all-MiniLM-L6-v2',
    batchSize: 8,
    cacheDir: testModelCacheDir(),
    device: getTestDevice(),
  })
}

/** Whitespace is dropped because units are joined with a single space while the
 * source holds none between CJK sentences (`semantic-chunker.ts` joinUnits). */
function withoutWhitespace(text: string): string {
  return text.replace(/\s+/g, '')
}

describe('Measured containment on the real tokenizer', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('contains a CJK document that overflows the cap before chunking', async () => {
    const stderr = vi.spyOn(console, 'error').mockImplementation(() => {})
    const embedder = defaultModelEmbedder()
    const cap = expectDefined(await embedder.getTokenLimit())

    // Before containment: the units `chunkText` will measure, measured the same
    // way. Without this the case would pass with no containment code at all.
    const units = splitIntoSentenceUnits(CJK_DOCUMENT, [])
    const unitLengths = await embedder.countTokens(units.map((unit) => unit.text))
    expect(Math.max(...unitLengths)).toBeGreaterThan(cap)

    const chunks = await new SemanticChunker().chunkText(CJK_DOCUMENT, embedder)

    // After containment: every stored chunk fits the window the model has.
    expect(chunks.length).toBeGreaterThan(1)
    const chunkLengths = await embedder.countTokens(chunks.map((chunk) => chunk.text))
    for (const length of chunkLengths) {
      expect(length).toBeLessThanOrEqual(cap)
    }
    // Nothing the garbage filter would reject is present, so the whole source
    // is covered by the stored chunks, in order.
    expect(withoutWhitespace(chunks.map((chunk) => chunk.text).join(''))).toBe(
      withoutWhitespace(CJK_DOCUMENT)
    )
    expect(chunks.map((chunk) => chunk.index)).toEqual(chunks.map((_, index) => index))
    expect(chunks[0]?.sourceStart).toBe(0)
    expect(chunks[chunks.length - 1]?.sourceEnd).toBe(CJK_DOCUMENT.length)

    // AC-009: containment happened instead of truncation, so the embedder never
    // saw an input above the cap and reported nothing.
    const warnings = stderr.mock.calls
      .map((call) => String(call[0]))
      .filter((message) => TRUNCATION_WARNING.test(message))
    expect(warnings).toEqual([])
  }, 180000)
})
