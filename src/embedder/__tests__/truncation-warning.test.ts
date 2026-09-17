// Truncation reporting and degraded-mode reporting (AC-005, AC-007).
//
// Fake pipelines are handed to the instance the way `initialize()` does, so the
// warnings are observed through `embedBatch`/`getTokenLimit` without a model
// download. Mocking `@huggingface/transformers` would leak across files
// (`isolate: false`, see lazy-initialization.test.ts).

import { afterEach, describe, expect, it, vi } from 'vitest'
import { expectDefined, privateMembers } from '../../__tests__/test-doubles.js'
import type { PipelineTokenizer } from '../index.js'
import { Embedder, installTokenLimitClamp } from '../index.js'

const TRUNCATION_WARNING = /input exceeds the model token limit/
const DEGRADED_WARNING = /no position-window token limit could be resolved/

interface FakePipelineShape {
  /** Token length reported for each text, by text. */
  tokenLengths: Record<string, number>
  modelMaxLength?: number
}

function createEmbedderWithFakePipeline(shape: FakePipelineShape): {
  embedder: Embedder
  modelCalls: string[][]
} {
  const modelCalls: string[][] = []
  const tokenizer = (texts: string[]) => ({
    input_ids: texts.map((text) => Array.from({ length: shape.tokenLengths[text] ?? 1 }, () => 1)),
  })
  if (shape.modelMaxLength !== undefined) {
    Object.defineProperty(tokenizer, 'model_max_length', { get: () => shape.modelMaxLength })
  }
  const pipeline = Object.assign(
    async (texts: string[]) => {
      modelCalls.push([...texts])
      return { data: Float32Array.from(texts.map(() => 1)), dims: [texts.length, 1] }
    },
    { tokenizer }
  )

  const embedder = new Embedder({
    modelPath: 'unused-by-fake-pipeline',
    batchSize: 16,
    cacheDir: 'unused-by-fake-pipeline',
  })
  const clamp = installTokenLimitClamp(pipeline)
  const members = privateMembers<{
    model: unknown
    tokenLimit: number | null
    measurementTokenizer: PipelineTokenizer | null
  }>(embedder)
  members.model = pipeline
  members.tokenLimit = clamp.tokenLimit
  members.measurementTokenizer = clamp.measurementTokenizer
  return { embedder, modelCalls }
}

function spyOnStderr() {
  return vi.spyOn(console, 'error').mockImplementation(() => {})
}

function warningsMatching(spy: ReturnType<typeof spyOnStderr>, pattern: RegExp): string[] {
  return spy.mock.calls.map((call) => String(call[0])).filter((message) => pattern.test(message))
}

describe('Embedder truncation reporting', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('warns once per instance, names the cap and the longest length, and still embeds', async () => {
    const stderr = spyOnStderr()
    const { embedder } = createEmbedderWithFakePipeline({
      tokenLengths: { oversized: 600, short: 10 },
      modelMaxLength: 512,
    })

    const first = await embedder.embedBatch(['oversized', 'short'])
    await embedder.embedBatch(['oversized'])

    const warnings = warningsMatching(stderr, TRUNCATION_WARNING)
    expect(warnings).toHaveLength(1)
    expect(expectDefined(warnings[0])).toContain('512')
    expect(expectDefined(warnings[0])).toContain('600')
    expect(first).toEqual([[1], [1]])
  })

  it('does not warn for an input exactly at the cap', async () => {
    const stderr = spyOnStderr()
    const { embedder } = createEmbedderWithFakePipeline({
      tokenLengths: { 'at-the-cap': 512 },
      modelMaxLength: 512,
    })

    await embedder.embedBatch(['at-the-cap'])

    expect(warningsMatching(stderr, TRUNCATION_WARNING)).toEqual([])
  })

  it('plans batches on the clamped length, so an over-cap input is not deferred alone', async () => {
    const stderr = spyOnStderr()
    // True lengths 600 and 5000 look like a padding outlier and would be split;
    // clamped to the cap both are 512, which is the work the model actually does.
    const { embedder, modelCalls } = createEmbedderWithFakePipeline({
      tokenLengths: { 'over-cap-a': 600, 'over-cap-b': 5000 },
      modelMaxLength: 512,
    })

    const embeddings = await embedder.embedBatch(['over-cap-a', 'over-cap-b'])

    expect(modelCalls).toEqual([['over-cap-a', 'over-cap-b']])
    expect(embeddings).toEqual([[1], [1]])
    expect(warningsMatching(stderr, TRUNCATION_WARNING)).toHaveLength(1)
  })
})

describe('Embedder degraded mode reporting', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('warns once that no limit could be resolved and leaves embedding unchanged', async () => {
    const stderr = spyOnStderr()
    // The 1e30 sentinel with no position window: no cap is resolvable.
    const { embedder } = createEmbedderWithFakePipeline({
      tokenLengths: { unbounded: 5000 },
      modelMaxLength: 1e30,
    })

    expect(await embedder.getTokenLimit()).toBeNull()
    const embeddings = await embedder.embedBatch(['unbounded'])
    await embedder.embedBatch(['unbounded'])

    expect(warningsMatching(stderr, DEGRADED_WARNING)).toHaveLength(1)
    // No cap means nothing to exceed, so no truncation is reported.
    expect(warningsMatching(stderr, TRUNCATION_WARNING)).toEqual([])
    expect(embeddings).toEqual([[1]])
  })

  it('warns once and resolves no cap for an unrecognized pipeline shape', async () => {
    const stderr = spyOnStderr()
    const embedder = new Embedder({
      modelPath: 'unused-by-fake-pipeline',
      batchSize: 16,
      cacheDir: 'unused-by-fake-pipeline',
    })
    privateMembers<{ model: unknown }>(embedder).model = { tokenizer: 'not-callable' }

    expect(await embedder.getTokenLimit()).toBeNull()
    expect(await embedder.getTokenLimit()).toBeNull()

    expect(warningsMatching(stderr, DEGRADED_WARNING)).toHaveLength(1)
  })
})
