// Cap resolution and clamp installation (AC-002, AC-007).
//
// The fake pipelines are passed to the module-level clamp function directly.
// Mocking `@huggingface/transformers` would leak across files (`isolate: false`,
// see lazy-initialization.test.ts) and that module is reached by every suite
// importing src/embedder/index.ts.

import { describe, expect, it } from 'vitest'
import { installTokenLimitClamp } from '../index.js'

interface TokenizerOptions {
  padding?: boolean
  truncation?: boolean
  return_tensor?: boolean
  max_length?: number
}

interface FakePipelineShape {
  modelMaxLength?: number
  maxPositionEmbeddings?: number
}

function createFakePipeline(shape: FakePipelineShape) {
  const tokenizerCalls: (TokenizerOptions | undefined)[] = []
  const tokenizer = (texts: string[], options?: TokenizerOptions) => {
    tokenizerCalls.push(options)
    return { input_ids: texts.map(() => [1]) }
  }
  if (shape.modelMaxLength !== undefined) {
    Object.defineProperty(tokenizer, 'model_max_length', {
      get: () => shape.modelMaxLength,
    })
  }
  const pipeline = Object.assign(async () => ({ data: new Float32Array([1]), dims: [1, 1] }), {
    tokenizer,
    ...(shape.maxPositionEmbeddings === undefined
      ? {}
      : { model: { config: { max_position_embeddings: shape.maxPositionEmbeddings } } }),
  })
  return { pipeline, tokenizer, tokenizerCalls }
}

describe('installTokenLimitClamp', () => {
  it('takes the smaller of a usable tokenizer limit and the position window', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 8192, maxPositionEmbeddings: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(512)
  })

  it('keeps the tokenizer limit as-is for the repo default shape', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 512, maxPositionEmbeddings: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(512)
  })

  it('reserves two positions when only the position window is usable', () => {
    const { pipeline } = createFakePipeline({ maxPositionEmbeddings: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(510)
  })

  it('rejects the 1e30 sentinel and falls back to the reserved position window', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 1e30, maxPositionEmbeddings: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(510)
  })

  it('uses the tokenizer limit when no position window is reported', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(512)
  })

  it('resolves no cap and leaves the pipeline untouched when neither limit is usable', () => {
    const { pipeline, tokenizer } = createFakePipeline({ modelMaxLength: 1e30 })

    const installation = installTokenLimitClamp(pipeline)

    expect(installation.tokenLimit).toBeNull()
    expect(pipeline.tokenizer).toBe(tokenizer)
  })

  it('resolves no cap for an unrecognized pipeline shape', () => {
    const notAPipeline = { tokenizer: 'not-callable' }

    const installation = installTokenLimitClamp(notAPipeline)

    expect(installation.tokenLimit).toBeNull()
    expect(installation.measurementTokenizer).toBeNull()
    expect(notAPipeline.tokenizer).toBe('not-callable')
  })

  it('injects the resolved cap and truncation into every tokenizer call', () => {
    const { pipeline, tokenizer, tokenizerCalls } = createFakePipeline({
      maxPositionEmbeddings: 512,
    })

    installTokenLimitClamp(pipeline)
    expect(pipeline.tokenizer).not.toBe(tokenizer)
    pipeline.tokenizer(['clamped'], { padding: true, truncation: false })

    expect(tokenizerCalls).toEqual([{ padding: true, truncation: true, max_length: 510 }])
  })

  it('returns the unclamped tokenizer for measurement', () => {
    const { pipeline, tokenizer, tokenizerCalls } = createFakePipeline({
      maxPositionEmbeddings: 512,
    })

    const installation = installTokenLimitClamp(pipeline)
    expect(installation.measurementTokenizer).toBe(tokenizer)
    installation.measurementTokenizer?.(['measured'], {
      padding: false,
      truncation: false,
      return_tensor: false,
    })

    expect(tokenizerCalls).toEqual([{ padding: false, truncation: false, return_tensor: false }])
  })

  it('preserves the other tokenizer members through the clamp proxy', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 1e30, maxPositionEmbeddings: 512 })

    installTokenLimitClamp(pipeline)

    expect(Reflect.get(pipeline.tokenizer, 'model_max_length')).toBe(1e30)
  })
})
