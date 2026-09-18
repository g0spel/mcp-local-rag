// The #202 regression against the real tokenizer and onnxruntime (AC-001).
//
// A double cannot show that the clamp holds against the real pipeline, so the
// cached default model is loaded, put back into its pre-clamp state and given
// `Xenova/bge-large-zh-v1.5`'s `1e30` sentinel, reproducing the shape that
// crashes in #202. The other embedder criteria run on injected fakes.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { getTestDevice, testModelCacheDir } from '../../__tests__/test-device.js'
import { expectDefined, privateMembers } from '../../__tests__/test-doubles.js'
import type { EmbedderConfig, PipelineTokenizer, TokenLimitClamp } from '../index.js'
import { Embedder, installTokenLimitClamp } from '../index.js'

/** 900 CJK characters, which the default tokenizer measures as 902 tokens. */
const OVERSIZED_CJK_TEXT = '漢'.repeat(900)
const MEASURED_CJK_TOKENS = 902
/** `Xenova/bge-large-zh-v1.5`'s reported `model_max_length`. */
const SENTINEL_MODEL_MAX_LENGTH = 1e30
/** The model's 512 positions on the window-only branch, less the two reserved. */
const RESOLVED_CAP = 510
const EMBEDDING_DIMENSIONS = 384
const TRUNCATION_WARNING = /input exceeds the model token limit/

interface LoadedPipeline {
  (input: string[], options: unknown): Promise<unknown>
  tokenizer: PipelineTokenizer
}

interface EmbedderInternals {
  model: LoadedPipeline
  tokenLimit: number | null
  measurementTokenizer: PipelineTokenizer | null
}

function defaultModelConfig(): EmbedderConfig {
  return {
    modelPath: 'Xenova/all-MiniLM-L6-v2',
    batchSize: 8,
    cacheDir: testModelCacheDir(),
    device: getTestDevice(),
  }
}

/**
 * The cached default model in the state a sentinel-reporting model loads in:
 * the original tokenizer back on the pipeline, reporting `1e30`.
 */
async function loadSentinelReportingPipeline(): Promise<LoadedPipeline> {
  const loader = new Embedder(defaultModelConfig())
  await loader.initialize()

  const internals = privateMembers<EmbedderInternals>(loader)
  const loaded = internals.model
  // `measurementTokenizer` is the reference the pipeline carried before
  // `initialize()` installed the clamp, so restoring it undoes the clamp.
  loaded.tokenizer = expectDefined(internals.measurementTokenizer)
  Object.defineProperty(loaded.tokenizer, 'model_max_length', {
    get: () => SENTINEL_MODEL_MAX_LENGTH,
    configurable: true,
  })
  return loaded
}

/** An embedder handed the clamped pipeline exactly as `initialize()` leaves it. */
function embedderWithClampedPipeline(pipeline: LoadedPipeline, clamp: TokenLimitClamp): Embedder {
  const embedder = new Embedder(defaultModelConfig())
  const internals = privateMembers<EmbedderInternals>(embedder)
  internals.model = pipeline
  internals.tokenLimit = clamp.tokenLimit
  internals.measurementTokenizer = clamp.measurementTokenizer
  return embedder
}

describe('Embedder position-window clamp on the real pipeline', () => {
  let sabotagedTokenizer: PipelineTokenizer | null = null

  afterEach(() => {
    if (sabotagedTokenizer !== null) {
      // Drop the shadowing accessor so the real model's own reported limit is
      // visible again to any later suite in this process (`isolate: false`).
      Reflect.deleteProperty(sabotagedTokenizer, 'model_max_length')
      sabotagedTokenizer = null
    }
    vi.restoreAllMocks()
  })

  it('embeds a 900-character CJK text that the unclamped sentinel pipeline cannot', async () => {
    const stderr = vi.spyOn(console, 'error').mockImplementation(() => {})
    const pipeline = await loadSentinelReportingPipeline()
    sabotagedTokenizer = pipeline.tokenizer

    // The #202 failure path: with the sentinel nothing truncates, so more
    // positions are sent than the model has position embeddings.
    await expect(
      pipeline([OVERSIZED_CJK_TEXT], { pooling: 'mean', normalize: true })
    ).rejects.toThrow(/Add node\. Name:'\/embeddings\/Add_1'/)

    const clamp = installTokenLimitClamp(pipeline)
    expect(clamp.tokenLimit).toBe(RESOLVED_CAP)

    const embedding = await embedderWithClampedPipeline(pipeline, clamp).embed(OVERSIZED_CJK_TEXT)

    expect(embedding).toHaveLength(EMBEDDING_DIMENSIONS)
    expect(embedding.every((value) => Number.isFinite(value))).toBe(true)
    const warnings = stderr.mock.calls
      .map((call) => String(call[0]))
      .filter((message) => TRUNCATION_WARNING.test(message))
    expect(warnings).toHaveLength(1)
    expect(expectDefined(warnings[0])).toContain(String(RESOLVED_CAP))
    expect(expectDefined(warnings[0])).toContain(String(MEASURED_CJK_TOKENS))
  }, 180000)
})
