import { describe, expect, it } from 'vitest'
import { privateMembers } from '../../__tests__/test-doubles.js'
import type { EmbeddingRole, PipelineTokenizer } from '../index.js'
import { Embedder, installTokenLimitClamp } from '../index.js'
import type { ModelPrompts } from '../sentence-transformers-config.js'

const PROMPTS: ModelPrompts = { query: 'Q: ', document: 'D: ', default: 'DEF: ' }

/** An initialized embedder over a fake pipeline that records what it measures and embeds. */
function createEmbedder(): { embedder: Embedder; measured: string[][]; embedded: string[][] } {
  const measured: string[][] = []
  const embedded: string[][] = []
  const tokenizer = Object.assign(
    (texts: string[]) => {
      measured.push([...texts])
      return { input_ids: texts.map(() => [1]) }
    },
    { model_max_length: 512 }
  )
  const pipeline = Object.assign(
    async (texts: string[]) => {
      embedded.push([...texts])
      return { data: new Float32Array(texts.length), dims: [texts.length, 1] }
    },
    { tokenizer }
  )

  const embedder = new Embedder({ modelPath: 'unused', batchSize: 8, cacheDir: 'unused' })
  const clamp = installTokenLimitClamp(pipeline)
  const members = privateMembers<{
    model: unknown
    tokenLimit: number | null
    measurementTokenizer: PipelineTokenizer | null
    prompts: ModelPrompts
  }>(embedder)
  members.model = pipeline
  members.tokenLimit = clamp.tokenLimit
  members.measurementTokenizer = clamp.measurementTokenizer
  members.prompts = PROMPTS

  return { embedder, measured, embedded }
}

describe('Embedder model prompts', () => {
  it.each<[EmbeddingRole | undefined, string]>([
    ['query', 'Q: text'],
    ['document', 'D: text'],
    [undefined, 'DEF: text'],
  ])('measures and embeds the same %s-prompted string', async (role, expected) => {
    const { embedder, measured, embedded } = createEmbedder()

    await embedder.embedBatch(['text'], role)

    expect(measured).toEqual([[expected]])
    expect(embedded).toEqual([[expected]])
  })

  it('counts tokens behind the prompt for the requested role', async () => {
    const { embedder, measured } = createEmbedder()

    await embedder.countTokens(['text'], 'document')

    expect(measured).toEqual([['D: text']])
  })
})
