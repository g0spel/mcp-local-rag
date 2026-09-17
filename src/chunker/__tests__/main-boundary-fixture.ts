// Boundary-preservation fixtures for measured token containment.
// Purpose: pin the no-op promise of AC-011 and AC-016 — content that fits the
// resolved cap, and any degraded embedder, must chunk exactly as `main` does.
//
// `latinChunks` and `denseScriptChunks` were captured by running
// `SemanticChunker.chunkText` from `main` (826eda2's merge base) over the
// documents below with `fixtureEmbeddings`, so a test comparing against them
// compares against pre-containment behavior rather than against the new
// implementation's own output.

import type { TextChunk } from '../index.js'

/**
 * Two-topic Latin document. Sentences are short enough that every unit and
 * every group measures well inside the cap the fixture tests resolve, which is
 * the precondition AC-011 states.
 */
export const latinDocument = [
  'Local retrieval keeps every document on the same machine as the index.',
  'A retrieval request reads the stored vectors and ranks them by distance.',
  'Retrieval quality depends on how the source text was divided into chunks.',
  'The weather forecast for the weekend promises a long stretch of sunshine.',
  'Weather models disagree about how much rain the valley will receive.',
  'A weather warning would change the plan for the hike along the ridge.',
].join(' ')

/**
 * The same two topics in Japanese. A dense script is the case #202 crashed on,
 * so the no-op promise is pinned for it separately from the Latin document.
 */
export const denseScriptDocument = [
  'ローカル検索はすべての文書を索引と同じ計算機に保存します。',
  '検索要求は保存されたベクトルを読み取り距離で並べ替えます。',
  '検索の品質は元の文章をどのように分割したかに依存します。',
  '週末の天気予報は長い晴天が続くと伝えています。',
  '天気のモデルは谷にどれだけ雨が降るかで一致していません。',
  '天気の警報が出れば尾根を歩く計画は変わります。',
].join('')

/** Topic markers, one per sentence in both documents, in axis order. */
const TOPIC_MARKERS: readonly string[][] = [
  ['retrieval', 'Retrieval', '検索'],
  ['weather', 'Weather', '天気'],
]

/**
 * Embeddings derived from the text itself, so the same unit always yields the
 * same vector and grouping cannot drift between the captured run and the
 * comparison run. A sentence's topic marker selects one axis; a sentence with
 * no marker gets the spare axis.
 */
export function fixtureEmbeddings(texts: string[]): number[][] {
  return texts.map((text) => {
    const axis = TOPIC_MARKERS.findIndex((markers) =>
      markers.some((marker) => text.includes(marker))
    )
    const vector = [0, 0, 0]
    vector[axis === -1 ? 2 : axis] = 1
    return vector
  })
}

/** Chunks `main` produces for {@link latinDocument}. */
export const latinChunks: TextChunk[] = [
  {
    text: 'Local retrieval keeps every document on the same machine as the index. A retrieval request reads the stored vectors and ranks them by distance. Retrieval quality depends on how the source text was divided into chunks.',
    index: 0,
    sourceStart: 0,
    sourceEnd: 217,
  },
  {
    text: 'The weather forecast for the weekend promises a long stretch of sunshine. Weather models disagree about how much rain the valley will receive. A weather warning would change the plan for the hike along the ridge.',
    index: 1,
    sourceStart: 218,
    sourceEnd: 430,
  },
]

/** Chunks `main` produces for {@link denseScriptDocument}. */
export const denseScriptChunks: TextChunk[] = [
  {
    text: 'ローカル検索はすべての文書を索引と同じ計算機に保存します。 検索要求は保存されたベクトルを読み取り距離で並べ替えます。 検索の品質は元の文章をどのように分割したかに依存します。',
    index: 0,
    sourceStart: 0,
    sourceEnd: 86,
  },
  {
    text: '週末の天気予報は長い晴天が続くと伝えています。 天気のモデルは谷にどれだけ雨が降るかで一致していません。 天気の警報が出れば尾根を歩く計画は変わります。',
    index: 1,
    sourceStart: 86,
    sourceEnd: 160,
  },
]
