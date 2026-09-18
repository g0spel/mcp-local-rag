// Semantic Chunker implementation using Max-Min algorithm
// Based on: "Max–Min semantic chunking of documents for RAG application" (Springer, 2025)

import type { AtomicTextRange, TextChunk } from './index.js'
import { type SentenceUnit, splitIntoSentenceUnits } from './sentence-splitter.js'
import {
  type ContainmentBudget,
  splitUnitsIntoFittingRuns,
  splitUnitToFit,
} from './token-containment.js'

// ============================================
// Type Definitions
// ============================================

/**
 * Semantic Chunker configuration
 * Based on paper recommendations: hardThreshold=0.6, initConst=1.5, c=0.9
 */
export interface SemanticChunkerConfig {
  /** Hard threshold for minimum similarity (default: 0.6) */
  hardThreshold: number
  /** Initial constant for first sentence pair (default: 1.5) */
  initConst: number
  /** Scaling constant for threshold calculation (default: 0.9) */
  c: number
  /** Minimum chunk length in characters (default: 50) */
  minChunkLength: number
}

/**
 * Embedder interface for generating embeddings.
 *
 * `getTokenLimit` and `countTokens` are optional; without them the chunker
 * skips token containment.
 */
export interface EmbedderInterface {
  embedBatch(texts: string[]): Promise<number[][]>
  /** Resolved token cap, or `null` when no limit could be resolved. */
  getTokenLimit?(): Promise<number | null>
  /** True, unclamped token lengths of each text. */
  countTokens?(texts: string[]): Promise<number[]>
}

// ============================================
// Performance Optimization Constants
// ============================================

/**
 * Number of recent sentences to compare in getMinSimilarity.
 * Based on Max-Min paper's experimental conditions (median 5 sentences per chunk).
 * Reduces complexity from O(k²) to O(WINDOW_SIZE²) = O(25) = O(1).
 */
const WINDOW_SIZE = 5

/**
 * Maximum number of sentences per chunk before forced split.
 * Safety limit to prevent computational explosion on homogeneous documents.
 * Set to 3x the paper's median chunk size for reasonable margin.
 */
const MAX_SENTENCES = 15

/**
 * Garbage chunks, language-agnostically: empty after trimming, only decoration
 * characters (`----`, `====`), or one character repeated over 80% of the text.
 * Anything alphanumeric is kept. Applied after the minChunkLength filter.
 */
export function isGarbageChunk(text: string): boolean {
  const trimmed = text.trim()
  if (trimmed.length === 0) {
    return true
  }

  // If contains any alphanumeric, consider valid content
  if (/[a-zA-Z0-9]/.test(trimmed)) {
    return false
  }

  // Decoration line patterns only (----, ====, ****, etc.)
  if (/^[-=_.*#|~`@!%^&*()[\]{}\\/<>:+\s]+$/.test(trimmed)) {
    return true
  }

  // Excessive repetition of single character (>80%)
  const charCounts = new Map<string, number>()
  for (const char of trimmed) {
    charCounts.set(char, (charCounts.get(char) ?? 0) + 1)
  }
  const maxCount = Math.max(...charCounts.values())
  if (maxCount / trimmed.length > 0.8) {
    return true
  }

  return false
}

// ============================================
// Default Configuration
// ============================================

/** Default minimum chunk length in characters */
export const DEFAULT_MIN_CHUNK_LENGTH = 50

// ============================================
// Token Containment (stages A and C)
// ============================================

/** Joins a group's units into the chunk text, which is also what containment measures. */
function joinUnits(units: readonly SentenceUnit[]): string {
  return units.map((unit) => unit.text).join(' ')
}

/**
 * The embedder's token budget, or `null` when the optional members are absent
 * or no limit could be resolved. Containment is then skipped entirely.
 */
async function resolveContainmentBudget(
  embedder: EmbedderInterface
): Promise<ContainmentBudget | null> {
  const { getTokenLimit, countTokens } = embedder
  if (!getTokenLimit || !countTokens) {
    return null
  }
  const cap = await getTokenLimit.call(embedder)
  if (cap === null) {
    return null
  }
  return { cap, countTokens: (texts) => countTokens.call(embedder, texts) }
}

/** One measurement pass over `texts`, rejecting a counter that drops inputs. */
async function measureAll(texts: string[], budget: ContainmentBudget): Promise<number[]> {
  const measured = await budget.countTokens(texts)
  if (measured.length !== texts.length) {
    throw new Error(
      `Token counter returned ${measured.length} measurements for ${texts.length} texts`
    )
  }
  return measured
}

/**
 * Reduce sentence units to pieces within the cap, before embedding.
 *
 * Admission is judged here, on the whole unit, and recorded on its pieces: a
 * fragment can be noise by itself — a run of one character reads as pure
 * repetition — so judging pieces would drop text the parent was admitted with.
 */
async function containUnits(
  units: SentenceUnit[],
  budget: ContainmentBudget
): Promise<SentenceUnit[]> {
  const measured = await measureAll(
    units.map((unit) => unit.text),
    budget
  )
  const contained: SentenceUnit[] = []
  for (const [index, unit] of units.entries()) {
    const unitTokens = measured[index] ?? 0
    if (unitTokens <= budget.cap) {
      contained.push(unit)
      continue
    }
    const admitted = !isGarbageChunk(unit.text)
    const pieces = await splitUnitToFit(unit, budget)
    contained.push(
      ...pieces.map((piece) => (admitted ? { ...piece, containmentSplit: true } : piece))
    )
  }
  return contained
}

/**
 * Reduce admitted groups to runs of whole units within the cap. A group that
 * fits is returned untouched, so content inside the window keeps its boundaries.
 */
async function containGroups(
  groups: SentenceUnit[][],
  budget: ContainmentBudget
): Promise<SentenceUnit[][]> {
  const measured = await measureAll(groups.map(joinUnits), budget)
  const contained: SentenceUnit[][] = []
  for (const [index, group] of groups.entries()) {
    if ((measured[index] ?? 0) <= budget.cap) {
      contained.push(group)
      continue
    }
    contained.push(...(await splitUnitsIntoFittingRuns(group, { ...budget, joinUnits })))
  }
  return contained
}

/**
 * Build the stored chunks. Offsets come from the first and last unit, never from
 * an index into the joined text: the join uses a single space while the source
 * may hold newlines or repeated whitespace.
 */
function convertPiecesToChunks(pieces: SentenceUnit[][]): TextChunk[] {
  const chunks: TextChunk[] = []
  for (const piece of pieces) {
    const firstUnit = piece[0]
    const lastUnit = piece[piece.length - 1]
    if (!firstUnit || !lastUnit) {
      continue
    }
    chunks.push({
      text: joinUnits(piece),
      index: chunks.length,
      sourceStart: firstUnit.sourceStart,
      sourceEnd: lastUnit.sourceEnd,
    })
  }
  return chunks
}

const DEFAULT_SEMANTIC_CHUNKER_CONFIG: SemanticChunkerConfig = {
  hardThreshold: 0.6,
  initConst: 1.5,
  c: 0.9,
  minChunkLength: DEFAULT_MIN_CHUNK_LENGTH,
}

// ============================================
// SemanticChunker Class
// ============================================

/**
 * Semantic chunker using the Max-Min algorithm: a sentence joins the current
 * chunk when its maximum similarity to any member exceeds the minimum
 * similarity between existing members, adjusted by the threshold.
 */
export class SemanticChunker {
  private readonly config: SemanticChunkerConfig

  constructor(config: Partial<SemanticChunkerConfig> = {}) {
    this.config = { ...DEFAULT_SEMANTIC_CHUNKER_CONFIG, ...config }
  }

  /**
   * Split text into semantically coherent chunks
   */
  async chunkText(
    text: string,
    embedder: EmbedderInterface,
    atomicRanges: readonly AtomicTextRange[] = []
  ): Promise<TextChunk[]> {
    // Handle empty input
    if (!text || text.trim().length === 0) {
      // Supplied ranges are programmer contracts and must fail fast even when
      // ordinary empty text would otherwise return before sentence splitting.
      if (atomicRanges.length > 0) {
        splitIntoSentenceUnits(text, atomicRanges)
      }
      return []
    }

    // Split into sentences
    const sentenceUnits = splitIntoSentenceUnits(text, atomicRanges)
    if (sentenceUnits.length === 0) {
      return []
    }

    const budget = await resolveContainmentBudget(embedder)
    const units = budget ? await containUnits(sentenceUnits, budget) : sentenceUnits

    // Generate embeddings for all sentences
    const embeddings = await embedder.embedBatch(units.map((unit) => unit.text))

    // Apply Max-Min algorithm to group sentences into chunks
    const sentenceGroups = this.groupSentences(units, embeddings)

    const admitted = sentenceGroups.filter((group) => this.admitsGroup(group))
    const pieces = budget ? await containGroups(admitted, budget) : admitted

    return convertPiecesToChunks(pieces)
  }

  /**
   * Whether a sentence group becomes stored chunks. A group holding a
   * containment piece is already admitted, judged before the split, so neither
   * filter re-runs on a fragment of it. Otherwise garbage is rejected and the
   * minimum length is waived for an atomic unit, which must stay whole.
   */
  private admitsGroup(group: SentenceUnit[]): boolean {
    if (group.some((unit) => unit.containmentSplit)) {
      return true
    }
    const groupText = joinUnits(group)
    if (isGarbageChunk(groupText)) {
      return false
    }
    return group.some((unit) => unit.atomic) || groupText.length >= this.config.minChunkLength
  }

  /**
   * Group sentences into chunks using Max-Min algorithm
   */
  /**
   * Whether a sentence continues the group being built: `initConst`-scaled
   * similarity while the group holds one sentence, Max-Min beyond that, and
   * never past `MAX_SENTENCES`.
   */
  private continuesGroup(embedding: number[], groupEmbeddings: number[][]): boolean {
    if (groupEmbeddings.length === 1) {
      const firstEmbedding = groupEmbeddings[0]
      if (!firstEmbedding) {
        return false
      }
      const similarity = this.cosineSimilarity(firstEmbedding, embedding)
      return this.config.initConst * similarity > this.config.hardThreshold
    }
    if (groupEmbeddings.length >= MAX_SENTENCES) {
      return false
    }
    return this.shouldAddToChunk(embedding, groupEmbeddings)
  }

  private groupSentences(sentences: SentenceUnit[], embeddings: number[][]): SentenceUnit[][] {
    if (sentences.length === 0) {
      return []
    }
    if (sentences.length === 1) {
      const sentence = sentences[0]
      return sentence ? [[sentence]] : []
    }

    const groups: SentenceUnit[][] = []
    let currentGroup: SentenceUnit[] = []
    let currentGroupEmbeddings: number[][] = []

    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i]
      const embedding = embeddings[i]

      if (!sentence || !embedding) {
        continue
      }

      if (currentGroup.length === 0) {
        // Start new group with first sentence
        currentGroup.push(sentence)
        currentGroupEmbeddings.push(embedding)
        continue
      }

      if (this.continuesGroup(embedding, currentGroupEmbeddings)) {
        currentGroup.push(sentence)
        currentGroupEmbeddings.push(embedding)
        continue
      }

      // Start new group
      groups.push([...currentGroup])
      currentGroup = [sentence]
      currentGroupEmbeddings = [embedding]
    }

    // Don't forget the last group
    if (currentGroup.length > 0) {
      groups.push(currentGroup)
    }

    return groups
  }

  /**
   * Decide if a sentence should be added to the current chunk
   * Based on Max-Min algorithm from the paper
   */
  private shouldAddToChunk(newEmbedding: number[], chunkEmbeddings: number[][]): boolean {
    // Calculate min similarity within current chunk
    const minSim = this.getMinSimilarity(chunkEmbeddings)

    // Calculate max similarity between new sentence and chunk
    const maxSim = this.getMaxSimilarity(newEmbedding, chunkEmbeddings)

    // Calculate dynamic threshold
    const threshold = this.calculateThreshold(minSim, chunkEmbeddings.length)

    return maxSim > threshold
  }

  /**
   * Minimum pairwise similarity within a chunk, over the last WINDOW_SIZE
   * sentences only. The approximation follows the Max-Min paper: recent
   * sentences are what determine coherence.
   */
  private getMinSimilarity(embeddings: number[][]): number {
    if (embeddings.length < 2) {
      return 1.0
    }

    // Only compare the last WINDOW_SIZE embeddings to reduce O(k²) to O(1)
    const startIdx = Math.max(0, embeddings.length - WINDOW_SIZE)
    const windowEmbeddings = embeddings.slice(startIdx)

    let minSim = 1.0
    for (let i = 0; i < windowEmbeddings.length; i++) {
      for (let j = i + 1; j < windowEmbeddings.length; j++) {
        const embI = windowEmbeddings[i]
        const embJ = windowEmbeddings[j]
        if (!embI || !embJ) {
          continue
        }

        const sim = this.cosineSimilarity(embI, embJ)
        if (sim < minSim) {
          minSim = sim
        }
      }
    }
    return minSim
  }

  /**
   * Get maximum similarity between a sentence and any sentence in the chunk
   */
  private getMaxSimilarity(embedding: number[], chunkEmbeddings: number[][]): number {
    let maxSim = -1.0
    for (const chunkEmb of chunkEmbeddings) {
      const sim = this.cosineSimilarity(embedding, chunkEmb)
      if (sim > maxSim) {
        maxSim = sim
      }
    }
    return maxSim
  }

  /**
   * Calculate dynamic threshold based on chunk size
   * threshold = max(c * minSim * sigmoid(|C|), hardThreshold)
   */
  private calculateThreshold(minSim: number, chunkSize: number): number {
    const sigmoidValue = this.sigmoid(chunkSize)
    const dynamicThreshold = this.config.c * minSim * sigmoidValue
    return Math.max(dynamicThreshold, this.config.hardThreshold)
  }

  /**
   * Sigmoid function
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x))
  }

  /**
   * Calculate cosine similarity between two vectors
   * Public for testing
   */
  cosineSimilarity(vec1: number[], vec2: number[]): number {
    if (vec1.length !== vec2.length || vec1.length === 0) {
      return 0
    }

    let dotProduct = 0
    let norm1 = 0
    let norm2 = 0

    for (let i = 0; i < vec1.length; i++) {
      const v1 = vec1[i] ?? 0
      const v2 = vec2[i] ?? 0
      dotProduct += v1 * v2
      norm1 += v1 * v1
      norm2 += v2 * v2
    }

    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2)
    if (denominator === 0) {
      return 0
    }

    return dotProduct / denominator
  }
}
