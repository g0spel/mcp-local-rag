// Test-wide offline guard for the local-model suites: they rely on the
// pre-warmed ./tmp/models cache (scripts/prewarm-embedder-cache.mjs). When a
// cache entry is missing, fail fast with a transformers.js load error instead
// of silently downloading (~90MB per model) mid-run.
import { env } from '@huggingface/transformers'

env.allowRemoteModels = false
