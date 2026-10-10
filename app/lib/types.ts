// Domain types for the vanilla-JS modules at the repo root. We type them as
// minimal interfaces so the React layer stays strictly typed without reaching
// for `any` at the boundary.
import type { Tensor } from "@huggingface/transformers";

export interface Embedder {
  tokenizer: (
    texts: string | string[],
    opts?: { padding?: boolean | string; truncation?: boolean; max_length?: number },
  ) => Promise<{
    input_ids: { data: BigInt64Array; dims: number[] };
    attention_mask: { data: BigInt64Array; dims: number[] };
  }>;
  model: (
    inputs: Record<string, Tensor>,
  ) => Promise<{
    last_hidden_state?: { data: Float32Array; dims: number[] };
    logits?: { data: Float32Array; dims: number[] };
    token_embeddings?: { data: Float32Array; dims: number[] };
  }>;
}

// The shape our PCA module returns.
export interface PcaResult {
  points: Float32Array;
  components: Float32Array;
  mean: Float32Array;
  varianceExplained: number[];
  rank: number;
  n: number;
  d: number;
}

// The shape of a typed (n*d row-major) Float32Array with metadata, as returned
// by embed(). The vanilla embed() attaches _n and _d to a typed array.
export interface EmbeddedMatrix extends Float32Array {
  _n: number;
  _d: number;
}