"use client";
// Browser-side wrapper around @huggingface/transformers. Kept separate from
// the repo-root embed.js (which the bench uses) because the bundler picks
// the Node-specific entry of the transformers package for the client build,
// which then resolves WASM paths that don't exist on the web.
//
// We import the browser entry file directly to bypass the package.json
// "node" condition. transformers.web.js is the UMD-style browser bundle
// exposed by @huggingface/transformers.
// @ts-expect-error — no .d.ts for the deep browser bundle path.
import * as transformers from "../../node_modules/@huggingface/transformers/dist/transformers.web.js";
import type { EmbeddedMatrix, Embedder } from "./types";

type TransformersMod = typeof transformers & {
  env: typeof transformers.env & {
    allowRemoteModels: boolean;
    allowLocalModels: boolean;
    useBrowserCache: boolean;
  };
  pipeline: (
    task: "feature-extraction",
    model: string,
    opts?: { quantized?: boolean; dtype?: string },
  ) => Promise<Embedder>;
  Tensor: any;
};

const t = transformers as TransformersMod;

t.env.allowRemoteModels = true;
t.env.allowLocalModels = false;
t.env.useBrowserCache = true;

const cache = new Map<string, Embedder>();

export async function getExtractor(
  modelId: string,
  opts: { quantized?: boolean } = {},
): Promise<Embedder> {
  const cached = cache.get(modelId);
  if (cached) return cached;
  // Call the transformers.js factory. The returned object is a Callable
  // pipeline with `tokenizer` and `model` properties. Calling the object
  // as a function (its `__call__`) would invoke `_call(texts, opts)` which
  // tokenizes the first argument. We never want that — we drive tokenizer
  // and model manually. Override `_call` so the Callable's `__call__`
  // throws a clear error if anything in the React tree accidentally
  // invokes the pipeline as a function.
  const ext = (await t.pipeline("feature-extraction", modelId, {
    quantized: opts.quantized ?? true,
  })) as unknown as Embedder & { _call?: (...args: unknown[]) => unknown };
  ext._call = () => {
    throw new TypeError(
      "Pipeline is not callable. Use .tokenizer(...) or .model(...) instead.",
    );
  };
  cache.set(modelId, ext as Embedder);
  return ext as Embedder;
}

// Embed a list of strings. Returns Float32Array of length n*d (row-major), with metadata ._n ._d.
//
// normalize=false: skip the L2-normalize step (PCA centers the data anyway).
// Tokenization is cached per-text (deterministic), so re-embedding the same
// strings skips the tokenizer. Model call is unchanged.
//
// max_length: optional integer cap on tokenized sequence length. The transformer
// is O(seq^2) in attention, so shorter sequences are dramatically faster.
export async function embed(
  extractor: Embedder,
  texts: string[],
  opts: { max_length?: number } = {},
): Promise<EmbeddedMatrix> {
  const n = texts.length;
  if (n === 0) {
    const e = new Float32Array(0) as EmbeddedMatrix;
    e._n = 0;
    e._d = 0;
    return e;
  }

  // Build Tensor inputs directly. We skip the pipeline factory's mean-pool
  // and post-processing so we control it and can reuse the tokenization
  // cache.
  const idsList: BigInt64Array[] = [];
  const maskList: BigInt64Array[] = [];
  const tokenCache = new Map<string, { ids: BigInt64Array; mask: BigInt64Array }>();
  const misses: string[] = [];
  const missIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = texts[i];
    const hit = tokenCache.get(t);
    if (hit && (opts.max_length == null || hit.ids.length <= opts.max_length)) {
      idsList[i] = hit.ids;
      maskList[i] = hit.mask;
    } else {
      missIdx.push(i);
      misses.push(t);
    }
  }
  if (misses.length) {
    // Defensive: some build / state paths in Next.js can leave an undefined or
    // null entry in the input array (React's basicStateReducer + a stale
    // closure was the original culprit). Filter them out so the tokenizer
    // never sees a null.
    const cleanMisses: string[] = [];
    const remap: number[] = [];
    for (let k = 0; k < misses.length; k++) {
      const t = misses[k];
      if (typeof t === "string" && t.length > 0) {
        cleanMisses.push(t);
        remap.push(missIdx[k]);
      } else {
        // Drop the bad entry from output slots so the indices stay aligned.
        const slot = missIdx[k];
        idsList[slot] = new BigInt64Array(0);
        maskList[slot] = new BigInt64Array(0);
      }
    }
    if (cleanMisses.length) {
      const tokOpts: { padding: boolean | string; truncation: boolean; max_length?: number } = {
        padding: true,
        truncation: true,
      };
      if (opts.max_length != null) tokOpts.max_length = opts.max_length;
      const out = (await extractor.tokenizer(cleanMisses, tokOpts)) as unknown as {
        input_ids: { data: BigInt64Array; dims: number[] };
        attention_mask: { data: BigInt64Array; dims: number[] };
      };
      const idsData = out.input_ids.data;
      const maskData = out.attention_mask.data;
      const B = cleanMisses.length;
      const S = out.input_ids.dims[1];
      for (let k = 0; k < B; k++) {
        const base = k * S;
        let realEnd = S;
        for (let j = S - 1; j >= 0; j--) {
          if (maskData[base + j]) {
            realEnd = j + 1;
            break;
          }
        }
        const ids = idsData.slice(base, base + realEnd);
        const mask = maskData.slice(base, base + realEnd);
        const slot = remap[k];
        idsList[slot] = ids;
        maskList[slot] = mask;
        tokenCache.set(cleanMisses[k], { ids, mask });
      }
    }
  }

  let maxLen = 0;
  for (let i = 0; i < n; i++) if (idsList[i].length > maxLen) maxLen = idsList[i].length;
  const S = maxLen;
  const ids = new BigInt64Array(n * S);
  const mask = new BigInt64Array(n * S);
  for (let i = 0; i < n; i++) {
    const a = idsList[i];
    const b = maskList[i];
    const s = a.length;
    for (let j = 0; j < s; j++) {
      ids[i * S + j] = a[j];
      mask[i * S + j] = b[j];
    }
  }

  // As unknown as Record<string, unknown>: the library accepts a plain
  // object with input_ids/attention_mask; the Tensor constructor is loaded
  // from the same module to keep the API aligned.
  const Tensor = t.Tensor as new (type: string, data: BigInt64Array, dims: number[]) => unknown;
  const inputs = {
    input_ids: new Tensor("int64", ids, [n, S]),
    attention_mask: new Tensor("int64", mask, [n, S]),
  };
  const result = (await extractor.model(inputs as unknown as Record<string, never>)) as {
    last_hidden_state?: { data: Float32Array; dims: number[] };
    logits?: { data: Float32Array; dims: number[] };
    token_embeddings?: { data: Float32Array; dims: number[] };
  };
  const lh = result.last_hidden_state ?? result.logits ?? result.token_embeddings;
  if (!lh) throw new Error("Model output missing hidden state");
  const d = lh.dims[lh.dims.length - 1];
  const out = new Float32Array(n * d) as EmbeddedMatrix;
  out._n = n;
  out._d = d;

  const data = lh.data;
  const D4 = d & ~3;
  for (let i = 0; i < n; i++) {
    const ml = maskList[i];
    const sL = ml.length;
    let count = 0;
    for (let j = 0; j < sL; j++) if (ml[j]) count++;
    if (count === 0) continue;
    const rowBase = i * S * d;
    const dst = i * d;
    for (let j = 0; j < sL; j++) {
      if (!ml[j]) continue;
      const offset = rowBase + j * d;
      for (let k = 0; k < D4; k += 4) {
        out[dst + k]     += data[offset + k];
        out[dst + k + 1] += data[offset + k + 1];
        out[dst + k + 2] += data[offset + k + 2];
        out[dst + k + 3] += data[offset + k + 3];
      }
      for (let k = D4; k < d; k++) out[dst + k] += data[offset + k];
    }
    const inv = 1 / count;
    for (let k = 0; k < D4; k += 4) {
      out[dst + k]     *= inv;
      out[dst + k + 1] *= inv;
      out[dst + k + 2] *= inv;
      out[dst + k + 3] *= inv;
    }
    for (let k = D4; k < d; k++) out[dst + k] *= inv;
  }
  return out;
}