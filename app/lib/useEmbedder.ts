"use client";
// Client-side embedder hook. Loads the model on demand and exposes a stable
// `embedText(texts)` function. State machine: idle → loading → ready/error.
//
// The embed-browser module imports a platform-specific bundle
// (transformers.web.js) that calls `new URL(...)` at evaluation time. We
// load it via a dynamic import so the SSR/build pipeline never executes
// it — `useEmbedder` only ever runs on the client.
import { useCallback, useEffect, useRef, useState } from "react";
import type { Embedder, EmbeddedMatrix } from "./types";

type Status = "idle" | "loading" | "ready" | "error";

type BrowserEmbed = typeof import("./embed-browser");

export function useEmbedder(modelId: string, maxLength: number | null) {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [extractor, setExtractor] = useState<Embedder | null>(null);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const loadToken = useRef(0);
  // Cached module handle for the embed wrapper. Loaded once, on demand.
  const browserEmbed = useRef<BrowserEmbed | null>(null);

  // Reload when modelId changes (e.g. user picks a different model).
  useEffect(() => {
    const myToken = ++loadToken.current;
    setStatus("loading");
    setError(null);
    setExtractor(null);
    setActiveModel(null);
    (async () => {
      try {
        // Platform-specific exception: transformers.web.js is browser-only.
        if (!browserEmbed.current) {
          browserEmbed.current = await import("./embed-browser");
        }
        const ext = await browserEmbed.current.getExtractor(modelId);
        if (myToken !== loadToken.current) return;
        // Wrap in a thunk: transformers.js pipelines are Callable, and React's
        // basicStateReducer treats any function passed to a setter as the
        // (prev) => next updater form. Passing the pipeline directly would
        // make React call it on every render and blow up via the `_call`
        // guard in embed-browser.ts.
        setExtractor(() => ext);
        setActiveModel(modelId);
        setStatus("ready");
      } catch (e) {
        if (myToken !== loadToken.current) return;
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      }
    })();
    return () => {
      loadToken.current++;
    };
  }, [modelId]);

  const embedText = useCallback(
    async (texts: string[]): Promise<EmbeddedMatrix> => {
      if (!extractor) throw new Error("Model not loaded");
      if (!browserEmbed.current) {
        browserEmbed.current = await import("./embed-browser");
      }
      return await browserEmbed.current.embed(extractor, texts, {
        max_length: maxLength ?? undefined,
      });
    },
    [extractor, maxLength],
  );

  return { status, error, extractor, activeModel, embedText };
}