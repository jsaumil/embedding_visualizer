"use client";
// Top-level client component. Owns all state and orchestrates the embed
// pipeline: load model -> embed dataset -> PCA -> render. The new-point
// path is debounced-pre-embedded while the user types so the explicit
// "Add new point" lands instantly (mirrors the pre-embed UX flow the bench
// measures).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { pcaReduce, pcaProject } from "../pca.js";
import {
  COLORS,
  EXAMPLE_DATASET,
  type DatasetItem,
  type Dim,
  type NewPoint,
  type ProjectedItem,
} from "./lib/state";
import type { EmbeddedMatrix, PcaResult } from "./lib/types";
import { useEmbedder } from "./lib/useEmbedder";
import { Controls, type StatusKind } from "./components/Controls";
import { DatasetPanel } from "./components/DatasetPanel";
import { NewPointPanel } from "./components/NewPointPanel";
import { Scatter2D } from "./components/Scatter2D";
import { Scatter3D } from "./components/Scatter3D";

const PREEMBED_DELAY_MS = 350;

type Status = { message: string; kind: StatusKind };

export default function VectorSpaceApp() {
  // --- UI state ---
  const [modelId, setModelId] = useState("Xenova/all-MiniLM-L6-v2");
  const [maxLength, setMaxLength] = useState<number | null>(4);
  const [dim, setDim] = useState<Dim>(2);
  const [status, setStatus] = useState<Status>({ message: "Ready. Add some data to begin — or click \"Try example\".", kind: "" });

  // --- Data state ---
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [newPoints, setNewPoints] = useState<NewPoint[]>([]);
  const [vectors, setVectors] = useState<EmbeddedMatrix | null>(null);
  const [pca, setPca] = useState<PcaResult | null>(null);
  const [pendingNewText, setPendingNewText] = useState("");
  const preEmbedRef = useRef({ token: 0, result: null as { text: string; vectors: EmbeddedMatrix } | null });

  const { status: modelStatus, error: modelError, embedText, activeModel } = useEmbedder(modelId, maxLength);

  // --- Model status -> status line ---
  useEffect(() => {
    if (modelStatus === "loading") {
      setStatus({ message: `Loading model ${modelId}…`, kind: "busy" });
    } else if (modelStatus === "error") {
      setStatus({ message: `Failed to load model: ${modelError ?? "unknown error"}`, kind: "error" });
    } else if (modelStatus === "ready" && activeModel) {
      setStatus({ message: `Model ${activeModel} loaded.`, kind: "ok" });
    }
  }, [modelStatus, modelError, activeModel, modelId]);

  // --- Refresh dataset embeddings + PCA whenever inputs change ---
  const recompute = useCallback(async () => {
    if (!items.length) {
      setVectors(null);
      setPca(null);
      return;
    }
    if (modelStatus !== "ready") return;
    setStatus({ message: `Embedding ${items.length} items…`, kind: "busy" });
    const labels = items.map((i) => i.label);
    let v: EmbeddedMatrix | null = null;
    try {
      v = await embedText(labels);
    } catch (e) {
      setStatus({ message: `Embed failed: ${e instanceof Error ? e.message : String(e)}`, kind: "error" });
      return;
    }
    if (!v) return;
    setVectors(v);
    setStatus({ message: `Running ${dim}D PCA on ${items.length} × ${v._d}…`, kind: "busy" });
    const p = pcaReduce(v, dim) as PcaResult;
    setPca(p);
    setStatus({
      message: `Done. PC1 ${(p.varianceExplained[0] * 100).toFixed(1)}%, PC2 ${(p.varianceExplained[1] * 100).toFixed(1)}% (total ${((p.varianceExplained[0] + p.varianceExplained[1]) * 100).toFixed(1)}%).`,
      kind: "ok",
    });
  }, [items, modelStatus, dim, embedText]);

  useEffect(() => {
    void recompute();
  }, [recompute]);

  // Re-project new points when PCA changes.
  useEffect(() => {
    if (!pca || newPoints.length === 0) return;
    if (modelStatus !== "ready") return;
    let cancelled = false;
    (async () => {
      const labels = newPoints.map((p) => p.label);
      const v = await embedText(labels);
      if (cancelled || !v) return;
      const proj = pcaProject(v, pca) as Float32Array;
      setNewPoints((prev) => prev.map((p, i) => ({
        ...p,
        x: proj[i * pca.rank],
        y: proj[i * pca.rank + 1],
        z: pca.rank === 3 ? proj[i * pca.rank + 2] : 0,
      })));
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pca, modelStatus]);

  // --- Mutations ---
  const onAdd = useCallback((lines: string[]) => {
    const newItems: DatasetItem[] = lines.map((label) => ({ id: crypto.randomUUID().slice(0, 8), label }));
    setItems((prev) => [...prev, ...newItems]);
  }, []);

  const onClear = useCallback(() => {
    setItems([]);
    setNewPoints([]);
    setVectors(null);
    setPca(null);
    preEmbedRef.current = { token: 0, result: null };
    setStatus({ message: "Dataset cleared.", kind: "" });
  }, []);

  const onLoadExample = useCallback(() => {
    const newItems: DatasetItem[] = EXAMPLE_DATASET.map((label) => ({ id: crypto.randomUUID().slice(0, 8), label }));
    setItems(newItems);
  }, []);

  const onFileText = useCallback((lines: string[]) => {
    const newItems: DatasetItem[] = lines.map((label) => ({ id: crypto.randomUUID().slice(0, 8), label }));
    setItems((prev) => [...prev, ...newItems]);
  }, []);

  const onAddNew = useCallback(async (text: string) => {
    if (!pca) {
      setStatus({ message: "Add a dataset first to compute PCA.", kind: "error" });
      return;
    }
    if (modelStatus !== "ready") return;
    let v: EmbeddedMatrix;
    const pre = preEmbedRef.current.result;
    if (pre && pre.text === text) {
      v = pre.vectors;
      preEmbedRef.current.result = null;
    } else {
      setStatus({ message: "Embedding new point…", kind: "busy" });
      v = await embedText([text]);
    }
    if (!v) return;
    const proj = pcaProject(v, pca) as Float32Array;
    const point: NewPoint = {
      id: crypto.randomUUID().slice(0, 8),
      label: text,
      x: proj[0],
      y: proj[1],
      z: pca.rank === 3 ? proj[2] : 0,
    };
    setNewPoints((prev) => [...prev, point]);
    setStatus({
      message: `New point placed at (${point.x.toFixed(2)}, ${point.y.toFixed(2)}${pca.rank === 3 ? ", " + point.z.toFixed(2) : ""}).`,
      kind: "ok",
    });
  }, [pca, modelStatus, embedText]);

  const onRemoveNew = useCallback((id: string) => {
    setNewPoints((prev) => prev.filter((p) => p.id !== id));
  }, []);

  // --- Debounced pre-embed as user types ---
  useEffect(() => {
    if (!pendingNewText || modelStatus !== "ready" || !pca) {
      preEmbedRef.current = { token: 0, result: null };
      return;
    }
    const myToken = ++preEmbedRef.current.token;
    const handle = setTimeout(async () => {
      if (preEmbedRef.current.token !== myToken) return;
      const v = await embedText([pendingNewText]);
      if (preEmbedRef.current.token !== myToken) return;
      preEmbedRef.current = { token: myToken, result: { text: pendingNewText, vectors: v } };
    }, PREEMBED_DELAY_MS);
    return () => clearTimeout(handle);
  }, [pendingNewText, modelStatus, pca, embedText]);

  // --- Variance text for the footer of the plot ---
  const projectedItems = useMemo<ProjectedItem[]>(() => {
    if (!pca) return [];
    const k = pca.rank;
    return items.map((it, i) => ({
      id: it.id,
      label: it.label,
      x: pca.points[i * k],
      y: pca.points[i * k + 1],
      z: k === 3 ? pca.points[i * k + 2] : 0,
    }));
  }, [items, pca]);

  const varianceText = useMemo(() => {
    if (!pca) return "";
    const v = pca.varianceExplained;
    const total = ((v[0] ?? 0) + (v[1] ?? 0)) * 100;
    const parts = [
      `PC1 ${(v[0] * 100).toFixed(2)}%`,
      `PC2 ${(v[1] * 100).toFixed(2)}%`,
    ];
    if (v[2] !== undefined) parts.push(`PC3 ${(v[2] * 100).toFixed(2)}%`);
    parts.push(`total ${total.toFixed(2)}%`);
    return parts.join(" · ");
  }, [pca]);

  return (
    <>
      <header>
        <h1>Vector Space Explorer</h1>
        <p className="tagline">
          Embed your data, project to 2D or 3D with PCA, and drop in new
          points to see where they land.
        </p>
      </header>

      <Controls
        modelId={modelId}
        onModelChange={(id) => {
          setVectors(null);
          setPca(null);
          preEmbedRef.current = { token: 0, result: null };
          setModelId(id);
        }}
        dim={dim}
        onDimChange={setDim}
        maxLength={maxLength}
        onMaxLengthChange={setMaxLength}
        status={status.message}
        statusKind={status.kind}
      />

      <section className="io">
        <DatasetPanel
          count={items.length}
          onAdd={onAdd}
          onClear={onClear}
          onLoadExample={onLoadExample}
          onFileText={onFileText}
        />
        <NewPointPanel
          newPoints={newPoints}
          onAdd={onAddNew}
          onRemove={onRemoveNew}
          onChange={setPendingNewText}
        />
      </section>

      <section className="viz">
        <div className="viz-header">
          <h2>Vector space</h2>
          <div className="legend">
            <span>
              <span className="swatch" style={{ background: COLORS.dataset }}></span>
              dataset ({items.length})
            </span>
            {newPoints.length > 0 && (
              <span>
                <span className="swatch" style={{ background: COLORS.newPoint }}></span>
                new ({newPoints.length})
              </span>
            )}
          </div>
        </div>
        <div className="chart-wrap" style={{ position: "relative" }}>
          {dim === 2 ? (
            <Scatter2D items={projectedItems} newPoints={newPoints} />
          ) : (
            <Scatter3D items={projectedItems} newPoints={newPoints} />
          )}
        </div>
        <div id="variance" className="muted">{varianceText}</div>
      </section>

      <footer>
        <p>Runs entirely in your browser. Embedding model fetched once on first use, cached after.</p>
      </footer>
    </>
  );
}