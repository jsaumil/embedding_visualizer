"use client";
// Top control bar: model selector, 2D/3D toggle, context-length cap, status line.
import type { Dim } from "../lib/state";

export type StatusKind = "" | "busy" | "ok" | "error";

export function Controls({
  modelId,
  onModelChange,
  dim,
  onDimChange,
  maxLength,
  onMaxLengthChange,
  status,
  statusKind,
}: {
  modelId: string;
  onModelChange: (id: string) => void;
  dim: Dim;
  onDimChange: (d: Dim) => void;
  maxLength: number | null;
  onMaxLengthChange: (ml: number | null) => void;
  status: string;
  statusKind: StatusKind;
}) {
  return (
    <section className="controls">
      <div className="row">
        <label htmlFor="model">Embedding model</label>
        <select id="model" value={modelId} onChange={(e) => onModelChange(e.target.value)}>
          <option value="Xenova/all-MiniLM-L6-v2">MiniLM-L6 (text, 384d, fast)</option>
          <option value="Xenova/all-mpnet-base-v2">MPNet base (text, 768d, accurate)</option>
          <option value="Xenova/bge-small-en-v1.5">BGE small (text, 384d)</option>
        </select>
      </div>

      <div className="row">
        <label htmlFor="dim">Dimensions</label>
        <div className="seg" role="tablist">
          <button
            type="button"
            className={"seg-btn" + (dim === 2 ? " active" : "")}
            onClick={() => onDimChange(2)}
          >
            2D
          </button>
          <button
            type="button"
            className={"seg-btn" + (dim === 3 ? " active" : "")}
            onClick={() => onDimChange(3)}
          >
            3D
          </button>
        </div>
      </div>

      <div className="row">
        <label htmlFor="maxlen">Context</label>
        <select
          id="maxlen"
          value={maxLength === null ? "auto" : String(maxLength)}
          onChange={(e) => {
            const v = e.target.value;
            onMaxLengthChange(v === "auto" ? null : parseInt(v, 10));
          }}
          title="Maximum tokens per item. Shorter contexts are faster but truncate long texts."
        >
          <option value="auto">Auto (model default)</option>
          <option value="3">Shortest (≤ 3 tokens)</option>
          <option value="4">Short (≤ 4 tokens)</option>
          <option value="8">Medium (≤ 8 tokens)</option>
          <option value="16">Long (≤ 16 tokens)</option>
          <option value="32">X-Long (≤ 32 tokens)</option>
          <option value="128">Full (≤ 128 tokens)</option>
        </select>
      </div>

      <div id="status" className={"status " + statusKind}>
        {status}
      </div>
    </section>
  );
}