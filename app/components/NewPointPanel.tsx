"use client";
// Right panel: type a new point, add it. Shows the chips for already-added
// new points. Pre-embed is debounced in the parent; this component just
// reports input.
import { useState } from "react";
import type { NewPoint } from "../lib/state";

export function NewPointPanel({
  newPoints,
  onAdd,
  onRemove,
  onChange,
}: {
  newPoints: NewPoint[];
  onAdd: (text: string) => void;
  onRemove: (id: string) => void;
  onChange: (text: string) => void;
}) {
  const [text, setText] = useState("");

  const setAndReport = (v: string) => {
    setText(v);
    onChange(v);
  };

  const submit = () => {
    const t = text.trim();
    if (!t) return;
    onAdd(t);
    setText("");
  };

  const truncate = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

  return (
    <div className="panel">
      <h2>2. New point</h2>
      <p className="hint">
        Anything goes. This will be embedded by the same model and projected with
        the dataset's PCA, then drawn in a contrasting color so it stands out
        from the dataset.
      </p>
      <div className="input-row">
        <textarea
          value={text}
          onChange={(e) => setAndReport(e.target.value)}
          placeholder="A short description, sentence, or question."
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className="actions">
          <button className="primary" onClick={submit}>
            Add new point
          </button>
        </div>
      </div>
      <div className="chips">
        {newPoints.map((p) => (
          <span className="chip new" key={p.id}>
            {truncate(p.label, 28)}
            <span
              className="x"
              title="Remove"
              onClick={() => onRemove(p.id)}
            >
              ×
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}