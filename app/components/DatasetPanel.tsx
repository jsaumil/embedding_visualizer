"use client";
// Left panel: paste/type items, load files, add/clear/example.
import { useRef, useState } from "react";
import type { DatasetItem } from "../lib/state";

export function DatasetPanel({
  count,
  onAdd,
  onClear,
  onLoadExample,
  onFileText,
}: {
  count: number;
  onAdd: (lines: string[]) => void;
  onClear: () => void;
  onLoadExample: () => void;
  onFileText: (lines: string[]) => void;
}) {
  const [text, setText] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  const submit = () => {
    const lines = text
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (!lines.length) return;
    onAdd(lines);
    setText("");
  };

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    const all: string[] = [];
    for (const f of files) {
      const text = await f.text();
      for (const line of text.split(/\r?\n/)) {
        const t = line.trim();
        if (t) all.push(t);
      }
    }
    if (all.length) onFileText(all);
    e.target.value = "";
  };

  return (
    <div className="panel">
      <h2>1. Your dataset</h2>
      <p className="hint">
        Drop text files, paste text, or type a short label per item. Each line
        becomes one point.
      </p>
      <div className="input-row">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={
            "One item per line. E.g.\nA photo of a red rose\nA bowl of fresh pasta\nA sound of thunder"
          }
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className="actions">
          <button className="primary" onClick={submit}>
            Add to dataset
          </button>
          <button onClick={onClear}>Clear dataset</button>
          <button className="ghost" onClick={onLoadExample} title="Load a built-in sample dataset">
            Try example
          </button>
        </div>
      </div>
      <div className="file-row">
        <label className="file-btn">
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".txt,.csv,text/*"
            onChange={onFile}
          />
          <span>Load text file(s)</span>
        </label>
        <span className="muted">{count} items</span>
      </div>
    </div>
  );
}