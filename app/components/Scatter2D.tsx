"use client";
// 2D scatter plot rendered to a canvas. Same drawing logic as the legacy
// app.js, lifted into a React component. Tooltip is plain DOM (no canvas
// hit-test pixel read).
import { useEffect, useRef } from "react";
import { COLORS, type ProjectedItem } from "../lib/state";

export function Scatter2D({
  items,
  newPoints,
}: {
  items: ProjectedItem[];
  newPoints: ProjectedItem[];
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const positions = useRef<{ x: number; y: number; label: string; kind: "dataset" | "new" }[]>([]);
  const tooltip = useRef<HTMLDivElement>(null);

  // Resize the canvas backing buffer to match the displayed size × DPR.
  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    draw();
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, newPoints]);

  const draw = () => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight || Math.round(w * 0.55);
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.height = h + "px";
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pts: { x: number; y: number; label: string; kind: "dataset" | "new" }[] = [];
    for (const it of items) pts.push({ x: it.x, y: it.y, label: it.label, kind: "dataset" });
    for (const p of newPoints) pts.push({ x: p.x, y: p.y, label: p.label, kind: "new" });

    if (!pts.length) {
      positions.current = [];
      return;
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const padX = (maxX - minX) * 0.08 || 1;
    const padY = (maxY - minY) * 0.08 || 1;
    minX -= padX; maxX += padX; minY -= padY; maxY += padY;

    const sx = (v: number) => ((v - minX) / (maxX - minX)) * w;
    const sy = (v: number) => h - ((v - minY) / (maxY - minY)) * h;

    // Grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * w;
      const y = (i / 10) * h;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    // Axes
    ctx.strokeStyle = COLORS.axis;
    if (minX < 0 && maxX > 0) {
      const x0 = sx(0);
      ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, h); ctx.stroke();
    }
    if (minY < 0 && maxY > 0) {
      const y0 = sy(0);
      ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke();
    }

    // Dataset: one color
    ctx.fillStyle = COLORS.dataset;
    for (const p of pts) {
      if (p.kind !== "dataset") continue;
      const px = sx(p.x), py = sy(p.y);
      ctx.beginPath(); ctx.arc(px, py, 5.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = COLORS.datasetEdge;
    ctx.lineWidth = 1.5;
    for (const p of pts) {
      if (p.kind !== "dataset") continue;
      const px = sx(p.x), py = sy(p.y);
      ctx.beginPath(); ctx.arc(px, py, 5.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // New points: distinct color with glow
    for (const p of pts) {
      if (p.kind !== "new") continue;
      const px = sx(p.x), py = sy(p.y);
      ctx.save();
      ctx.globalAlpha = 0.25;
      ctx.fillStyle = COLORS.newPoint;
      ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      ctx.fillStyle = COLORS.newPoint;
      ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Cache for hit testing
    const pos: { x: number; y: number; label: string; kind: "dataset" | "new" }[] = [];
    for (const p of pts) pos.push({ x: sx(p.x), y: sy(p.y), label: p.label, kind: p.kind });
    positions.current = pos;
  };

  // Hit-test tooltip on mousemove.
  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = canvas.current;
    const tip = tooltip.current;
    if (!c || !tip) return;
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const ps = positions.current;
    let best: typeof ps[number] | null = null;
    let bestDist = 16;
    for (const p of ps) {
      const dx = mx - p.x, dy = my - p.y;
      const d = Math.hypot(dx, dy);
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    if (best) {
      tip.hidden = false;
      tip.textContent = (best.kind === "new" ? "★ " : "") + best.label;
      tip.style.left = mx + 12 + "px";
      tip.style.top = my + 12 + "px";
    } else {
      tip.hidden = true;
    }
  };

  return (
    <div className="chart-wrap" style={{ position: "relative" }}>
      <canvas
        ref={canvas}
        width={900}
        height={560}
        aria-label="2D scatter of embeddings"
        onMouseMove={onMove}
        onMouseLeave={() => {
          if (tooltip.current) tooltip.current.hidden = true;
        }}
      />
      <div ref={tooltip} className="tooltip" hidden />
    </div>
  );
}