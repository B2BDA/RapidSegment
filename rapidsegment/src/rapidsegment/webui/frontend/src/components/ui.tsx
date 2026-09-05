import React, { lazy, Suspense, useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

// ── Plotly chart (code-split; plotly.js is large) ─────────────────────────────
const Plot = lazy(() => import('./Plot'));

export function PlotChart({ figure, height = 420 }: { figure?: any; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  if (!figure || !figure.layout || !figure.data) return null;
  const layout = { ...figure.layout, height, autosize: true, margin: { ...(figure.layout.margin || {}), t: 50, l: 60, r: 20, b: 60 } };
  return (
    <div ref={ref} style={{ width: '100%', height }}>
      <Suspense fallback={<div className="caption" style={{ padding: 24 }}>Loading chart…</div>}>
        <Plot
          data={figure.data}
          layout={layout}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
          config={{ responsive: true, displaylogo: false }}
        />
      </Suspense>
    </div>
  );
}

// ── Layout primitives ─────────────────────────────────────────────────────────
export function Card({ title, children, style }: { title?: React.ReactNode; children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div className="card" style={style}>
      {title && <h3 style={{ marginTop: 0 }}>{title}</h3>}
      {children}
    </div>
  );
}

export function Caption({ children }: { children: React.ReactNode }) {
  return <div className="caption" style={{ margin: '4px 0' }}>{children}</div>;
}

export function Metric({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="metric">
      <div className="m-label">{label}</div>
      <div className="m-value">{value ?? '—'}</div>
      {sub && <div className="m-sub">{sub}</div>}
    </div>
  );
}

export function Alert({ kind, children }: { kind: 'success' | 'info' | 'warning' | 'error'; children: React.ReactNode }) {
  return <div className={`alert alert-${kind}`}>{children}</div>;
}

export function Pill({ state, icon, step, name }: { state: string; icon?: string; step?: number; name: string }) {
  const icons: Record<string, string> = { done: '✅', active: '⏳', pending: '⏳', error: '❌' };
  return (
    <div className={`pill ${state}`}>
      <div className="p-icon">{icon || icons[state] || ''}</div>
      {step && <div className="p-step">Step {step}</div>}
      <div className="p-name">{name}</div>
    </div>
  );
}

export function Tabs({ tabs, active, onChange }: { tabs: string[]; active: string; onChange: (t: string) => void }) {
  return (
    <div className="tabs">
      {tabs.map((t) => (
        <div key={t} className={`tab ${t === active ? 'active' : ''}`} onClick={() => onChange(t)}>
          {t}
        </div>
      ))}
    </div>
  );
}

// ── Generic data table ─────────────────────────────────────────────────────────
export function DataTable({ rows, columns, maxHeight = 420, idKey }: { rows: Record<string, any>[]; columns?: string[]; maxHeight?: number; idKey?: string }) {
  const cols = columns || (rows.length ? Object.keys(rows[0]) : []);
  if (!rows.length) return <Caption>No rows.</Caption>;
  return (
    <div className="table-scroll" style={{ maxHeight }}>
      <table className="data">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={idKey ? String(r[idKey]) : i}>
              {cols.map((c) => {
                const v = r[c];
                let cell: React.ReactNode = v;
                if (v === null || v === undefined) cell = '—';
                else if (typeof v === 'number') cell = v.toLocaleString();
                else if (typeof v === 'object') cell = JSON.stringify(v);
                else cell = String(v);
                return <td key={c}>{cell}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Download({ url, label, filename }: { url: string; label: string; filename: string }) {
  return <a className="btn small" href={url} download={filename}>{label}</a>;
}

export function copyToClipboard(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {});
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="small"
      onClick={() => { copyToClipboard(text); setDone(true); setTimeout(() => setDone(false), 1200); }}
    >
      {done ? '✓ Copied' : label}
    </button>
  );
}

export function Terminal({ lines, height = 380 }: { lines: { ts: string; level: string; msg: string }[]; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines.length]);
  return (
    <div className="terminal" style={{ height }} ref={ref}>
      {lines.length === 0 && <span className="muted">— no log output yet —</span>}
      {lines.map((l, i) => (
        <div key={i} className="log-line">
          <span className="log-time">{l.ts}</span>{' '}
          <span className={`log-level lvl-${l.level}`}>[{l.level}]</span>{' '}
          <span>{l.msg}</span>
        </div>
      ))}
    </div>
  );
}

// ── GoToWorkbench helper (Streamlit page_link equivalent) ─────────────────────
export function useGo() {
  const nav = useNavigate();
  return { goTo: (path: string) => nav(path) };
}
