import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, post } from '../api';
import type { ExperimentFull, LogRecord, RunSnapshot } from '../types';
import { Alert, Caption, Card, Metric, Pill, PlotChart } from '../components/ui';

const REFRESH_MS = 2000;
const LEVEL_RANK: Record<string, number> = { INFO: 10, WARNING: 30, ERROR: 40, DEBUG: 5 };

function fmtDuration(secs: number): string {
  secs = Math.max(0, Math.round(secs));
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${(secs % 60).toString().padStart(2, '0')}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor(secs % 3600 / 60).toString().padStart(2, '0')}m`;
}

function filterLogs(records: LogRecord[], lvl: string): LogRecord[] {
  if (lvl === 'All') return records;
  const minRank = LEVEL_RANK[lvl] ?? 10;
  return records.filter((r) => (LEVEL_RANK[r.level] ?? 10) >= minRank);
}

type Mode =
  | { kind: 'running'; expId: string }
  | { kind: 'final'; exp: ExperimentFull }
  | { kind: 'idle' };

export default function M3Execution() {
  const nav = useNavigate();
  const [mode, setMode] = useState<Mode>({ kind: 'idle' });
  const [snap, setSnap] = useState<RunSnapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const lastCount = useRef(0);
  const [logFilter, setLogFilter] = useState('All');
  const [notice, setNotice] = useState<{ kind: 'success' | 'warning' | 'error'; text: string } | null>(null);

  const startPending = useCallback(async () => {
    const pending = sessionStorage.getItem('wb_pending_run');
    if (!pending) return false;
    sessionStorage.removeItem('wb_pending_run');
    try {
      const { exp_id } = await post<{ exp_id: string }>('/api/m3/start', { cfg: JSON.parse(pending) });
      setMode({ kind: 'running', expId: exp_id });
      return true;
    } catch (e) {
      setErr((e as Error).message);
      return false;
    }
  }, []);

  // Start a pending run handed off from Module 2.
  useEffect(() => {
    (async () => {
      const started = await startPending();
      if (started) return;
      // Otherwise show the most recent experiment if any.
      try {
        const r = await api<{ experiment: ExperimentFull | null }>('/api/m3/latest');
        if (r.experiment) setMode({ kind: 'final', exp: r.experiment });
      } catch (e) {
        setErr((e as Error).message);
      }
    })();
  }, [startPending]);

  // Poll live runs.
  useEffect(() => {
    if (mode.kind !== 'running') return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const s = await api<RunSnapshot>(`/api/m3/status/${mode.expId}?after=${lastCount.current}`);
        if (stopped) return;
        lastCount.current = s.log_count;
        setSnap(s);
        if (s.finalized) {
          try {
            const exp = await api<ExperimentFull>(`/api/m3/experiment/${mode.expId}`);
            setMode({ kind: 'final', exp });
          } catch {
            setMode({ kind: 'final', exp: { exp_id: mode.expId, name: s.experiment_name, created_at: '', status: s.status, execution_time_sec: s.elapsed } as ExperimentFull });
          }
          return;
        }
        timer = setTimeout(tick, REFRESH_MS);
      } catch (e) {
        setErr((e as Error).message);
      }
    };
    timer = setTimeout(tick, 300);
    return () => { stopped = true; clearTimeout(timer); };
  }, [mode]);

  const cancelRun = async () => {
    if (mode.kind !== 'running') return;
    try {
      await post(`/api/m3/cancel/${mode.expId}`, {});
      setNotice({ kind: 'warning', text: 'Cancel requested — partial results will be saved.' });
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const download = (url: string, filename: string) => {
    fetch(url).then((r) => r.blob()).then((b) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = filename;
      a.click();
    }).catch((e) => setErr(e.message));
  };

  return (
    <div>
      <h1>Module 3: Execution & Artifact Console</h1>
      <Caption>Real-time extraction monitoring, log / SQL console, cancel & export hub.</Caption>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}
      {err && <Alert kind="error">{err}</Alert>}

      {mode.kind === 'running' && snap && (
        <LiveView snap={snap} logFilter={logFilter} setLogFilter={setLogFilter} onCancel={cancelRun} />
      )}

      {mode.kind === 'final' && (
        <FinalView exp={mode.exp} download={download} goWorkbench={() => nav('/m2')} />
      )}

      {mode.kind === 'idle' && !snap && !err && (
        <Alert kind="info">
          No experiment pending. Configure one in the Workbench (Module 2) and press Run Experiment.
          <div style={{ marginTop: 8 }}><button onClick={() => nav('/m2')}>Go to Workbench</button></div>
        </Alert>
      )}
    </div>
  );
}

// ── Live running console ──────────────────────────────────────────────────────
function LiveView({ snap, logFilter, setLogFilter, onCancel }: {
  snap: RunSnapshot;
  logFilter: string;
  setLogFilter: (l: string) => void;
  onCancel: () => void;
}) {
  const status = snap.finalized ? snap.status : undefined;
  const states = stepStates(snap.step, status);
  const segs = snap.segments_preview || [];
  const lifts = segs.map((s) => Number(s.lift) || 0).filter(Boolean);

  return (
    <div>
      <div className="spread">
        <div>
          <h2 style={{ marginBottom: 0 }}>{snap.experiment_name}</h2>
          <Caption>{snap.exp_id} · target=`{snap.target_col}` · {snap.n_rows.toLocaleString()} rows × {snap.n_cols} cols</Caption>
        </div>
        <button className="primary danger" onClick={onCancel} disabled={snap.cancel_requested}>
          {snap.cancel_requested ? 'Cancel requested…' : '⛔ Cancel Extraction'}
        </button>
      </div>

      <Card>
        <h3>Status Timeline</h3>
        <div className="steps">
          {snap.step_names.map((n, i) => (
            <Pill key={n} step={i + 1} name={n} state={states[i]} />
          ))}
        </div>
        <Caption>
          ⏱ Elapsed: {fmtDuration(snap.elapsed)} · Found <b>{snap.segments_found}</b> segment(s) so far… · Current feature: <b>{snap.current_feature || '—'}</b>
        </Caption>
      </Card>

      <div className="metrics">
        <Metric label="Segments found" value={snap.segments_found} />
        <Metric label="Total coverage %" value={snap.coverage_pct != null ? `${snap.coverage_pct.toFixed(1)}%` : '—'} />
        <Metric label="Average lift" value={snap.avg_lift != null ? `${snap.avg_lift.toFixed(2)}×` : '—'} />
        <Metric label="Best segment (lift)" value={snap.best_lift != null ? `${snap.best_lift.toFixed(2)}×` : '—'} sub={snap.best_rule || undefined} />
      </div>
      {!snap.finalized && (
        <Caption>Coverage % is a live estimate (residual counts ÷ total rows) until the final coverage pass completes.</Caption>
      )}

      <h3>Top candidates</h3>
      <Card>
        {snap.top_candidates.length
          ? snap.top_candidates.map((s, i) => (
              <Caption key={i}>· `{s.rule_string}` — {s.count.toLocaleString()} rows · lift {s.lift.toFixed(2)}×</Caption>
            ))
          : <Caption>No segments yet — extraction in progress…</Caption>}
      </Card>

      <div className="grid2" style={{ gridTemplateColumns: '1fr 1fr', alignItems: 'start' }}>
        <LogPane logs={filterLogs(snap.logs, logFilter)} setLogFilter={setLogFilter} logFilter={logFilter} expId={snap.exp_id} />
        <SqlPane segments={segs} expId={snap.exp_id} />
      </div>
    </div>
  );
}

function LogPane({ logs, logFilter, setLogFilter, expId }: { logs: LogRecord[]; logFilter: string; setLogFilter: (l: string) => void; expId: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs.length]);
  return (
    <Card>
      <div className="spread">
        <h3 style={{ margin: 0 }}>Log Terminal</h3>
        <div className="row">
          <select value={logFilter} onChange={(e) => setLogFilter(e.target.value)} style={{ width: 110 }}>
            {['All', 'Info', 'Warning', 'Error'].map((l) => <option key={l}>{l}</option>)}
          </select>
          <DownloadBtn url={`/api/m3/export/${expId}/logs.txt`} filename={`logs_${expId}.txt`} label="Copy Logs (.txt)" />
        </div>
      </div>
      <div ref={ref} className="terminal" style={{ height: 320 }}>
        {logs.length === 0 && <span className="muted">— no log output yet —</span>}
        {logs.map((l, i) => (
          <div key={i} className="log-line">
            <span className="log-time">{l.ts}</span>{' '}
            <span className={`log-level lvl-${l.level}`}>[{l.level}]</span>{' '}
            <span>{l.msg}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

function SqlPane({ segments, expId }: { segments: { segment_id?: number; rule_string?: string; sql_filter?: string }[]; expId: string }) {
  return (
    <Card title="SQL Inspector">
      {segments.length ? segments.map((s, i) => (
        <div key={i} style={{ marginBottom: 10 }}>
          <div className="spread">
            <Caption>Segment {s.segment_id} · `{s.rule_string}`</Caption>
            <CopySql text={s.sql_filter || ''} expId={expId} segmentId={String(s.segment_id ?? i + 1)} />
          </div>
          <pre className="code" style={{ margin: 0 }}>{s.sql_filter || '—'}</pre>
        </div>
      )) : <Caption>Waiting for segments…</Caption>}
    </Card>
  );
}

function DownloadBtn({ url, filename, label }: { url: string; filename: string; label: string }) {
  const go = () => {
    fetch(url).then((r) => r.blob()).then((b) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = filename;
      a.click();
    }).catch(() => {});
  };
  return <button className="small" onClick={go}>{label}</button>;
}

function CopySql({ text, expId, segmentId }: { text: string; expId: string; segmentId: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="small"
      onClick={() => { navigator.clipboard?.writeText(text).catch(() => {}); setDone(true); setTimeout(() => setDone(false), 1000); }}
    >
      {done ? '✓ Copied' : 'Copy SQL'}
    </button>
  );
}

function stepStates(step: number, status?: string): string[] {
  if (status === 'completed') return ['done', 'done', 'done', 'done', 'done', 'done'];
  if (status === 'cancelled' || status === 'failed') {
    return Array.from({ length: 6 }, (_, i) => (i < step ? 'done' : (i === step ? 'error' : 'pending')));
  }
  return Array.from({ length: 6 }, (_, i) => (i < step ? 'done' : (i === step ? 'active' : 'pending')));
}

// ── Final / view mode ─────────────────────────────────────────────────────────
function FinalView({ exp, download, goWorkbench }: {
  exp: ExperimentFull;
  download: (u: string, f: string) => void;
  goWorkbench: () => void;
}) {
  const [tab, setTab] = useState<'console' | 'results'>('results');
  const res = (exp.result || {}) as any;
  const segments = res.segments || [];
  const coverage = res.coverage || [];
  const status = exp.status;

  return (
    <div>
      <div className="spread">
        <div>
          <h2 style={{ marginBottom: 0 }}>{exp.name}</h2>
          <Caption>{exp.exp_id} · {exp.created_at || ''} · status=`{status}` · target=`{exp.target_col || ''}` · {(exp.data_rows ?? '?').toLocaleString?.() ?? exp.data_rows} rows</Caption>
        </div>
      </div>

      {status === 'cancelled' && <Alert kind="warning">This experiment was cancelled — partial results shown below.</Alert>}
      {status === 'failed' && <Alert kind="error">This experiment failed: {res.error_msg || 'unknown error'}</Alert>}

      <div className="tabs">
        {['Results Summary', 'Console / Segments'].map((t) => (
          <div key={t} className={`tab${(tab === 'console' ? 'Console / Segments' : 'Results Summary') === t ? ' active' : ''}`} onClick={() => setTab(t === 'Results Summary' ? 'results' : 'console')}>{t}</div>
        ))}
      </div>

      {tab === 'results' ? (
        <ResultsSummary exp={exp} segments={segments} coverage={coverage} />
      ) : (
        <ViewConsole exp={exp} />
      )}

      <ExportHub expId={exp.exp_id} download={download} />

      <div className="divider" />
      <div className="grid2">
        <div>
          <button className="primary" onClick={goWorkbench}>⚙️ Configure new experiment in Workbench (Module 2)</button>
        </div>
        <div className="row">
          <button onClick={() => { sessionStorage.setItem('wb_pending_run', JSON.stringify(exp.config || {})); window.location.reload(); }}>♻️ Re-run last config</button>
        </div>
      </div>
    </div>
  );
}

function ViewConsole({ exp }: { exp: ExperimentFull }) {
  const res = (exp.result || {}) as any;
  const segments = res.segments || [];
  const runSnap: RunSnapshot = {
    exp_id: exp.exp_id,
    status: exp.status || 'completed',
    finalized: true,
    step: res.step_reached || 6,
    step_names: ['Configure & load data', 'Feature ranking (IV / response rate)', 'Candidate rule generation', 'Binning & rule complexity', 'Residual extraction (per segment)', 'Final coverage'],
    elapsed: exp.execution_time_sec || 0,
    n_rows: exp.data_rows || 0,
    n_cols: exp.data_cols || 0,
    target_col: exp.target_col || '',
    experiment_name: exp.name,
    segments_found: segments.length,
    coverage_pct: res.coverage_pct ?? null,
    avg_lift: res.avg_lift ?? null,
    best_lift: res.max_lift ?? null,
    best_rule: null,
    current_feature: null,
    top_candidates: segments.slice(0, 3).map((s: any) => ({ rule_string: s.rule_string, count: s.count || 0, lift: Number(s.lift) || 0 })),
    segments_preview: (segments || []).slice(0, 20).map((s: any) => ({ segment_id: s.segment_id, rule_string: s.rule_string, sql_filter: s.sql_filter, count: s.count, rate: s.rate, lift: s.lift })),
    logs: exp.logs || [],
    log_count: (exp.logs || []).length,
  };
  const [logFilter, setLogFilter] = useState('All');
  return (
    <div>
      <div className="steps" style={{ marginBottom: 8 }}>
        {runSnap.step_names.map((n, i) => (
          <Pill key={n} step={i + 1} name={n} state={stepStates(runSnap.step, exp.status || 'completed')[i]} />
        ))}
      </div>
      <Caption>⏱ Elapsed: {fmtDuration(exp.execution_time_sec || 0)} · Segments found: {segments.length} · stop_reason: {res.stop_reason || '—'}</Caption>
      <div className="grid2" style={{ alignItems: 'start' }}>
        <LogPane logs={filterLogs(exp.logs || [], logFilter)} logFilter={logFilter} setLogFilter={setLogFilter} expId={exp.exp_id} />
        <SqlPane segments={segments} expId={exp.exp_id} />
      </div>
    </div>
  );
}

function ResultsSummary({ exp, segments, coverage }: { exp: ExperimentFull; segments: any[]; coverage: any[] }) {
  const res = (exp.result || {}) as any;
  const segCols = ['segment_id', 'rule_string', 'sql_filter', 'count', 'rate', 'lift', 'meta_applied_sample_size', 'meta_applied_min_lift'];
  const metrics = [
    <Metric key="0" label="Segments" value={res.segments_count ?? segments.length} />,
    <Metric key="1" label="Coverage %" value={Number(res.coverage_pct ?? 0).toFixed(2)} />,
    <Metric key="2" label="Avg lift" value={`${Number(res.avg_lift ?? 0).toFixed(2)}×`} />,
    <Metric key="3" label="Max lift" value={`${Number(res.max_lift ?? 0).toFixed(2)}×`} />,
    <Metric key="4" label="Baseline rate" value={`${Number(res.baseline_rate_pct ?? 0).toFixed(2)}%`} />,
    <Metric key="5" label="Elapsed" value={fmtDuration(exp.execution_time_sec || 0)} />,
  ];
  const rows = segments.map((s) => {
    const out: Record<string, any> = {};
    for (const c of segCols) if (c in s) out[c] = s[c];
    return out;
  });
  return (
    <div>
      <h3>Results Summary</h3>
      <div className="metrics" style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)' }}>{metrics}</div>
      {segments.length ? (
        <>
          <h3>Segments</h3>
          <div className="table-scroll"><table className="data"><tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {Object.keys(r).map((k) => <td key={k}><b>{k}</b>: {typeof r[k] === 'number' ? (typeof r[k] === 'number' && Number.isFinite(r[k]) ? String(Math.round(r[k] * 100) / 100) : r[k]) : String(r[k])}</td>)}
              </tr>
            ))}
          </tbody></table></div>
        </>
      ) : <Caption>No segments — experiment produced none.</Caption>}

      {coverage.length ? (
        <>
          <h3>Final coverage (events vs. non-events)</h3>
          <div className="table-scroll"><table className="data"><tbody>
            {coverage.map((r, i) => (
              <tr key={i}>{Object.entries(r).map(([k, v]) => <td key={k}>{k}={typeof v === 'number' ? (Number.isFinite(v) ? Math.round(v * 100) / 100 : String(v)) : String(v)}</td>)}</tr>
            ))}
          </tbody></table></div>
        </>
      ) : <Caption>No coverage rows — experiment produced no segments.</Caption>}
    </div>
  );
}

function ExportHub({ expId, download }: { expId: string; download: (u: string, f: string) => void }) {
  return (
    <div>
      <div className="divider" />
      <h3>Export Hub</h3>
      <div className="row">
        <button className="small" onClick={() => download(`/api/m3/export/${expId}/logs.txt`, `logs_${expId}.txt`)}>⬇️ Logs.txt</button>
        <button className="small" onClick={() => download(`/api/m3/export/${expId}/sql.sql`, `segments_${expId}.sql`)}>⬇️ SQL.sql</button>
        <button className="small" onClick={() => download(`/api/m3/export/${expId}/config.json`, `config_${expId}.json`)}>⬇️ Config.json</button>
      </div>
    </div>
  );
}

export function StepPillPreview() {
  return null;
}
