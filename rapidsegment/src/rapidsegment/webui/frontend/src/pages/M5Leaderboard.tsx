import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, post } from '../api';
import type { LeaderboardRow } from '../types';
import { Alert, Caption, Card, DataTable, Metric } from '../components/ui';

const KPI_OPTS: Record<string, string> = {
  'Avg Lift': 'avg_lift',
  'Max Lift': 'max_lift',
  'Coverage %': 'coverage_pct',
  'Cumulative Event Capture %': 'cumulative_event_capture',
  'Segments': 'segments_count',
};

const num = (x: unknown, d = 0): number => {
  const n = Number(x ?? 0);
  return Number.isFinite(n) ? Number(n.toFixed(d)) : 0;
};

export default function M5Leaderboard() {
  const nav = useNavigate();
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [readErr, setReadErr] = useState('');
  const [dataset, setDataset] = useState('All datasets');
  const [kpiLabel, setKpiLabel] = useState('Avg Lift');
  const [statusSel, setStatusSel] = useState<string[]>([]);
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState<{ kind: 'success' | 'error' | 'warning'; text: string } | null>(null);
  const [confirmDel, setConfirmDel] = useState<string | null>(null);
  const [cmpA, setCmpA] = useState('');
  const [cmpB, setCmpB] = useState('');

  const load = async () => {
    try {
      const r = await api<{ rows: LeaderboardRow[] }>('/api/m5/experiments');
      setRows(r.rows);
      const statuses = Array.from(new Set(r.rows.map((x) => x.status)));
      setStatusSel((p) => (p.length === 0 ? statuses : p));
    } catch (e) {
      setReadErr((e as Error).message);
    }
  };

  useEffect(() => { void load(); }, []);

  const groups = useMemo(() => {
    const g: Record<string, LeaderboardRow[]> = {};
    for (const r of rows) (g[r.dataset_name || '(unnamed)'] ||= []).push(r);
    return g;
  }, [rows]);

  const dsList = useMemo(
    () => Object.entries(groups).sort((a, b) => (b[1].length - a[1].length) || (a[0] < b[0] ? -1 : 1)).map(([k]) => k),
    [groups],
  );

  const viewRows = dataset === 'All datasets' ? rows : (groups[dataset] || []);
  const statuses = Array.from(new Set(viewRows.map((r) => r.status)));

  useEffect(() => {
    if (statuses.length) setStatusSel((p) => p.filter((s) => statuses.includes(s)).length ? p.filter((s) => statuses.includes(s)) : statuses);
  }, [dataset]); // eslint-disable-line react-hooks/exhaustive-deps

  const kpiKey = KPI_OPTS[kpiLabel];
  const filtered = viewRows.filter((r) => statusSel.includes(r.status) && r.name.toLowerCase().includes(q.toLowerCase()));
  const completed = filtered.filter((r) => r.status === 'completed');
  const best = completed.length
    ? completed.reduce((a, b) => (num(b[kpiKey as keyof LeaderboardRow]) > num(a[kpiKey as keyof LeaderboardRow]) ? b : a))
    : null;

  const nComp = completed.length;
  const bestLift = completed.reduce((m, r) => Math.max(m, num(r.avg_lift)), 0);
  const bestCov = completed.reduce((m, r) => Math.max(m, num(r.coverage_pct)), 0);

  const ranked = [...filtered].sort((a, b) => (a.status !== 'completed' ? 1 : -1) - (b.status !== 'completed' ? 1 : -1) || (num(b[kpiKey as keyof LeaderboardRow]) - num(a[kpiKey as keyof LeaderboardRow])));

  if (readErr) {
    return <div><h1>Leaderboard</h1><Alert kind="error">Could not read the experiment database: {readErr}</Alert></div>;
  }

  if (!rows.length) {
    return (
      <div>
        <h1>🏆 Leaderboard — Best Experiment per Dataset</h1>
        <Caption>Rank every experiment run on a dataset by a performance KPI and spot the winner. No date filter — every saved run counts.</Caption>
        <Alert kind="info">No experiments yet. Run a configuration in Module 3 · Execution Console — it is saved automatically and shows up here.</Alert>
      </div>
    );
  }

  const tableRows = ranked.map((r, i) => ({
    '#': i + 1,
    '🏆': best && best.exp_id === r.exp_id ? '🏆 Best' : '',
    Experiment: r.name,
    Dataset: r.dataset_name || '(unnamed)',
    Status: r.status,
    'Avg Lift ×': num(r.avg_lift, 3),
    'Max Lift ×': num(r.max_lift, 3),
    'Coverage %': num(r.coverage_pct, 1),
    'Cumulative Event Capture %': num(r.cumulative_event_capture, 1),
    Segments: r.segments_count,
    Rows: r.data_rows,
    'Time (s)': num(r.execution_time_sec, 2),
  }));

  const viewResults = (expId: string) => {
    sessionStorage.setItem('m4_exp_id', expId);
    nav('/m4');
  };

  const doDelete = async (expId: string) => {
    setBusy('Deleting…');
    try {
      await post('/api/m5/delete', { exp_id: expId });
      setConfirmDel(null);
      await load();
    } catch (e) { setNotice({ kind: 'error', text: (e as Error).message }); }
    setBusy('');
  };

  const cloneToWorkbench = async (expId: string) => {
    setBusy('Cloning…');
    try {
      const c = await api<{ cfg: Record<string, unknown> }>(`/api/m5/clone/${expId}`);
      sessionStorage.setItem('wb_pending_run', JSON.stringify(c.cfg));
      nav('/m2');
    } catch (e) { setNotice({ kind: 'error', text: (e as Error).message }); }
    setBusy('');
  };

  const doDuplicate = async (expId: string) => {
    setBusy('Duplicating…');
    try {
      await post('/api/m5/duplicate', { exp_id: expId });
      await load();
      setNotice({ kind: 'success', text: 'Experiment duplicated.' });
    } catch (e) { setNotice({ kind: 'error', text: (e as Error).message }); }
    setBusy('');
  };

  return (
    <div>
      <h1>🏆 Leaderboard — Best Experiment per Dataset</h1>
      <Caption>Rank every experiment run on a dataset by a performance KPI and spot the winner. No date filter — every saved run counts.</Caption>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}
      {busy && <Caption>{busy}</Caption>}

      <div className="row">
        <label style={{ margin: 0 }}>Dataset</label>
        <select value={dataset} onChange={(e) => setDataset(e.target.value)} style={{ width: 260 }}>
          <option>All datasets</option>
          {dsList.map((d) => <option key={d}>{d}</option>)}
        </select>
        <label style={{ margin: 0 }}>Rank by</label>
        <select value={kpiLabel} onChange={(e) => setKpiLabel(e.target.value)} style={{ width: 240 }}>
          {Object.keys(KPI_OPTS).map((k) => <option key={k}>{k}</option>)}
        </select>
        <label style={{ margin: 0 }}>Status</label>
        <span className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
          {statuses.map((s) => (
            <label key={s} className="chip-check">
              <input type="checkbox" checked={statusSel.includes(s)} onChange={(e) => setStatusSel((p) => e.target.checked ? [...p, s] : p.filter((x) => x !== s))} />
              {s}
            </label>
          ))}
        </span>
        <input type="text" placeholder="Search by name…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 200 }} />
      </div>

      {filtered.length === 0 ? (
        <Alert kind="warning">No experiments match the current filters.</Alert>
      ) : (
        <>
          <div className="metrics">
            <Metric label="Experiments" value={filtered.length} sub={`Dataset: ${dataset}`} />
            <Metric label="Completed" value={nComp} />
            <Metric label="Best avg lift" value={`${bestLift.toFixed(2)}×`} />
            <Metric label="Best coverage" value={`${bestCov.toFixed(1)}%`} />
          </div>
          {best && (
            <Alert kind="success">
              🏆 <b>Best performer</b> ({kpiLabel}): <b>{best.name}</b> — avg lift {num(best.avg_lift, 2)}×, max lift {num(best.max_lift, 2)}×, coverage {num(best.coverage_pct, 1)}%, {best.segments_count} segments.
            </Alert>
          )}

          <DataTable rows={tableRows} maxHeight={440} />

          <div className="divider" />
          <h3>Actions</h3>
          {ranked.map((r) => (
            <Card key={r.exp_id}>
              <div className="spread">
                <b>{best && best.exp_id === r.exp_id ? '🏆 ' : ''}{r.name} · {r.status}</b>
                {r.status !== 'completed' && <Caption>⚠️ {r.error_msg || 'Run did not complete.'}</Caption>}
              </div>
              <div className="row" style={{ marginTop: 8 }}>
                <button className="small" onClick={() => cloneToWorkbench(r.exp_id)}>Clone → Workbench</button>
                <button className="small" onClick={() => viewResults(r.exp_id)}>View results</button>
                <button className="small" onClick={() => api<Record<string, unknown>>(`/api/m5/export/${r.exp_id}/json`).then((j) => downloadText(JSON.stringify(j, null, 2), `${r.exp_id}.json`))}>Download run JSON</button>
                <button className="small" onClick={() => doDuplicate(r.exp_id)}>Duplicate</button>
                <button className="small danger" onClick={() => setConfirmDel(r.exp_id)}>🗑 Delete</button>
              </div>
              {confirmDel === r.exp_id && (
                <Alert kind="warning">
                  Delete <b>{r.name}</b>? This cannot be undone.
                  <div className="row" style={{ marginTop: 8 }}>
                    <button className="danger" onClick={() => doDelete(r.exp_id)}>Yes, delete</button>
                    <button onClick={() => setConfirmDel(null)}>Cancel</button>
                  </div>
                </Alert>
              )}
            </Card>
          ))}

          <div className="divider" />
          <h3>Compare two runs</h3>
          {ranked.length >= 2 ? (
            <ComparePanel rows={ranked} a={cmpA} b={cmpB} setA={setCmpA} setB={setCmpB} />
          ) : (
            <Alert kind="info">Need at least two runs to compare.</Alert>
          )}
        </>
      )}
    </div>
  );
}

function ComparePanel({ rows, a, b, setA, setB }: {
  rows: LeaderboardRow[]; a: string; b: string;
  setA: (s: string) => void; setB: (s: string) => void;
}) {
  const opts = rows.map((r) => r.exp_id);
  useEffect(() => {
    if (!a && opts.length) setA(opts[0]);
    if (!b && opts.length) setB(opts[Math.min(1, opts.length - 1)]);
  }, [rows.join('|')]); // eslint-disable-line react-hooks/exhaustive-deps

  const [diffs, setDiffs] = useState<{ parameter: string; run_a: string; run_b: string }[] | null>(null);

  const runCompare = async () => {
    if (a === b) { setDiffs([]); return; }
    try {
      const r = await api<{ diffs: { parameter: string; run_a: string; run_b: string }[]; identical: boolean }>(`/api/m5/compare?run_a=${a}&run_b=${b}`);
      setDiffs(r.diffs);
    } catch (e) {
      setDiffs([]);
    }
  };

  return (
    <Card>
      <div className="grid2">
        <Field label="Run A"><select value={a} onChange={(e) => setA(e.target.value)}>{rows.map((r) => <option key={r.exp_id} value={r.exp_id}>{r.name}</option>)}</select></Field>
        <Field label="Run B"><select value={b} onChange={(e) => setB(e.target.value)}>{rows.map((r) => <option key={r.exp_id} value={r.exp_id}>{r.name}</option>)}</select></Field>
      </div>
      {a === b ? <Caption>Pick two different runs to compare.</Caption> : (
        <>
          <button style={{ marginTop: 8 }} onClick={runCompare}>Compare configs</button>
          {diffs !== null && diffs.length > 0 && <DataTable rows={diffs.map((d) => ({ Parameter: d.parameter, 'Run A': d.run_a, 'Run B': d.run_b }))} maxHeight={300} />}
          {diffs !== null && diffs.length === 0 && <Alert kind="success">Configs are identical.</Alert>}
        </>
      )}
    </Card>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label>{label}</label>
      {children}
    </div>
  );
}

function downloadText(content: string, filename: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
  a.download = filename;
  a.click();
}
