import React, { useEffect, useState } from 'react';
import { api } from '../api';
import { Alert, Caption, Card, DataTable, Metric, PlotChart, Tabs } from '../components/ui';

interface ArenaKpi { metric: string; run_a: number; run_b: number; higher_better: boolean; winner: 'A' | 'B' | 'tie' }
interface ArenaParam { parameter: string; run_a: string; run_b: string; different: boolean }
interface ArenaOverlap { shared_count: number; unique_a: number; unique_b: number; jaccard: number; shared: { rule: string; a_lift: number; b_lift: number; delta_lift: number }[] }
interface ArenaCompare {
  run_a: { exp_id: string; name: string; status: string; created_at: string; target_col: string };
  run_b: { exp_id: string; name: string; status: string; created_at: string; target_col: string };
  kpis: ArenaKpi[];
  param_diff: ArenaParam[];
  overlap: ArenaOverlap;
  sql_diff: { rule: string; run_a_sql: string; run_b_sql: string }[];
  lift_distribution: { run_a: { x: number; y: number }[]; run_b: { x: number; y: number }[] };
  has_segments: boolean;
}

const winnerTxt = (w: string) => (w === 'tie' ? 'tie' : `${w} 🏆`);

export default function M6Arena() {
  const [rows, setRows] = useState<{ exp_id: string; name: string; created_at: string; status: string }[]>([]);
  const [aId, setAId] = useState('');
  const [bId, setBId] = useState('');
  const [tab, setTab] = useState('KPI Face-off');
  const [cmp, setCmp] = useState<ArenaCompare | null>(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    api<{ rows: { exp_id: string; name: string; created_at: string; status: string }[] }>('/api/m6/experiments')
      .then((r) => {
        setRows(r.rows);
        if (r.rows.length) {
          setAId(r.rows[0].exp_id);
          setBId(r.rows[Math.min(1, r.rows.length - 1)].exp_id);
        }
      })
      .catch((e) => setErr((e as Error).message));
  }, []);

  useEffect(() => {
    if (!aId || !bId || aId === bId) return;
    setCmp(null);
    api<ArenaCompare>(`/api/m6/compare?run_a=${aId}&run_b=${bId}`).then(setCmp).catch((e) => setErr((e as Error).message));
  }, [aId, bId]);

  const label = (id: string) => {
    const r = rows.find((x) => x.exp_id === id);
    return r ? `${r.name}  (${String(r.created_at).slice(0, 19)})` : id;
  };

  if (err && !rows.length) return <div><h1>6 · Arena — 1v1 Comparison</h1><Alert kind="error">{err}</Alert></div>;

  if (!rows.length) {
    return (
      <div>
        <h1>6 · Arena — 1v1 Comparison</h1>
        <Caption>Pick two experiments and compare them head-to-head across KPIs, parameters, segments and generated SQL.</Caption>
        <Alert kind="info">No experiments found. Run a segmentation in the Workbench first.</Alert>
      </div>
    );
  }

  const num2 = (v: unknown) => (Number(v ?? 0) % 1 ? Number(Number(v).toFixed(3)) : Number(v ?? 0));

  const figure = cmp ? {
    data: [
      ...(cmp.lift_distribution.run_a.length ? [{
        x: cmp.lift_distribution.run_a.map((p) => p.x),
        y: cmp.lift_distribution.run_a.map((p) => p.y),
        mode: 'markers' as const, name: 'Run A',
        type: 'scatter' as const, marker: { color: '#58a6ff' },
      }] : []),
      ...(cmp.lift_distribution.run_b.length ? [{
        x: cmp.lift_distribution.run_b.map((p) => p.x),
        y: cmp.lift_distribution.run_b.map((p) => p.y),
        mode: 'markers' as const, name: 'Run B',
        type: 'scatter' as const, marker: { color: '#3fb950' },
      }] : []),
    ],
    layout: { title: 'Segment Lift Distribution', yaxis: { title: 'lift' }, xaxis: { title: 'segment id', range: [1, null] } },
  } : undefined;

  return (
    <div>
      <h1>6 · Arena — 1v1 Comparison</h1>
      <Caption>Pick two experiments and compare them head-to-head across KPIs, parameters, segments and generated SQL.</Caption>

      <div className="grid2">
        <div><label>Run A</label>
          <select value={aId} onChange={(e) => setAId(e.target.value)}>{rows.map((r) => <option key={r.exp_id} value={r.exp_id}>{label(r.exp_id)}</option>)}</select>
        </div>
        <div><label>Run B</label>
          <select value={bId} onChange={(e) => setBId(e.target.value)}>{rows.map((r) => <option key={r.exp_id} value={r.exp_id}>{label(r.exp_id)}</option>)}</select>
        </div>
      </div>

      {cmp ? (
        <Tabs tabs={['KPI Face-off', 'Parameter Diff', 'Segment Comparison', 'SQL Diff']} active={tab} onChange={setTab} />
      ) : (
        <Caption>Loading comparison…</Caption>
      )}

      {cmp && tab === 'KPI Face-off' && (
        <Card title={`${cmp.run_a.name}  vs  ${cmp.run_b.name}`}>
          <DataTable rows={cmp.kpis.map((k) => ({
            Metric: k.metric,
            'Run A': num2(k.run_a),
            'Run B': num2(k.run_b),
            Winner: winnerTxt(k.winner),
          }))} />
        </Card>
      )}

      {cmp && tab === 'Parameter Diff' && (
        <Card title="Full parameter diff — differing fields highlighted">
          <DataTable rows={cmp.param_diff.map((p) => ({
            Parameter: p.parameter,
            'Run A': String(p.run_a),
            'Run B': String(p.run_b),
            'Different?': p.different ? 'YES' : '',
          }))} />
        </Card>
      )}

      {cmp && tab === 'Segment Comparison' && (
        <>
          {!cmp.has_segments ? (
            <Alert kind="info">No segment data available for these runs (artifacts missing).</Alert>
          ) : (
            <>
              <div className="metrics">
                <Metric label="Shared segments" value={cmp.overlap.shared_count} />
                <Metric label="Unique to A" value={cmp.overlap.unique_a} />
                <Metric label="Unique to B" value={cmp.overlap.unique_b} />
                <Metric label="Jaccard overlap" value={cmp.overlap.jaccard.toFixed(2)} />
              </div>
              <PlotChart figure={figure} height={360} />
              {cmp.overlap.shared.length ? (
                <Card title="Shared segments — lift by run">
                  <DataTable rows={cmp.overlap.shared.map((s) => ({
                    Rule: s.rule,
                    'A lift': num2(s.a_lift),
                    'B lift': num2(s.b_lift),
                    'Δ lift': num2(s.delta_lift),
                  }))} maxHeight={340} />
                </Card>
              ) : (
                <Caption>No shared segment rules between these runs.</Caption>
              )}
            </>
          )}
        </>
      )}

      {cmp && tab === 'SQL Diff' && (
        <>
          {!cmp.sql_diff.length ? (
            <Alert kind="info">No segment SQL available for these runs.</Alert>
          ) : (
            <DataTable rows={cmp.sql_diff.map((s) => ({ Rule: s.rule, 'Run A SQL': s.run_a_sql, 'Run B SQL': s.run_b_sql }))} maxHeight={520} />
          )}
        </>
      )}
    </div>
  );
}
