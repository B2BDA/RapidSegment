import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, post } from '../api';
import type { ExperimentRef, Figure, Segment } from '../types';
import { Alert, Caption, Card, DataTable, Metric, PlotChart } from '../components/ui';

interface SummaryPayload {
  experiment: Record<string, any> | null;
  source: string;
  segments: Segment[];
  coverage: Record<string, any>[];
  scorecard: any;
  weights: Record<string, number>;
  features: string[];
  cfg: Record<string, any>;
  dataset_available: boolean;
}

function fmtDuration(secs: number): string {
  secs = Math.max(0, Math.round(secs));
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${(secs % 60).toString().padStart(2, '0')}s`;
  return `${Math.floor(secs / 3600)}h ${(secs % 3600 / 60).toFixed(0).padStart(2, '0')}m`;
}

function download(url: string, filename: string) {
  fetch(url).then((r) => r.blob()).then((b) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = filename;
    a.click();
  }).catch(() => {});
}

const downloadBtn = (expId: string) => (urlSuffix: string, filename: string) => download(`/api/m4/export/${expId}/${urlSuffix}`, filename);

export default function M4Results() {
  const nav = useNavigate();
  const [exps, setExps] = useState<ExperimentRef[]>([]);
  const [expId, setExpId] = useState<string>('');
  const [sum, setSum] = useState<SummaryPayload | null>(null);
  const [charts, setCharts] = useState<{ segments: Segment[]; charts: Record<string, Figure | null> } | null>(null);
  const [notice, setNotice] = useState<{ kind: 'success' | 'warning' | 'error' | 'info'; text: string } | null>(null);
  const [busy, setBusy] = useState('');

  useEffect(() => {
    api<{ rows: ExperimentRef[] }>('/api/m4/experiments').then((r) => {
      setExps(r.rows);
      if (r.rows.length) {
        const stored = sessionStorage.getItem('m4_exp_id');
        const match = stored ? r.rows.find((x) => x.exp_id === stored) : undefined;
        setExpId(match ? match.exp_id : r.rows[0].exp_id);
        sessionStorage.removeItem('m4_exp_id');
      }
    }).catch((e) => setNotice({ kind: 'error', text: e.message }));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const load = useCallback(async (id: string) => {
    if (!id) return;
    setBusy('Loading…');
    setNotice(null);
    try {
      const s = await api<SummaryPayload>(`/api/m4/summary?exp_id=${encodeURIComponent(id)}`);
      setSum(s);
      if (s.segments.length) {
        try {
          const c = await api<any>(`/api/m4/charts?exp_id=${encodeURIComponent(id)}`);
          setCharts(c);
        } catch { setCharts(null); }
      } else {
        setCharts(null);
      }
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally { setBusy(''); }
  }, []);

  useEffect(() => {
    if (expId) void load(expId);
  }, [expId, load]);

  const exp = sum?.experiment;
  const res = (exp?.result as any) || {};

  if (sum && !exp) {
    return (
      <div>
        <h1>Module 4: Results Dashboard & Visualization</h1>
        <Alert kind="warning">No experiment found. Run an experiment in the Workbench (Module 2) / Execution Console (Module 3) first.</Alert>
        <button onClick={() => nav('/m2')}>Go to Workbench</button>
      </div>
    );
  }

  return (
    <div>
      <h1>Module 4: Results Dashboard & Visualization</h1>
      <Caption>Extracted segments, visualizations, deployable scorecard, and diagnostics.</Caption>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}

      <div className="row">
        <label style={{ margin: 0 }}>Experiment</label>
        <select value={expId} onChange={(e) => setExpId(e.target.value)} style={{ width: 380 }} disabled={!exps.length}>
          {!exps.length && <option>No experiments yet</option>}
          {exps.map((x) => <option key={x.exp_id} value={x.exp_id}>{x.name} · {x.created_at} · {x.status}</option>)}
        </select>
        {busy && <Caption>{busy}</Caption>}
      </div>

      {exp && (
        <>
          <Caption>
            Source: <b>{sum?.source}</b> · `{exp.exp_id}` · target=`{exp.target_col}` · status=`{exp.status}`
          </Caption>

          <h3>Summary</h3>
          <div className="metrics" style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)' }}>
            <Metric label="Segments" value={res.segments_count ?? sum?.segments.length ?? 0} />
            <Metric label="Coverage %" value={`${Number(res.coverage_pct ?? 0).toFixed(2)}`} />
            <Metric label="Avg lift" value={`${Number(res.avg_lift ?? 0).toFixed(2)}x`} />
            <Metric label="Max lift" value={`${Number(res.max_lift ?? 0).toFixed(2)}x`} />
            <Metric label="Baseline rate" value={`${Number(res.baseline_rate_pct ?? 0).toFixed(2)}%`} />
            <Metric label="Elapsed" value={fmtDuration(Number(exp.execution_time_sec ?? 0))} />
          </div>

          <h3>Segments</h3>
          <SegmentsTable segments={sum?.segments || []} coverage={sum?.coverage || []} weights={sum?.weights || {}} />

          <h3>Visualizations</h3>
          <Visualizations expId={expId} charts={charts} datasetAvailable={!!sum?.dataset_available}
            scorecard={sum?.scorecard} onRefreshScorecard={async () => {
              setBusy('Scoring population (StrategicSegmentScore)…');
              try {
                const r = await post<any>('/api/m4/scorecard', { exp_id: expId });
                setNotice({ kind: 'success', text: 'Scorecard refreshed.' });
                void r;
              } catch (e) { setNotice({ kind: 'error', text: (e as Error).message }); } finally { setBusy(''); }
            }}
          />

          <h3>Scorecard</h3>
          <ScorecardView scorecard={sum?.scorecard} />

          <Diagnostics cfg={sum?.cfg || {}} expId={expId} features={sum?.features || []} datasetAvailable={!!sum?.dataset_available} />

          <div className="divider" />
          <h3>Export Hub</h3>
          <ExportHub expId={expId} hasSegments={(sum?.segments.length || 0) > 0} hasScorecard={!!sum?.scorecard} />
        </>
      )}
    </div>
  );
}

function SegmentsTable({ segments, coverage, weights }: { segments: Segment[]; coverage: Record<string, any>[]; weights: Record<string, number> }) {
  if (!segments.length) return <Caption>No segments were produced by this experiment.</Caption>;
  const covBySeg: Record<number, Record<string, any>> = {};
  for (const r of coverage) if (r.segment) covBySeg[Number(r.segment)] = r;
  const cols = ['segment_id', 'rule_string', 'sql_filter', 'count', 'rate', 'lift', 'capture_rate', 'cumulative_event_capture', 'weight'];
  const rows = segments.map((s) => {
    const c = covBySeg[Number(s.segment_id)] || {};
    return {
      segment_id: s.segment_id,
      rule_string: s.rule_string || '',
      sql_filter: s.sql_filter || '',
      count: s.count ?? 0,
      rate: typeof s.rate === 'number' ? `${s.rate.toFixed(2)}%` : (s.rate ?? '—'),
      lift: typeof s.lift === 'number' ? `${s.lift.toFixed(2)}x` : (s.lift ?? '—'),
      capture_rate: c.capture_rate != null ? `${Number(c.capture_rate).toFixed(2)}%` : '—',
      cumulative_event_capture: c.cumulative_event_capture != null ? `${Number(c.cumulative_event_capture).toFixed(2)}%` : '—',
      weight: weights[Number(s.segment_id)] != null ? Number(weights[Number(s.segment_id)]).toFixed(3) : '—',
    };
  });
  return <DataTable rows={rows} columns={cols} maxHeight={420} />;
}

function Visualizations({ expId, charts, datasetAvailable, scorecard, onRefreshScorecard }: {
  expId: string;
  charts: any;
  datasetAvailable: boolean;
  scorecard: any;
  onRefreshScorecard: () => void;
}) {
  const [vis, setVis] = useState(false);
  if (!charts || !charts.charts) {
    return (
      <Caption>
        No visualizations. {datasetAvailable && <button className="small" onClick={onRefreshScorecard}>Generate / refresh scorecard</button>}
      </Caption>
    );
  }
  const c = charts.charts;
  const figs: { key: string; label: string; fig: Figure | null }[] = [
    { key: 'scatter', label: 'Lift vs. Volume', fig: c.scatter },
    { key: 'distribution', label: 'Segment Distribution', fig: c.distribution },
    { key: 'sunburst', label: 'Rule Complexity', fig: c.sunburst },
    { key: 'decile', label: 'Decile Thresholds', fig: c.decile },
    { key: 'feature_importance', label: 'Feature Importance', fig: c.feature_importance },
  ];
  return (
    <div className="grid2" style={{ alignItems: 'start' }}>
      {figs.map((f) => (
        <Card key={f.key} title={f.label}>
          {f.fig ? <PlotChart figure={f.fig} height={f.fig.layout?.height ?? 380} /> : <Caption>Not available — {f.key === 'decile' ? 'generate a scorecard first.' : 'no data.'}</Caption>}
        </Card>
      ))}
    </div>
  );
}

function ScorecardView({ scorecard }: { scorecard: any }) {
  if (!scorecard) return <Caption>No scorecard yet — generate it above (requires segments + dataset).</Caption>;
  const segWeights = scorecard.segment_weights || {};
  const rows = Object.entries(segWeights).map(([seg, w]: any) => ({ segment: seg, weight: w?.weight, ...(w?.metrics || {}) }));
  return (
    <div>
      <pre className="code" style={{ maxHeight: 260, overflow: 'auto' }}>{JSON.stringify(scorecard, null, 2)}</pre>
      {rows.length > 0 && <DataTable rows={rows as Record<string, any>[]} maxHeight={200} />}
    </div>
  );
}

function Diagnostics({ cfg, expId, features, datasetAvailable }: { cfg: Record<string, any>; expId: string; features: string[]; datasetAvailable: boolean }) {
  const [fj, setFj] = useState(features[0] || '');
  const [journey, setJourney] = useState('');
  const [fjBusy, setFjBusy] = useState(false);
  const [healthSel, setHealthSel] = useState<string[]>([]);
  const [healthRows, setHealthRows] = useState<Record<string, any>[] | null>(null);
  const [healthCsv, setHealthCsv] = useState('');
  const [noSeg, setNoSeg] = useState('');

  useEffect(() => {
    if (features.length && !fj) setFj(features[0]);
    if (!healthSel.length) setHealthSel(features.slice(0, 5));
  }, [features.join('|')]); // eslint-disable-line react-hooks/exhaustive-deps

  const showJourney = async () => {
    setFjBusy(true);
    try {
      const r = await post<{ text: string }>('/api/m4/feature-journey', { feature: fj, exp_id: expId });
      setJourney(r.text || '(no journey recorded for this feature)');
    } catch (e) {
      setJourney((e as Error).message);
    } finally { setFjBusy(false); }
  };

  const showHealth = async () => {
    try {
      const r = await post<{ rows: Record<string, any>[]; csv: string }>('/api/m4/feature-health', { features: healthSel, exp_id: expId });
      setHealthRows(r.rows);
      setHealthCsv(r.csv);
    } catch (e) {
      setHealthRows([]);
      setHealthCsv('');
      alert((e as Error).message);
    }
  };

  const explainNoSeg = async () => {
    try {
      const r = await post<{ text: string }>('/api/m4/explain-no-segments', { exp_id: expId });
      setNoSeg(r.text);
    } catch (e) {
      setNoSeg((e as Error).message);
    }
  };

  const toggleFeature = (f: string) => {
    setHealthSel((p) => (p.includes(f) ? p.filter((x) => x !== f) : [...p, f]));
  };

  return (
    <div>
      <h3>Diagnostic Drilldown</h3>

      <div className="card">
        <h3>Feature Journey (audit trail per feature)</h3>
        {!features.length ? <Caption>No features were tracked (no dataset / diagnostics available).</Caption> : (
          <>
            <div className="row">
              <select value={fj} onChange={(e) => setFj(e.target.value)} style={{ flex: 1 }}>
                {features.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
              <button onClick={showJourney} disabled={fjBusy}>{fjBusy ? '…' : 'Show feature journey'}</button>
            </div>
            {journey && <pre className="code" style={{ marginTop: 8, maxHeight: 260, overflow: 'auto' }}>{journey}</pre>}
          </>
        )}
      </div>

      <div className="card">
        <h3>Feature Health Report (bin-level stats)</h3>
        {!features.length ? <Caption>No features to profile (no dataset / diagnostics available).</Caption> : (
          <>
            <Caption>Features to profile</Caption>
            <div className="row" style={{ gap: 4 }}>
              {features.map((f) => (
                <button key={f} className={`small${healthSel.includes(f) ? ' active' : ''}`}
                  style={healthSel.includes(f) ? { background: 'rgba(52,211,153,0.25)' } : undefined}
                  onClick={() => toggleFeature(f)}>{f}</button>
              ))}
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <button onClick={showHealth}>Generate health report</button>
              {healthRows && <button className="small" onClick={() => downloadText(healthCsv, 'feature_health.csv')}>Download health report (CSV)</button>}
            </div>
            {healthRows && <DataTable rows={healthRows} maxHeight={360} />}
          </>
        )}
      </div>

      <div className="card">
        <h3>Why did it stop? (no-segments explanation)</h3>
        <button onClick={explainNoSeg}>Run full diagnostics</button>
        {noSeg && <pre className="code" style={{ marginTop: 8, maxHeight: 260, overflow: 'auto' }}>{noSeg}</pre>}
      </div>
    </div>
  );
}

function downloadText(content: string, filename: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: 'text/plain' }));
  a.download = filename;
  a.click();
}

function ExportHub({ expId, hasSegments, hasScorecard }: { expId: string; hasSegments: boolean; hasScorecard: boolean }) {
  const dl = downloadBtn(expId);
  const e = expId;
  return (
    <div>
      <div className="row">
        <button className="small" onClick={() => dl('segments.csv', `segments_${e}.csv`)} disabled={!hasSegments}>Segments (CSV)</button>
        <button className="small" onClick={() => dl('coverage.csv', `coverage_${e}.csv`)} disabled={!hasSegments}>Coverage (CSV)</button>
        <button className="small" onClick={() => dl('config.json', `config_${e}.json`)}>Config (JSON)</button>
        <button className="small" onClick={() => dl('sql.sql', `segments_${e}.sql`)} disabled={!hasSegments}>SQL (deployable)</button>
        <button className="small" onClick={() => dl('report.html', `report_${e}.html`)}>Report (HTML)</button>
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <button className="small" onClick={() => dl('scorecard.json', `scorecard_${e}.json`)} disabled={!hasScorecard}>Scorecard (JSON)</button>
        <button className="small" onClick={() => dl('zip', `rapidsegment_${e}.zip`)}>Download ALL (ZIP)</button>
      </div>
    </div>
  );
}
