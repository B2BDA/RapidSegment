import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, post, del } from '../api';
import type { LeaderboardRow, M2Options } from '../types';
import { Alert, Caption, Card, Metric } from '../components/ui';

interface Group { name: string; cols: string[] }

export interface WBCfg {
  experiment_name: string;
  description: string;
  data_table: string;
  target_col: string;
  primary_key: string;
  top_n_vars: number;
  max_segments: number;
  max_feature_reuse: number;
  feature_groups: Record<string, string[]>;
  enable_diversity: boolean;
  ignore_features: string[];
  binning_method: string;
  naive_bins: number;
  max_expansion_hops: number;
  enable_1way: boolean;
  enable_2way: boolean;
  enable_3way: boolean;
  selection_metric: string;
  min_sample_size: number;
  min_lift: number;
  min_events: number;
  param_grid: { min_sample_size: number[]; min_lift: number[] } | null;
  sort_priority: string;
  n_jobs: number;
  expand_log_mode: string;
}

function fmtDuration(secs: number): string {
  secs = Math.max(0, Math.round(secs));
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${(secs % 60).toString().padStart(2, '0')}s`;
  return `${Math.floor(secs / 3600)}h ${(Math.floor(secs % 3600) / 60).toFixed(0).padStart(2, '0')}m`;
}

export function wbEstimate(cfg: WBCfg, nRows: number): number {
  const pg = cfg.param_grid;
  const combos = Math.max(1, pg?.min_sample_size?.length || 1) * Math.max(1, pg?.min_lift?.length || 1);
  let base = (Math.max(nRows, 1000) / 100_000.0) * 6.0;
  base *= cfg.binning_method === 'naive' ? 0.45 : 2.0;
  base *= 1.0 + 0.35 * cfg.max_expansion_hops;
  if (cfg.enable_2way) base *= 1.4;
  if (cfg.enable_3way) base *= 2.1;
  base *= (cfg.max_segments / 10.0) * Math.max(1.0, cfg.top_n_vars / 15.0);
  return Math.max(15.0, base * combos);
}

function toName(s: string) {
  return s;
}

export default function M2Workbench() {
  const nav = useNavigate();
  const [loaded, setLoaded] = useState(false);
  const [nRows, setNRows] = useState(0);
  const [cols, setCols] = useState<string[]>([]);
  const [colInfo, setColInfo] = useState<Record<string, { column_type: string; approx_unique: number; likely_binary: boolean }>>({});
  const [tinfo, setTinfo] = useState<any>(null);
  const [targetFromM1, setTargetFromM1] = useState<string | null>(null);
  const [options, setOptions] = useState<M2Options | null>(null);
  const [templates, setTemplates] = useState<Record<string, Record<string, any>>>({});
  const [lbRows, setLbRows] = useState<LeaderboardRow[]>([]);
  const [notice, setNotice] = useState<{ kind: 'success' | 'error' | 'warning'; text: string } | null>(null);
  const [preset, setPreset] = useState('Quick Discovery');
  const [templateName, setTemplateName] = useState('');
  const [cloneSel, setCloneSel] = useState('');
  const [serverIssues, setServerIssues] = useState<string[]>([]);
  const [newGroupName, setNewGroupName] = useState('');

  const addGroup = () => {
    const name = newGroupName.trim();
    if (!name || groups.some((g) => g.name === name)) {
      setNotice({ kind: 'warning', text: 'Category name is empty or already exists.' });
      return;
    }
    setGroups((p) => [...p, { name, cols: [] }]);
    setNewGroupName('');
    setNotice(null);
  };

  // ── Form state ──────────────────────────────────────────────────────────────
  const [f, setF] = useState<WBCfg>({
    experiment_name: '', description: '', data_table: 'udl_data', target_col: '', primary_key: '(none)',
    top_n_vars: 15, max_segments: 10, max_feature_reuse: 1, feature_groups: {}, enable_diversity: false,
    ignore_features: [], binning_method: 'optimal_cart', naive_bins: 5, max_expansion_hops: 0,
    enable_1way: true, enable_2way: true, enable_3way: true, selection_metric: 'iv',
    min_sample_size: 1000, min_lift: 1.5, min_events: 100, param_grid: null,
    sort_priority: 'rate_lift_count', n_jobs: -1, expand_log_mode: 'none',
  });
  const [groups, setGroups] = useState<Group[]>([]);
  const [enableGrid, setEnableGrid] = useState(false);
  const [gridSizes, setGridSizes] = useState('500, 1000, 2000');
  const [gridLifts, setGridLifts] = useState('1.5, 2.0, 3.0');

  const set = (patch: Partial<WBCfg>) => setF((p) => ({ ...p, ...patch }));

  // ── Boot ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      const [st, opts, presetsRes, templatesRes, lbRes] = await Promise.all([
        api<any>('/api/m2/status'),
        api<M2Options>('/api/m2/options'),
        api<any>('/api/m2/presets'),
        api<any>('/api/m2/templates'),
        api<any>('/api/m2/leaderboard'),
      ]);
      setOptions(opts);
      setTemplates(templatesRes.templates || {});
      setLbRows(lbRes.rows || []);
      setLoaded(st.loaded);
      setNRows(st.n_rows || 0);
      if (st.loaded) {
        const colRes = await api<any>('/api/m2/columns');
        setCols(colRes.columns || []);
        setColInfo(colRes.info || {});
        setTargetFromM1(st.target_col || colRes.target || null);
        const t0 = st.target_col || colRes.target || '';
        setTinfo(st.tinfo || null);
        setF((p) => ({
          ...p, target_col: t0, data_table: st.data_table || 'udl_data',
          experiment_name: p.experiment_name || defaultName(),
        }));
      }
    })().catch((e) => setNotice({ kind: 'error', text: e.message }));
  }, []);

  const defaultName = () => {
    const d = new Date();
    const p = (n: number) => String(n).padStart(2, '0');
    return `exp_${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}`;
  };

  const synced = useMemo(() => buildCfg(f, groups, enableGrid, gridSizes, gridLifts), [f, groups, enableGrid, gridSizes, gridLifts]);

  if (!options) return <Caption>Loading workbench…</Caption>;

  if (!loaded) {
    return (
      <div>
        <h1>Module 2: The Workbench</h1>
        <Alert kind="warning">No dataset loaded yet. Run <b>Module 1 (Data Loader)</b> first, validate a binary target column, then come back to the Workbench.</Alert>
        <button onClick={() => nav('/m1')}>Go to Module 1</button>
      </div>
    );
  }

  return (
    <div>
      <h1>Module 2: The Workbench</h1>
      <Caption>Dataset: {f.data_table} · {nRows.toLocaleString()} rows · {cols.length} columns</Caption>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}

      <div className="grid2" style={{ gridTemplateColumns: '3fr 1.55fr', alignItems: 'start' }}>
        {/* ── Left column: form ── */}
        <div>
          <Section title="Basic Settings">
            <div className="grid2">
              <Field label="Experiment Name"><input type="text" value={f.experiment_name} onChange={(e) => set({ experiment_name: e.target.value })} /></Field>
              <Field label="Data table name"><input type="text" value={f.data_table} onChange={(e) => set({ data_table: e.target.value })} /></Field>
            </div>
            <Field label="Description (optional)"><textarea rows={3} value={f.description} onChange={(e) => set({ description: e.target.value })} /></Field>
            <div className="grid3">
              <Field label="Target column">
                <select value={f.target_col} onChange={(e) => set({ target_col: e.target.value })}>
                  {cols.map((c) => <option key={c} value={c}>{c}{colInfo[c]?.likely_binary ? ' ★' : ''}</option>)}
                </select>
              </Field>
              <Field label="Primary key column (for scorecard)">
                <select value={f.primary_key} onChange={(e) => set({ primary_key: e.target.value })}>
                  <option value="(none)">(none)</option>
                  {cols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </Field>
            </div>
          </Section>

          <Section title="Segment Discovery Strategy">
            <div className="grid3">
              <SliderField label="top_n_vars" min={5} max={50} value={f.top_n_vars} onChange={(v) => set({ top_n_vars: v })} />
              <SliderField label="max_segments" min={1} max={20} value={f.max_segments} onChange={(v) => set({ max_segments: v })} />
              <SliderField label="max_feature_reuse" min={1} max={5} value={f.max_feature_reuse} onChange={(v) => set({ max_feature_reuse: v })} />
            </div>
            <div className="grid2">
              <Field label="Sort priority (champion ranking)">
                <select value={f.sort_priority} onChange={(e) => set({ sort_priority: e.target.value })}>
                  {options.sort_priority_options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </Field>
              <Field label="Parallel jobs">
                <select value={String(f.n_jobs)} onChange={(e) => set({ n_jobs: Number(e.target.value) })}>
                  {options.n_jobs_options.map((o) => <option key={o} value={options.n_jobs_map[o]}>{o}</option>)}
                </select>
              </Field>
            </div>

            <div className="card" style={{ marginTop: 10 }}>
              <h3>Feature grouping (business categories)</h3>
              <Caption>Diversity toggle prevents rules that mix features from the same group.</Caption>
              <div className="row">
                <input type="text" id="wb_new_group" placeholder="e.g. Delinquency" value={newGroupName} onChange={(e) => setNewGroupName(e.target.value)} style={{ flex: 1 }} />
                <button onClick={addGroup}>Add category</button>
              </div>
              {groups.map((g, gi) => (
                <div key={g.name} className="card">
                  <div className="spread">
                    <b>{g.name}</b>
                    <button className="danger small" onClick={() => { setGroups((p) => p.filter((_, i) => i !== gi)); }}>Remove</button>
                  </div>
                  <MultiSelect columns={cols} value={g.cols} onChange={(cols_) => setGroups((p) => p.map((x, i) => (i === gi ? { ...x, cols: cols_ } : x)))} />
                </div>
              ))}
            </div>

            <Field label="Ignore features">
              <MultiSelect columns={cols} value={f.ignore_features} onChange={(v) => set({ ignore_features: v })} />
            </Field>
            <Checkbox label="Enable diversity (prevent mixing groups in one rule)" checked={f.enable_diversity} onChange={(v) => set({ enable_diversity: v })} />
          </Section>

          <Section title="Binning & Rule Complexity">
            <Field label="Binning method">
              <div className="row">
                {options.bin_options.map((b) => (
                  <label key={b} className="row" style={{ gap: 4, cursor: 'pointer' }}>
                    <input type="radio" checked={b === options.bin_rmap[f.binning_method]} onChange={() => set({ binning_method: options.bin_map[b] })} />
                    {b}
                  </label>
                ))}
              </div>
            </Field>
            <div className="grid2">
              {f.binning_method === 'naive'
                ? <SliderField label="Naive bins" min={3} max={20} value={f.naive_bins} onChange={(v) => set({ naive_bins: v })} />
                : <Caption>Naive bins only apply to the 'Naive' binning method.</Caption>}
              <SliderField label="Max expansion hops" min={0} max={5} value={f.max_expansion_hops} onChange={(v) => set({ max_expansion_hops: v })} />
            </div>
            <div className="row" style={{ margin: '10px 0' }}>
              <Checkbox label="Enable 1-way rules" checked={f.enable_1way} onChange={(v) => set({ enable_1way: v })} />
              <Checkbox label="Enable 2-way rules" checked={f.enable_2way} onChange={(v) => set({ enable_2way: v })} />
              <Checkbox label="Enable 3-way rules" checked={f.enable_3way} onChange={(v) => set({ enable_3way: v })} />
            </div>
            <div className="grid2">
              <Field label="Selection metric">
                <select value={f.selection_metric} onChange={(e) => set({ selection_metric: e.target.value })}>
                  {options.metric_options.map((m) => <option key={m} value={options.metric_map[m]}>{m}</option>)}
                </select>
              </Field>
              <Field label="Expansion log mode">
                <select value={f.expand_log_mode} onChange={(e) => set({ expand_log_mode: e.target.value })}>
                  {options.expand_log_options.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              </Field>
            </div>
          </Section>

          <Section title="Hard Constraints">
            <div className="grid3">
              <Field label="min_sample_size">
                <input type="number" min={100} max={10000000} step={100} value={f.min_sample_size} onChange={(e) => set({ min_sample_size: Number(e.target.value) || 100 })} />
              </Field>
              <Field label="min_lift">
                <input type="number" min={0.5} max={20} step={0.1} value={f.min_lift} onChange={(e) => set({ min_lift: Number(e.target.value) })} />
              </Field>
              <Field label="min_events">
                <input type="number" min={1} max={1000000} step={10} value={f.min_events} onChange={(e) => set({ min_events: Number(e.target.value) || 1 })} />
              </Field>
            </div>
          </Section>

          <Section title="Advanced: Grid Search (Optional)">
            <Checkbox label="Enable grid search" checked={enableGrid} onChange={setEnableGrid} />
            {enableGrid ? (
              <>
                <div className="grid2">
                  <Field label={`min_sample_size values (comma-separated, ${options.grid_size_range[0]}-${options.grid_size_range[1]})`}>
                    <input type="text" value={gridSizes} onChange={(e) => setGridSizes(e.target.value)} placeholder="e.g. 500, 1000, 2000, 5000" />
                  </Field>
                  <Field label={`min_lift values (comma-separated, ${options.grid_lift_range[0]}-${options.grid_lift_range[1]})`}>
                    <input type="text" value={gridLifts} onChange={(e) => setGridLifts(e.target.value)} placeholder="e.g. 1.5, 2.0, 3.0" />
                  </Field>
                </div>
                <GridInfo cfg={synced} />
              </>
            ) : (
              <Caption>Grid search disabled — single (min_sample_size, min_lift) pair will be used.</Caption>
            )}
          </Section>
        </div>

        {/* ── Right column: preset + summary + checklist ── */}
        <div>
          <Card title="Preset templates">
            <select value={preset} onChange={(e) => setPreset(e.target.value)}>
              {['Quick Discovery', 'Conservative', 'Last experiment', ...Object.keys(templates)].map((t) => <option key={t}>{t}</option>)}
            </select>
            <button className="wide" style={{ marginTop: 8 }} onClick={() => applyPreset(preset, templates, lbRows, f, setF, setGroups, setEnableGrid, setGridSizes, setGridLifts, setNotice)}>Apply Preset</button>
          </Card>

          <Summary f={synced} options={options} />
          <Checklist
            cfg={synced} tinfo={tinfo} selTarget={f.target_col} nRows={nRows} nCols={cols.length}
            serverIssues={serverIssues} onValidate={() => validateCfg(synced, setServerIssues, setNotice)}
          />
        </div>
      </div>

      {/* ── Action bar ── */}
      <div className="card" style={{ borderTop: '1px solid rgba(52,211,153,0.4)' }}>
        <div className="grid3">
          <div>
            <label>Template name</label>
            <input type="text" value={templateName} placeholder="e.g. FastBinning" onChange={(e) => setTemplateName(e.target.value)} />
            <button className="wide" style={{ marginTop: 8 }} onClick={async () => {
              const name = templateName.trim() || f.experiment_name.trim();
              if (!name) { setNotice({ kind: 'error', text: 'Enter a template name first.' }); return; }
              try {
                const r = await post<any>('/api/m2/templates', { name, cfg: buildCfg(f, groups, enableGrid, gridSizes, gridLifts) });
                setTemplates(r.templates || {});
                setNotice({ kind: 'success', text: `Template '${r.name}' saved to templates.json` });
              } catch (e) { setNotice({ kind: 'error', text: (e as Error).message }); }
            }}>Save as Template</button>
          </div>
          <div>
            <label>Clone from Leaderboard</label>
            <select value={cloneSel} onChange={(e) => setCloneSel(e.target.value)} disabled={lbRows.length === 0}>
              {lbRows.length === 0
                ? <option>No experiments yet</option>
                : lbRows.map((r) => <option key={r.exp_id} value={r.exp_id}>{r.name} · {r.created_at.slice(0, 16).replace('T', ' ')}</option>)}
            </select>
            <button className="wide" style={{ marginTop: 8 }} disabled={lbRows.length === 0} onClick={async () => {
              try {
                const cfg = await api<Record<string, any>>(`/api/m5/clone/${cloneSel}`);
                applyConfigInto(cfg, setF, setGroups, setEnableGrid, setGridSizes, setGridLifts, setNotice);
              } catch (e) { setNotice({ kind: 'error', text: (e as Error).message }); }
            }}>Apply Clone</button>
            {lbRows.length === 0 && <Caption>Leaderboard is empty — run an experiment first.</Caption>}
          </div>
          <div>
            <label>Estimated time: {fmtDuration(wbEstimate(synced, nRows))}</label>
            <button className="primary wide" style={{ marginTop: 8 }} onClick={() => runExperiment(synced, nav, setNotice)}>Run Experiment</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Config builder (mirrors build_params + apply_config) ────────────────────
export function buildCfg(
  f: WBCfg, groups: Group[], enableGrid: boolean, gridSizes: string, gridLifts: string,
): WBCfg {
  const featureGroups: Record<string, string[]> = {};
  for (const g of groups) if (g.name.trim()) featureGroups[g.name] = [...g.cols];
  let paramGrid: WBCfg['param_grid'] = null;
  if (enableGrid) {
    const sizes: number[] = [];
    const lifts: number[] = [];
    for (const v of gridSizes.split(',')) {
      const n = parseInt(v.trim(), 10);
      if (!Number.isNaN(n) && n >= 1000 && n <= 20000) sizes.push(n);
    }
    for (const v of gridLifts.split(',')) {
      const n = parseFloat(v.trim());
      if (!Number.isNaN(n) && n >= 1.0 && n <= 10.0) lifts.push(n);
    }
    if (sizes.length || lifts.length) {
      paramGrid = {
        min_sample_size: sizes.length ? sizes : [f.min_sample_size],
        min_lift: lifts.length ? lifts : [f.min_lift],
      };
    }
  }
  const name = f.experiment_name.trim() || defaultNameNow();
  return {
    ...f,
    feature_groups: featureGroups,
    experiment_name: name,
    param_grid: paramGrid,
    primary_key: f.primary_key === '(none)' ? '' : f.primary_key,
  };
}

function defaultNameNow() {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `exp_${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}`;
}

// ── Validation / run helpers ────────────────────────────────────────────────
function runExperiment(cfg: WBCfg, nav: (p: string) => void, setNotice: (n: any) => void) {
  post<any>('/api/m2/run', { cfg })
    .then((r) => {
      if (!r.ok) {
        setNotice({ kind: 'error', text: 'Validation failed:\n' + (r.issues || []).map((i: string) => `• ${i}`).join('\n') });
        return;
      }
      sessionStorage.setItem('wb_pending_run', JSON.stringify(cfg));
      nav('/m3');
    })
    .catch((e) => setNotice({ kind: 'error', text: (e as Error).message }));
}

function validateCfg(cfg: WBCfg, setServerIssues: (i: string[]) => void, setNotice: (n: any) => void) {
  post<any>('/api/m2/config/build', { cfg })
    .then((r) => setServerIssues(r.issues || []))
    .catch((e) => setNotice({ kind: 'error', text: (e as Error).message }));
}

// ── Apply preset / clone / template into the form ───────────────────────────
function applyConfigInto(
  src: Record<string, any>,
  setF: any, setGroups: any, setEnableGrid: any, setGridSizes: any, setGridLifts: any,
  setNotice: any,
) {
  if (!src || typeof src !== 'object') { setNotice({ kind: 'error', text: 'Selected config is empty.' }); return; }
  const cfg = src;
  setF((p: WBCfg) => {
    const next: WBCfg = { ...p };
    const map: Record<string, keyof WBCfg> = {
      experiment_name: 'experiment_name', description: 'description', data_table: 'data_table',
      target_col: 'target_col', top_n_vars: 'top_n_vars', max_segments: 'max_segments',
      max_feature_reuse: 'max_feature_reuse', enable_diversity: 'enable_diversity',
      ignore_features: 'ignore_features', binning_method: 'binning_method', naive_bins: 'naive_bins',
      max_expansion_hops: 'max_expansion_hops', enable_1way: 'enable_1way', enable_2way: 'enable_2way',
      enable_3way: 'enable_3way', selection_metric: 'selection_metric', min_sample_size: 'min_sample_size',
      min_lift: 'min_lift', min_events: 'min_events', sort_priority: 'sort_priority',
      n_jobs: 'n_jobs', expand_log_mode: 'expand_log_mode',
    };
    for (const [k, kk] of Object.entries(map)) {
      const v = cfg[k];
      if (v === undefined || v === null || v === '') continue;
      (next as any)[kk] = v;
    }
    if (cfg.primary_key === undefined || cfg.primary_key === null || cfg.primary_key === '') next.primary_key = '(none)';
    else next.primary_key = String(cfg.primary_key);
    return next;
  });
  const fg = cfg.feature_groups;
  if (fg && typeof fg === 'object') {
    setGroups(Object.entries(fg).map(([name, colsArr]) => ({ name, cols: Array.isArray(colsArr) ? (colsArr as string[]) : [] })));
  }
  const pg = cfg.param_grid;
  if (pg && typeof pg === 'object') {
    setEnableGrid(true);
    if (Array.isArray(pg.min_sample_size)) setGridSizes(pg.min_sample_size.join(', '));
    if (Array.isArray(pg.min_lift)) setGridLifts(pg.min_lift.join(', '));
  } else {
    setEnableGrid(false);
  }
  setNotice({ kind: 'success', text: 'Preset applied.' });
}

function applyPreset(
  preset: string,
  templates: Record<string, Record<string, any>>,
  lbRows: LeaderboardRow[],
  f: WBCfg,
  setF: any, setGroups: any, setEnableGrid: any, setGridSizes: any, setGridLifts: any,
  setNotice: any,
) {
  let src: Record<string, any> | null = null;
  if (preset === 'Quick Discovery') src = QUICK_DISCOVERY;
  else if (preset === 'Conservative') src = CONSERVATIVE;
  else if (preset === 'Last experiment') {
    if (lbRows.length) src = lbRows[0].config || null;
    else { setNotice({ kind: 'error', text: 'No previous experiment found to clone.' }); return; }
  } else {
    src = templates[preset] || null;
    if (!src) { setNotice({ kind: 'error', text: `Template '${preset}' not found.` }); return; }
  }
  if (src) applyConfigInto({ ...src, target_col: src.target_col || f.target_col }, setF, setGroups, setEnableGrid, setGridSizes, setGridLifts, setNotice);
}

const QUICK_DISCOVERY: Record<string, any> = {
  experiment_name: 'Quick Discovery', description: 'Aggressive discovery: fast naive binning, wide search.',
  top_n_vars: 20, max_segments: 15, max_feature_reuse: 2, enable_diversity: false, ignore_features: [],
  binning_method: 'naive', naive_bins: 5, max_expansion_hops: 1, enable_1way: true, enable_2way: true,
  enable_3way: true, selection_metric: 'iv', min_sample_size: 500, min_lift: 1.2, min_events: 50,
  param_grid: null, sort_priority: 'rate_lift_count', n_jobs: -1, expand_log_mode: 'none',
};
const CONSERVATIVE: Record<string, any> = {
  experiment_name: 'Conservative', description: 'Strict constraints, stable optimal quantile binning.',
  top_n_vars: 10, max_segments: 5, max_feature_reuse: 1, enable_diversity: false, ignore_features: [],
  binning_method: 'optimal_quantile', naive_bins: 5, max_expansion_hops: 0, enable_1way: true,
  enable_2way: true, enable_3way: false, selection_metric: 'response_rate', min_sample_size: 5000,
  min_lift: 2.0, min_events: 500, param_grid: null, sort_priority: 'rate_lift_count', n_jobs: -1,
  expand_log_mode: 'none',
};

// ── Small presentational helpers ────────────────────────────────────────────
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card title={title}>
      {children}
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

function SliderField({ label, min, max, value, onChange }: { label: string; min: number; max: number; value: number; onChange: (v: number) => void }) {
  return (
    <div>
      <label>{label} — {value}</label>
      <input type="range" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} style={{ width: '100%' }} />
    </div>
  );
}

function Checkbox({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="row" style={{ gap: 6, cursor: 'pointer', margin: '8px 0' }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

function MultiSelect({ columns, value, onChange }: { columns: string[]; value: string[]; onChange: (v: string[]) => void }) {
  const toggle = (c: string) => {
    if (value.includes(c)) onChange(value.filter((x) => x !== c));
    else onChange([...value, c]);
  };
  const sel = new Set(value);
  return (
    <div className="card" style={{ margin: '6px 0' }}>
      <div className="row" style={{ gap: 4 }}>
        {columns.map((c) => (
          <button
            key={c}
            className={`small${sel.has(c) ? ' active' : ''}`}
            style={sel.has(c) ? { background: 'rgba(52,211,153,0.25)' } : undefined}
            onClick={() => toggle(c)}
          >{c}</button>
        ))}
        {columns.length === 0 && <Caption>No columns available.</Caption>}
      </div>
    </div>
  );
}

function Summary({ f, options }: { f: WBCfg; options: M2Options }) {
  const binLabel = options.bin_rmap[f.binning_method] || f.binning_method;
  return (
    <>
      <Card title="Real-Time Summary">
        <b>Segment Discovery</b>
        <Caption>top_n_vars={f.top_n_vars} · max_segments={f.max_segments} · max_feature_reuse={f.max_feature_reuse}</Caption>
        <Caption>diversity={f.enable_diversity ? 'on' : 'off'} · {Object.keys(f.feature_groups).length} group(s) · {f.ignore_features.length} ignored</Caption>
        <b>Rule Complexity</b>
        <Caption>binning={binLabel}{f.binning_method === 'naive' ? ` · ${f.naive_bins} bins` : ''}</Caption>
        <Caption>hops={f.max_expansion_hops} · 1-way={f.enable_1way ? 'on' : 'off'} · 2-way={f.enable_2way ? 'on' : 'off'} · 3-way={f.enable_3way ? 'on' : 'off'} · metric={f.selection_metric}</Caption>
        <b>Constraints</b>
        <Caption>min_sample_size={f.min_sample_size.toLocaleString()} · min_lift={f.min_lift} · min_events={f.min_events}</Caption>
        {f.param_grid && (
          <Caption>Grid: {f.param_grid.min_sample_size.length} sizes × {f.param_grid.min_lift.length} lifts = {f.param_grid.min_sample_size.length * f.param_grid.min_lift.length} combinations</Caption>
        )}
      </Card>
    </>
  );
}

function GridInfo({ cfg }: { cfg: WBCfg }) {
  const pg = cfg.param_grid;
  if (!pg) return <Caption>Enter at least one value for min_sample_size or min_lift.</Caption>;
  const combos = pg.min_sample_size.length * pg.min_lift.length;
  return <Caption>Evaluating <b>{combos}</b> combination(s).</Caption>;
}

function Checklist({ cfg, tinfo, selTarget, nRows, nCols, serverIssues, onValidate }: any) {
  const issues: string[] = serverIssues || [];
  const targetOk = Boolean(tinfo && tinfo.is_binary && tinfo.col === selTarget);
  const eventRate = tinfo && tinfo.col === selTarget ? tinfo.event_rate : null;
  const imbalance = eventRate != null && (eventRate < 0.01 || eventRate > 0.99);
  const gridOn = Boolean(cfg.param_grid);
  const items: { label: string; ok: boolean; note: string }[] = [
    { label: 'Target column selected', ok: targetOk, note: `\`${selTarget}\` — ` + (targetOk ? 'validated as binary in Module 1' : 'not validated as binary') },
    { label: 'Data loaded', ok: true, note: `${nRows.toLocaleString()} rows · ${nCols} columns` },
    { label: 'Class imbalance detected', ok: !imbalance, note: eventRate != null ? `Event rate ${(eventRate * 100).toFixed(2)}%` : 'No validated event rate' },
    { label: 'Parameters valid', ok: issues.length === 0, note: issues[0] || 'All checks passed' },
  ];
  return (
    <Card title="Validation Checklist">
      {items.map((it) => (
        <div key={it.label} style={{ margin: '4px 0' }}>
          <div>{it.ok ? '✅' : '⚠️'} <b>{it.label}</b></div>
          <Caption>{it.note}</Caption>
        </div>
      ))}
      <div>
        <div>{gridOn ? '⏱️' : '—'} <b>Grid search time estimate</b></div>
        <Caption>{gridOn ? `Evaluating ${(cfg.param_grid.min_sample_size.length) * (cfg.param_grid.min_lift.length)} combination(s)` : 'Grid search disabled'}</Caption>
      </div>
      {issues.length > 1 && (
        <div className="alert alert-warning">
          {issues.map((i, k) => <div key={k}>• {i}</div>)}
        </div>
      )}
      <button className="small" onClick={onValidate} style={{ marginTop: 8 }}>Re-check parameters</button>
    </Card>
  );
}
