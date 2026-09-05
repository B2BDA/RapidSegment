import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, post } from '../api';
import type { M1Status, Quality, SummarizeRow, TInfo } from '../types';
import { Alert, Caption, Card, CopyButton, DataTable, Metric, PlotChart, Tabs } from '../components/ui';

const BAR_COLORS = ['#6366f1', '#f59e0b', '#22c55e', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6', '#f97316', '#8b5cf6', '#06b6d4'];
const OVERRIDE_OPTS = ['AUTO', 'NUMERIC', 'CATEGORICAL'];

async function loadFile(path: string, encoding: string, datasetName?: string) {
  const r = await post<{ error?: string }>('/api/m1/load/path', { path, encoding });
  void r;
  return datasetName;
}

export default function M1DataLoader() {
  const nav = useNavigate();
  const [status, setStatus] = useState<M1Status | null>(null);
  const [quality, setQuality] = useState<Quality | null>(null);
  const [preview, setPreview] = useState<{ columns: string[]; rows: Record<string, any>[] } | null>(null);
  const [samples, setSamples] = useState<{ name: string; path: string }[]>([]);
  const [hints, setHints] = useState<string[]>([]);

  const [datasetName, setDatasetName] = useState('');
  const [source, setSource] = useState<'Local File' | 'BigQuery' | 'Sample Datasets'>('Local File');
  const [method, setMethod] = useState<'File path' | 'Drag & drop upload'>('File path');
  const [fp, setFp] = useState('');
  const [encoding, setEncoding] = useState('Auto-detect');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [bqTable, setBqTable] = useState('');
  const [tab, setTab] = useState('Preview');
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState<{ kind: 'success' | 'warning' | 'error' | 'info'; text: string } | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const s = await api<M1Status>('/api/m1/status');
    setStatus(s);
    if (s.dataset_name && !datasetName) setDatasetName(s.dataset_name);
    if (s.loaded) {
      setQuality(await api<Quality>('/api/m1/quality'));
    }
  }, [datasetName]);

  useEffect(() => {
    refresh().catch((e) => setNotice({ kind: 'error', text: e.message }));
  }, [refresh]);

  useEffect(() => {
    if (!status?.loaded) {
      api<{ samples: { name: string; path: string }[] }>('/api/m1/samples').then((r) => setSamples(r.samples)).catch(() => {});
      api<{ hints: string[] }>('/api/m1/smart-defaults').then((r) => setHints(r.hints)).catch(() => {});
    }
  }, [status?.loaded]);

  const saveDatasetName = async () => {
    try {
      await post('/api/m1/dataset-name', { name: datasetName });
    } catch {
      /* non-fatal */
    }
  };

  const loaded = status?.loaded ?? false;

  // ── Load actions ─────────────────────────────────────────────────────────────
  const doLoad = async (path: string, enc: string, name: string) => {
    setBusy(`Loading ${name}…`);
    setNotice(null);
    try {
      await post('/api/m1/load/path', { path, encoding: enc });
      const s = await api<M1Status>('/api/m1/status');
      setStatus(s);
      if (s.loaded) setQuality(await api<Quality>('/api/m1/quality'));
      setNotice({ kind: 'success', text: `Loaded: ${s.dataset_name || name}` });
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally {
      setBusy('');
    }
  };

  const doUpload = async () => {
    if (!uploadFile) return;
    setBusy(`Loading '${uploadFile.name}'…`);
    setNotice(null);
    const fd = new FormData();
    fd.append('file', uploadFile);
    try {
      const res = await fetch('/api/m1/load/upload', { method: 'POST', body: fd });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.detail || res.statusText);
      }
      const s = await res.json();
      setStatus(s);
      if (s.loaded) setQuality(await api<Quality>('/api/m1/quality'));
      setUploadFile(null);
      if (fileInput.current) fileInput.current.value = '';
      setNotice({ kind: 'success', text: `Loaded '${uploadFile.name}'` });
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally {
      setBusy('');
    }
  };

  const doLoadSample = async (name: string) => {
    setBusy(`Loading ${name}…`);
    setNotice(null);
    try {
      const s = await post<M1Status>('/api/m1/load/sample', { name });
      setStatus(s);
      if (s.loaded) setQuality(await api<Quality>('/api/m1/quality'));
      setNotice({ kind: 'success', text: `Loaded sample: ${name}` });
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally {
      setBusy('');
    }
  };

  const doLoadBQ = async () => {
    setBusy('Loading from BigQuery…');
    setNotice(null);
    try {
      const s = await post<M1Status>('/api/m1/load/bigquery', { table_ref: bqTable });
      setStatus(s);
      if (s.loaded) setQuality(await api<Quality>('/api/m1/quality'));
      setNotice({ kind: 'success', text: `Loaded ${bqTable}` });
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally {
      setBusy('');
    }
  };

  const doFetchPreview = async () => {
    setPreview(await api('/api/m1/preview'));
  };

  useEffect(() => {
    if (loaded && tab === 'Preview' && !preview) doFetchPreview().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, tab]);

  // ── Title ────────────────────────────────────────────────────────────────────
  return (
    <div>
      <h1>Module 1: Data Source & Profiling</h1>
      <div className="caption">Universal data loader. Drop-in replacement for the Streamlit Module 1.</div>

      <div className="row" style={{ margin: '6px 0' }}>
        <label style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 6 }}>
          Dataset name
          <input
            type="text" value={datasetName} style={{ width: 260 }}
            onChange={(e) => setDatasetName(e.target.value)}
            onBlur={() => { if (datasetName.trim()) void saveDatasetName(); }}
            placeholder="Used to group experiments in the Leaderboard."
          />
        </label>
      </div>

      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}

      {!loaded ? (
        <div>
          <Alert kind="info">Load a dataset from the source panel to get started.</Alert>
          <DataSource
            source={source} setSource={setSource}
            method={method} setMethod={setMethod}
            fp={fp} setFp={setFp} encoding={encoding} setEncoding={setEncoding}
            uploadFile={uploadFile} setUploadFile={setUploadFile} fileInput={fileInput}
            bqTable={bqTable} setBqTable={setBqTable}
            samples={samples} hints={hints}
            busy={busy} doLoad={doLoad} doUpload={doUpload}
            doLoadSample={doLoadSample} doLoadBQ={doLoadBQ}
          />
        </div>
      ) : (
        <LoadedView
          status={status} quality={quality}
          setQuality={setQuality} setStatus={setStatus} setNotice={setNotice}
          refreshAll={async () => {
            const s = await api<M1Status>('/api/m1/status');
            setStatus(s);
            if (s.loaded) setQuality(await api<Quality>('/api/m1/quality'));
          }}
          setDatasetName={setDatasetName}
          onReset={() => {
            setStatus(null); setQuality(null); setPreview(null); setTab('Preview');
          }}
          goWorkbench={() => {
            const t = status?.tinfo;
            if (t?.is_binary) nav('/m2');
          }}
          tab={tab} setTab={setTab} preview={preview} busy={busy}
        />
      )}
    </div>
  );
}

function PosPicker({ dist, busy, onBinarize }: { dist: { val: string | null; cnt: number }[]; busy: string; onBinarize: (v: string) => void }) {
  const [pos, setPos] = useState<string>('');
  const nonNull = dist.filter((d) => d.val !== null);
  useEffect(() => {
    if (!pos && nonNull.length) setPos(String(nonNull[0].val));
  }, [nonNull.length]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      <select value={pos} onChange={(e) => setPos(e.target.value)}>
        {nonNull.map((d) => <option key={String(d.val)} value={String(d.val)}>{d.val}</option>)}
      </select>
      <button className="primary" style={{ marginTop: 8 }} disabled={!!busy || !pos} onClick={() => onBinarize(pos)}>
        {busy || 'Binarize into 0/1 column'}
      </button>
    </>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
interface DataSourceProps {
  source: string; setSource: (v: any) => void;
  method: string; setMethod: (v: any) => void;
  fp: string; setFp: (v: string) => void;
  encoding: string; setEncoding: (v: string) => void;
  uploadFile: File | null; setUploadFile: (f: File | null) => void;
  fileInput: React.RefObject<HTMLInputElement>;
  bqTable: string; setBqTable: (v: string) => void;
  samples: { name: string; path: string }[];
  hints: string[];
  busy: string;
  doLoad: (path: string, encoding: string, name: string) => void;
  doUpload: () => void;
  doLoadSample: (name: string) => void;
  doLoadBQ: () => void;
}

function DataSource(props: DataSourceProps) {
  const { source, setSource, method, setMethod, fp, setFp, encoding, setEncoding,
    uploadFile, setUploadFile, fileInput, bqTable, setBqTable, samples, hints,
    busy, doLoad, doUpload, doLoadSample, doLoadBQ } = props;

  return (
    <div className="grid2" style={{ alignItems: 'start' }}>
      <div className="card">
        <h3>Data Source</h3>
        <select value={source} onChange={(e) => setSource(e.target.value)}>
          {['Local File', 'BigQuery', 'Sample Datasets'].map((s) => <option key={s}>{s}</option>)}
        </select>

        {source === 'Local File' && (
          <>
            <label>Input method</label>
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              {['File path', 'Drag & drop upload'].map((s) => <option key={s}>{s}</option>)}
            </select>
            {method === 'File path' ? (
              <>
                {hints.length > 0 && (
                  <Caption>Smart defaults — detected: {hints.map((h) => h.split('/').pop()).join(', ')}</Caption>
                )}
                <label>File path</label>
                <input type="text" value={fp} onChange={(e) => setFp(e.target.value)} placeholder="/path/to/file.csv" />
                <label>Encoding</label>
                <select value={encoding} onChange={(e) => setEncoding(e.target.value)}>
                  {['Auto-detect', 'UTF-8', 'Latin-1'].map((e) => <option key={e}>{e}</option>)}
                </select>
                <Caption>Supported: CSV · Parquet · Arrow/Feather · Excel</Caption>
                <div style={{ marginTop: 12 }}>
                  <button className="primary" disabled={!fp || !!busy} onClick={() => doLoad(fp, encoding, fp.split('/').pop() || 'file')}>
                    {busy || 'Load File'}
                  </button>
                </div>
              </>
            ) : (
              <>
                <label>Drag & drop file here</label>
                <input
                  ref={fileInput} type="file"
                  accept=".csv,.tsv,.parquet,.pq,.arrow,.feather,.xlsx,.xls"
                  onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                />
                {uploadFile && (
                  <>
                    <Caption>Detected format: {uploadFile.name.split('.').pop()?.toUpperCase()} · {(uploadFile.size / 1e6).toFixed(1)} MB</Caption>
                    {uploadFile.size / 1e6 > 1000 && (
                      <Alert kind="warning">Large browser uploads are slow. For multi-GB files use the File path method instead.</Alert>
                    )}
                    <label>Encoding</label>
                    <select value={encoding} onChange={(e) => setEncoding(e.target.value)}>
                      {['Auto-detect', 'UTF-8', 'Latin-1'].map((e) => <option key={e}>{e}</option>)}
                    </select>
                    <div style={{ marginTop: 12 }}>
                      <button className="primary" disabled={!!busy} onClick={doUpload}>{busy || 'Load Uploaded File'}</button>
                    </div>
                  </>
                )}
              </>
            )}
          </>
        )}

        {source === 'BigQuery' && (
          <>
            <Caption>Authentication uses your environment credentials (gcloud auth application-default login or GOOGLE_APPLICATION_CREDENTIALS) — no secrets are stored in the app.</Caption>
            <label>BigQuery table</label>
            <input type="text" value={bqTable} onChange={(e) => setBqTable(e.target.value)} placeholder="project_id.dataset_id.table_id" />
            <div style={{ marginTop: 12 }}>
              <button className="primary" disabled={!bqTable || !!busy} onClick={doLoadBQ}>{busy || 'Load table'}</button>
            </div>
            <BQBrowser />
          </>
        )}

        {source === 'Sample Datasets' && (
          <>
            {samples.length > 0 ? (
              <>
                <label>Quick-start dataset</label>
                <select value={samples[0]?.name} onChange={(e) => {
                  const s = samples.find((x) => x.name === e.target.value);
                  if (s) doLoadSample(s.name);
                }}>
                  {samples.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
                </select>
                <Caption>Found at: {samples[0]?.path}</Caption>
                <div style={{ marginTop: 12 }}>
                  <button className="primary" disabled={!!busy} onClick={() => doLoadSample(samples[0].name)}>
                    {busy || `Load ${samples[0].name}`}
                  </button>
                </div>
              </>
            ) : (
              <Alert kind="warning">No sample datasets found. Drop bank-full.csv / train.csv into ./data/ or provide a manual path.</Alert>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function BQBrowser() {
  const [pid, setPid] = useState('');
  const [step, setStep] = useState<'idle' | 'datasets' | 'tables' | 'preview'>('idle');
  const [datasets, setDatasets] = useState<string[]>([]);
  const [dataset, setDataset] = useState('');
  const [tables, setTables] = useState<string[]>([]);
  const [table, setTable] = useState('');
  const [err, setErr] = useState('');
  const [preview, setPreview] = useState<Record<string, any>[] | null>(null);

  const browse = async (action: string, body: Record<string, unknown>) => {
    setErr('');
    try {
      const r = await post<{ datasets?: string[]; tables?: string[]; rows?: Record<string, any>[]; columns?: string[] }>(`/api/m1/browse/bigquery`, { action, ...body });
      if (r.datasets) { setDatasets(r.datasets); setStep('datasets'); }
      if (r.tables) { setTables(r.tables); setStep('tables'); }
      if (r.rows) { setPreview(r.rows); setStep('preview'); }
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  return (
    <div style={{ marginTop: 8 }}>
      <div className="caption" style={{ margin: '8px 0' }}>Browse BigQuery (optional discovery) — lists datasets/tables using your env credentials.</div>
      <label>GCP Project ID (for browsing)</label>
      <input type="text" value={pid} onChange={(e) => setPid(e.target.value)} placeholder="my-gcp-project" />
      <div className="row" style={{ margin: '8px 0' }}>
        <button className="small" disabled={!pid} onClick={() => browse('datasets', { project: pid })}>List datasets</button>
      </div>
      {err && <Alert kind="error">{err}</Alert>}
      {step === 'datasets' && datasets.length > 0 && (
        <>
          <label>Dataset</label>
          <select value={dataset} onChange={(e) => { setDataset(e.target.value); setTables([]); setStep('tables'); }}>
            <option value="">—</option>
            {datasets.map((d) => <option key={d}>{d}</option>)}
          </select>
          {dataset && (
            <button className="small" style={{ marginTop: 8 }} onClick={() => browse('tables', { project: pid, action: 'tables', dataset })}>List tables</button>
          )}
        </>
      )}
      {step === 'tables' && tables.length > 0 && dataset && (
        <>
          <label>Table</label>
          <select value={table} onChange={(e) => setTable(e.target.value)}>
            <option value="">—</option>
            {tables.map((t) => <option key={t}>{t}</option>)}
          </select>
          {table && (
            <button className="small" style={{ marginTop: 8 }} onClick={() => browse('preview', { project: pid, action: 'preview', dataset, table })}>Preview (first 1000 rows)</button>
          )}
        </>
      )}
      {step === 'preview' && preview && (
        <>
          <div className="caption" style={{ margin: '6px 0' }}>BigQuery streaming preview (first 1000 rows)</div>
          <DataTable rows={preview.slice(0, 100)} maxHeight={220} />
        </>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
function LoadedView(props: any) {
  const { status, quality, setQuality, setStatus, setNotice, refreshAll, setDatasetName, onReset, goWorkbench, tab, setTab, preview, busy } = props;
  const [viewBusy, setViewBusy] = useState('');

  const tinfo = status?.tinfo as TInfo | null | undefined;

  const doMaterialize = async () => {
    setViewBusy('Applying metadata…');
    try {
      const r = await post<{ message: string; target_col: string }>('/api/m1/materialize', {});
      await refreshAll();
      setNotice({ kind: 'success', text: r.message });
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally { setViewBusy(''); }
  };

  const doReset = async () => {
    setViewBusy('Resetting…');
    try {
      await post('/api/m1/reset', {});
      onReset();
      setNotice({ kind: 'success', text: 'Dataset removed. Load a new file to continue.' });
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally { setViewBusy(''); }
  };

  const downloadText = (content: string, filename: string) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([content], { type: 'text/plain' }));
    a.download = filename;
    a.click();
  };

  const doDownloadProfiling = async () => {
    try {
      const r = await api<{ csv: string; json: string }>('/api/m1/profiling-report');
      downloadText(r.csv, 'profiling_report.csv');
      downloadText(r.json, 'profiling_report.json');
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    }
  };

  return (
    <>
      <Tabs tabs={['Preview', 'Quality Report', 'Column Metadata', 'Target Selection']} active={tab} onChange={setTab} />

      {tab === 'Preview' && (
        <Card>
          <div className="spread">
            <h3>Preview — first 100 rows</h3>
            <Caption>DuckDB: {status.active_db}</Caption>
          </div>
          {preview ? <DataTable columns={preview.columns} rows={preview.rows} maxHeight={400} /> : <Caption>Loading…</Caption>}
        </Card>
      )}

      {tab === 'Quality Report' && quality && <QualityTab quality={quality} />}

      {tab === 'Column Metadata' && quality && (
        <MetadataTab
          quality={quality} status={status}
          viewBusy={viewBusy} doMaterialize={doMaterialize} doReset={doReset}
          setNotice={setNotice} refreshAll={refreshAll}
        />
      )}

      {tab === 'Target Selection' && (
        <TargetTab quality={quality} status={status} setStatus={setStatus} setQuality={setQuality}
          setNotice={setNotice} busy={viewBusy} setBusy={setViewBusy}
          goWorkbench={goWorkbench}
        />
      )}

      <div className="divider" />
      <h3>Actions</h3>
      <div className="row">
        <button className="primary" disabled={!tinfo?.is_binary} onClick={goWorkbench}>Proceed to Workbench</button>
        {!tinfo?.is_binary && <Caption>Enabled after a binary target is validated.</Caption>}
        <button disabled={!!viewBusy} onClick={doReset}>Upload Different File</button>
        <button className="small" disabled={!!viewBusy} onClick={doDownloadProfiling}>Download Profiling Report (CSV)</button>
        <button className="small" disabled={!!viewBusy} onClick={doDownloadProfiling}>Download Profiling Report (JSON)</button>
      </div>
      {busy && <Caption>{busy}</Caption>}
    </>
  );
}

function QualityTab({ quality }: { quality: Quality }) {
  const nullOffenders = useMemo(() => {
    return (quality.summarize || [])
      .filter((s) => Number(s.null_percentage) > 0)
      .sort((a, b) => Number(b.null_percentage) - Number(a.null_percentage));
  }, [quality]);

  const warns = (quality.profiling || []).filter((p) => p['Warning'] !== '✓');

  return (
    <Card title="Data Quality Report">
      <div className="metrics">
        <Metric label="Rows" value={quality.n_rows.toLocaleString()} />
        <Metric label="Columns" value={quality.n_cols} />
        <Metric label="Numeric / Categorical" value={`${quality.n_numeric} / ${quality.n_categorical}`} />
        <Metric label="On-disk size (DuckDB)" value={`${quality.db_mb} MB`} />
      </div>
      <div className="divider" />
      {nullOffenders.length === 0 ? (
        <Alert kind="success">No missing values — data ready for segmentation.</Alert>
      ) : (
        <div>
          <Alert kind="warning">{nullOffenders.length} column(s) have missing values (worst offenders first).</Alert>
          <DataTable
            rows={nullOffenders.map((s) => ({ Column: s.column_name, 'Null %': `${Number(s.null_percentage).toFixed(1)}%` }))}
            maxHeight={260}
          />
        </div>
      )}
      {warns.length === 0 ? (
        <Alert kind="success">No type warnings — data ready for segmentation.</Alert>
      ) : (
        <Alert kind="warning">{warns.length} column(s) flagged with type/cardinality warnings.</Alert>
      )}
      <Caption>DuckDB file size: {quality.db_mb} MB</Caption>
    </Card>
  );
}

function MetadataTab({ quality, status, viewBusy, doMaterialize, doReset, setNotice, refreshAll }: any) {
  const [ov, setOv] = useState<Record<string, string>>({});
  const cols = (quality.summarize || []).map((s: SummarizeRow) => String(s.column_name));

  useEffect(() => {
    setOv((prev) => {
      const next: Record<string, string> = {};
      for (const c of cols) next[c] = (status?.type_overrides?.[c]) || 'AUTO';
      return { ...prev, ...next };
    });
  }, [cols.join('|')]); // eslint-disable-line react-hooks/exhaustive-deps

  const applyOverrides = async () => {
    setNotice(null);
    try {
      await post('/api/m1/overrides', { overrides: ov });
      await refreshAll();
      setNotice({ kind: 'success', text: 'Overrides stored — click "Apply metadata" below to materialize them.' });
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    }
  };

  return (
    <Card title="Column Metadata">
      <div className="spread">
        <h3 style={{ margin: 0 }}>Manual type overrides</h3>
        <button className="small" onClick={applyOverrides} disabled={!!viewBusy}>Apply type overrides</button>
      </div>
      <DataTable
        rows={cols.map((c: string) => ({ Column: c }))}
        maxHeight={300}
      />
      <div style={{ maxHeight: 300, overflow: 'auto', marginTop: 8 }}>
        <table className="data">
          <thead><tr><th>Column</th><th style={{ width: 180 }}>Override</th></tr></thead>
          <tbody>
            {cols.map((c: string) => (
              <tr key={c}>
                <td>{c}</td>
                <td>
                  <select value={ov[c] || 'AUTO'} onChange={(e) => setOv((p) => ({ ...p, [c]: e.target.value }))}>
                    {OVERRIDE_OPTS.map((o) => <option key={o}>{o}</option>)}
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="divider" />
      <h3>Materialize modified dataset</h3>
      <Caption>Write a transformed copy (module1_data_modified.duckdb) with the type overrides applied and the target converted to 1/0. Downstream modules read this copy automatically.</Caption>
      <div className="row">
        <button className="primary" onClick={doMaterialize} disabled={!!viewBusy}>{viewBusy || 'Apply metadata & create modified dataset'}</button>
        {status.data_modified && (
          <>
            <Alert kind="success" >Modified dataset is active.</Alert>
            <button onClick={doReset} disabled={!!viewBusy}>Discard modified dataset (revert to raw)</button>
          </>
        )}
      </div>

      <div className="divider" />
      <h3>Effective column metadata</h3>
      <DataTable rows={(quality.profiling || []) as Record<string, any>[]} maxHeight={420} />
    </Card>
  );
}

function TargetTab({ quality, status, setStatus, setQuality, setNotice, busy, setBusy, goWorkbench }: any) {
  const [onlyBinary, setOnlyBinary] = useState(true);
  const [sel, setSel] = useState<string>('');
  const [tinfo, setTinfo] = useState<TInfo | null>(null);

  const summ: SummarizeRow[] = quality?.summarize || [];
  const labelOf = (c: string) => {
    const u = summ.find((s) => s.column_name === c)?.approx_unique;
    const card = u ?? 99;
    return `${c}${card <= 2 ? '  ★' : ''}`;
  };
  const displayNames = useMemo(() => {
    return summ
      .filter((s) => !onlyBinary || Number(s.approx_unique ?? 99) <= 2)
      .map((s) => labelOf(String(s.column_name)));
  }, [summ, onlyBinary]); // eslint-disable-line react-hooks/exhaustive-deps

  const candidates = useMemo(() => {
    return summ
      .filter((s) => !onlyBinary || Number(s.approx_unique ?? 99) <= 2)
      .map((s) => String(s.column_name));
  }, [summ, onlyBinary]);

  // preselect from status
  useEffect(() => {
    const t = status?.target_col as string | undefined;
    if (t && candidates.includes(t)) setSel(t);
    else if (candidates.length && !sel) setSel(candidates[0]);
  }, [status?.target_col, candidates.join('|')]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (status?.tinfo) setTinfo(status.tinfo);
  }, [status?.tinfo]);

  const validate = async () => {
    setBusy('Validating…');
    setNotice(null);
    try {
      const r = await post<TInfo>('/api/m1/target/validate', { target_col: sel });
      setTinfo(r);
      const s = await api<M1Status>('/api/m1/status');
      setStatus(s);
      if (s.loaded) setQuality(await api<Quality>('/api/m1/quality'));
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally { setBusy(''); }
  };

  const binarize = async (pos: string) => {
    setBusy('Binarizing…');
    try {
      const r = await post<{ message: string; target_col: string }>('/api/m1/materialize', { positive_value: pos });
      await api<M1Status>('/api/m1/status').then(setStatus);
      setTinfo(null);
      setNotice({ kind: 'success', text: r.message });
    } catch (e) {
      setNotice({ kind: 'error', text: (e as Error).message });
    } finally { setBusy(''); }
  };

  const dist = tinfo?.dist || [];
  const colors = BAR_COLORS.slice(0, dist.length || 1);
  const chart = dist.length
    ? {
        data: [{ type: 'bar', x: dist.map((d) => d.val), y: dist.map((d) => d.cnt), marker: { color: colors }, text: dist.map((d) => d.cnt.toLocaleString()), textposition: 'outside' }],
        layout: { title: `Class distribution — ${tinfo?.col}`, xaxis: { title: 'Value' }, yaxis: { title: 'Count' } },
      }
    : null;

  return (
    <Card title="Target Column Selection & Validation">
      <label className="row" style={{ gap: 6, cursor: 'pointer' }}>
        <input type="checkbox" checked={onlyBinary} onChange={(e) => setOnlyBinary(e.target.checked)} />
        Only show likely binary columns (≤ 2 unique values)
      </label>

      {candidates.length === 0 && (
        <Alert kind="warning">No binary candidates found — uncheck the filter to see all columns.</Alert>
      )}
      {candidates.length > 0 && (
        <>
          <label>Target column (★ = ≤ 2 unique values — likely binary)</label>
          <select value={sel} onChange={(e) => setSel(e.target.value)}>
            {displayNames.map((d, i) => <option key={candidates[i]} value={candidates[i]}>{d}</option>)}
          </select>
        </>
      )}

      <div style={{ margin: '10px 0' }}>
        <button className="primary" disabled={!sel || !!busy} onClick={validate}>{busy || 'Validate Target'}</button>
      </div>

      {tinfo && (
        <>
          <div className="divider" />
          {tinfo.is_binary ? (
            <Alert kind="success">Binary column — encoding: {tinfo.binary_label}</Alert>
          ) : (
            <Alert kind="warning">Multi-class — {tinfo.n_distinct} distinct values. Use the binarization helper below.</Alert>
          )}
          {tinfo.is_binary && tinfo.event_rate != null && (
            <div className="metrics">
              <Metric label="Event Rate" value={`${(tinfo.event_rate * 100).toFixed(2)}%`} />
              {tinfo.event_rate < 0.01 || tinfo.event_rate > 0.99
                ? <Metric label="Balance" value="⚠ Severe class imbalance" />
                : <Metric label="Balance" value="Healthy range" />}
            </div>
          )}

          {!tinfo.is_binary && (
            <div className="card">
              <h3>Binarization helper</h3>
              <Caption>Multi-class target? Pick the positive value to create a 0/1 column (also applies type overrides and writes the modified dataset).</Caption>
              <label>Positive (event) value</label>
              <PosPicker dist={dist} busy={busy} onBinarize={binarize} />
            </div>
          )}

          {tinfo.null_pct > 0 && (
            <Alert kind="error">Target column has {tinfo.null_pct.toFixed(1)}% nulls ({tinfo.null_count.toLocaleString()} rows).</Alert>
          )}

          {chart && (
            <PlotChart figure={chart} height={350} />
          )}

          <div className="row">
            <button className="primary" disabled={!tinfo.is_binary} onClick={goWorkbench}>Proceed to Workbench</button>
          </div>
        </>
      )}
    </Card>
  );
}
