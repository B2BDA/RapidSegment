"""
RapidSegment — Module 1: Universal Data Loader & Profiling
==========================================================
Drop-in replacement for `Module_1_data_loader.py` implementing
Module 1 (Data Source & Profiling) from UI_MVP.md.

Run with:  streamlit run Module_1_data_loader.py
"""
import json
import os
import shutil
import tempfile

import pyarrow.csv as pa_csv
import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rapidsegment.utils.data_loader import UniversalDataLoader
from rapidsegment.ui._theme import apply_cyberpunk_theme

# ── Constants & storage ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "pages" else _HERE
SUITE_DIR = os.path.join(_PROJECT_ROOT, ".rapidsegment_suite")
os.makedirs(SUITE_DIR, exist_ok=True)
DB_FILE = os.path.join(SUITE_DIR, "module1_data.duckdb")
DB_FILE_MOD = os.path.join(SUITE_DIR, "module1_data_modified.duckdb")


def active_db():
    """Return the materialized *modified* dataset if it exists, else the raw load.

    Module 1 writes a transformed copy (`module1_data_modified.duckdb`) when the
    user applies metadata (type overrides + target 1/0). Every downstream read
    goes through this so DuckDB sees the actual changed types, not just the UI.
    """
    return DB_FILE_MOD if os.path.exists(DB_FILE_MOD) else DB_FILE
MAX_UPLOAD_MB = 2000
SAMPLE_NAMES = ["bank-full.csv", "train.csv"]

NUMERIC = {"INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "REAL",
           "SMALLINT", "TINYINT", "HUGEINT", "INT", "UBIGINT", "UINTEGER"}


def is_num(t):
    return any(k in str(t).upper() for k in NUMERIC)


def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ── DuckDB helpers (short-lived connections) ─────────────────────────────────
def db_write(arrow_table):
    con = duckdb.connect(DB_FILE)
    con.execute("DROP TABLE IF EXISTS udl_data")
    con.execute("CREATE TABLE udl_data AS SELECT * FROM arrow_table")
    con.close()


def db_query(sql, read_only=True):
    con = duckdb.connect(active_db(), read_only=read_only)
    result = con.execute(sql).df()
    con.close()
    return result


def db_scalar(sql):
    con = duckdb.connect(active_db(), read_only=True)
    result = con.execute(sql).fetchone()[0]
    con.close()
    return result


def db_exec(sql):
    con = duckdb.connect(active_db())
    con.execute(sql)
    con.close()


def reset_dataset():
    try:
        con = duckdb.connect(DB_FILE)
        con.execute("DROP TABLE IF EXISTS udl_data")
        con.close()
    except Exception:
        pass
    try:
        if os.path.exists(DB_FILE_MOD):
            os.remove(DB_FILE_MOD)
    except Exception:
        pass
    st.session_state["loaded"] = False
    st.session_state["tinfo"] = None
    st.session_state["data_modified"] = False
    rerun()


# ── File helpers ──────────────────────────────────────────────────────────────
def materialize_modified(positive_value=None):
    """Write a transformed copy of the data to `module1_data_modified.duckdb`.

    Applies the UI type overrides (CATEGORICAL → VARCHAR, NUMERIC → DOUBLE) and
    converts the target column to integer 0/1 so the *actual* DuckDB types
    reflect the metadata the user set (no per-run overhead downstream).

    - positive_value=None  → binary target (known encoding) converted in place to 0/1
    - positive_value=val   → multi-class target binarized into `<col>__binary` (0/1)
    """
    if not os.path.exists(DB_FILE):
        st.error("Load a dataset first.")
        return
    con_raw = duckdb.connect(DB_FILE, read_only=True)
    try:
        raw = con_raw.execute("SELECT * FROM udl_data").df()
    finally:
        con_raw.close()
    if raw.empty:
        st.error("No data to materialize.")
        return

    overrides = st.session_state.get("type_overrides") or {}
    target_col = st.session_state.get("target_col")
    tinfo = st.session_state.get("tinfo") or {}

    select_parts = []
    new_target = target_col
    for col in raw.columns:
        if col == target_col and target_col:
            if positive_value is not None:
                pv = str(positive_value).replace("'", "''")
                select_parts.append(f'"{col}"')
                select_parts.append(
                    f"(CASE WHEN LOWER(TRIM(CAST(\"{col}\" AS VARCHAR)))='{pv.lower()}' "
                    f"THEN 1 ELSE 0 END)::INT AS \"{col}__binary\""
                )
                new_target = f"{col}__binary"
            elif tinfo.get("is_binary") and tinfo.get("binary_label"):
                select_parts.append(
                    f"(CASE WHEN TRY_CAST(\"{col}\" AS DOUBLE) IS NOT NULL "
                    f"THEN CAST(TRY_CAST(\"{col}\" AS DOUBLE) AS INT) "
                    f"WHEN LOWER(TRIM(CAST(\"{col}\" AS VARCHAR))) "
                    f"IN ('1','true','yes','y','t') THEN 1 ELSE 0 END)::INT AS \"{col}\""
                )
            else:
                select_parts.append(f'"{col}"')
        else:
            ov = overrides.get(str(col), "AUTO")
            if ov == "CATEGORICAL":
                select_parts.append(f'CAST("{col}" AS VARCHAR) AS "{col}"')
            elif ov == "NUMERIC":
                select_parts.append(f'TRY_CAST("{col}" AS DOUBLE) AS "{col}"')
            else:
                select_parts.append(f'"{col}"')

    select_sql = ", ".join(select_parts)
    con = duckdb.connect(DB_FILE_MOD)
    con.register("raw_df", raw)
    con.execute("DROP TABLE IF EXISTS udl_data")
    con.execute(f"CREATE TABLE udl_data AS SELECT {select_sql} FROM raw_df")
    con.close()

    if new_target != target_col:
        st.session_state["target_col"] = new_target
        st.session_state["tinfo"] = None
    st.session_state["data_modified"] = True
    st.success("☑️ Modified dataset written to `module1_data_modified.duckdb` and is now active.")
    rerun()


# ── File helpers ──────────────────────────────────────────────────────────────
def detect_encoding(path):
    with open(path, "rb") as fh:
        raw = fh.read(100_000)
    for enc in ("utf-8", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_format(name):
    ext = os.path.splitext(name)[1].lower()
    return {
        ".csv": "CSV", ".tsv": "CSV (tab-separated)",
        ".parquet": "Parquet", ".pq": "Parquet",
        ".arrow": "Arrow", ".feather": "Arrow / Feather",
        ".xlsx": "Excel", ".xls": "Excel (legacy)",
    }.get(ext, "Unknown")


def load_file_udl(path, encoding="Auto-detect"):
    ext = os.path.splitext(path)[1].lower()
    if encoding != "Auto-detect" and ext == ".csv":
        enc = {"UTF-8": "utf8", "Latin-1": "latin1"}.get(encoding, "utf8")
        table = pa_csv.read_csv(path, read_options=pa_csv.ReadOptions(encoding=enc))
        return UniversalDataLoader().load(fallback_data=table)
    try:
        return UniversalDataLoader(file_path=path).load()
    except Exception:
        if ext == ".csv":
            last = None
            for enc in ("utf-8", "latin-1"):
                try:
                    table = pa_csv.read_csv(path, read_options=pa_csv.ReadOptions(encoding=enc))
                    return UniversalDataLoader().load(fallback_data=table)
                except Exception as exc:
                    last = exc
            if last is not None:
                raise last
        raise


def load_and_persist(arrow_table, progress=None, dataset_name=None):
    if progress:
        progress.progress(0.2, text="Persisting to DuckDB…")
    # A fresh raw load invalidates any previously materialized modified copy.
    try:
        if os.path.exists(DB_FILE_MOD):
            os.remove(DB_FILE_MOD)
    except Exception:
        pass
    db_write(arrow_table)
    if progress:
        progress.progress(0.5, text="Profiling columns…")
    st.session_state["loaded"] = True
    st.session_state["tinfo"] = None
    st.session_state["data_modified"] = False
    if dataset_name:
        st.session_state["dataset_name"] = dataset_name
    if progress:
        progress.progress(1.0, text="Done")


def smart_default_hint():
    hints = []
    for base in (os.path.join(os.getcwd(), "data"), os.path.expanduser("~/Downloads")):
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            if detect_format(f) != "Unknown":
                hints.append(os.path.join(base, f))
    return hints[:3]


# ── Sample datasets ───────────────────────────────────────────────────────────
def find_sample_datasets():
    import rapidsegment
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(rapidsegment.__file__)))
    bases = [pkg_dir, os.path.join(os.getcwd(), "data"), os.path.expanduser("~/Downloads")]
    found = {}
    for base in bases:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f in SAMPLE_NAMES and f not in found:
                    p = os.path.join(root, f)
                    if os.path.getsize(p) > 0:
                        found[f] = p
    return found


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RapidSegment — Data Loader & Profiling", layout="wide")
apply_cyberpunk_theme()
st.title("RapidSegment — Module 1: Data Source & Profiling")

st.session_state["dataset_name"] = st.text_input(
    "Dataset name",
    value=st.session_state.get("dataset_name", ""),
    help="Used to group experiments for the same dataset in the Leaderboard.",
).strip()

for key, val in {
    "loaded": False, "tinfo": None,
    "bq_datasets": None, "bq_tables": None, "bq_preview": None,
    "type_overrides": {}, "target_col": None,
    "dataset_name": "", "data_modified": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar: source selection ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    source = st.radio("Select source", ["Local File", "BigQuery", "Sample Datasets"])
    st.divider()

    if source == "Local File":
        method = st.radio("Input method", ["File path", "Drag & drop upload"])

        if method == "File path":
            hints = smart_default_hint()
            if hints:
                st.caption("💡 Smart defaults — detected: " + ", ".join(os.path.basename(h) for h in hints))
            fp = st.text_input("File path", placeholder="/path/to/file.csv")
            encoding = st.selectbox("Encoding", ["Auto-detect", "UTF-8", "Latin-1"])
            st.caption("Supported: CSV · Parquet · Arrow/Feather · Excel")
            if st.button("Load File", type="primary", disabled=not fp):
                if not os.path.exists(fp):
                    st.error(f"File not found: {fp}")
                else:
                    with st.spinner("Reading file…"):
                        try:
                            progress = st.progress(0, text="Loading…")
                            data = load_file_udl(fp, encoding)
                            load_and_persist(data, progress, dataset_name=os.path.basename(fp))
                            st.success(f"Loaded: {os.path.basename(fp)}")
                        except Exception as exc:
                            st.error(str(exc))

        else:
            uploaded = st.file_uploader(
                "Drag & drop file here",
                type=["csv", "tsv", "parquet", "pq", "arrow", "feather", "xlsx", "xls"],
            )
            if uploaded is not None:
                size_mb = uploaded.size / 1e6
                if size_mb > MAX_UPLOAD_MB:
                    st.error(f"File too large: {size_mb:.1f} MB (limit {MAX_UPLOAD_MB} MB)")
                else:
                    encoding = st.selectbox("Encoding", ["Auto-detect", "UTF-8", "Latin-1"], key="up_enc")
                    st.caption(f"Detected format: **{detect_format(uploaded.name)}** · {size_mb:.1f} MB")
                    if st.button("Load Uploaded File", type="primary"):
                        with st.spinner(f"Loading '{uploaded.name}'…"):
                            ext = os.path.splitext(uploaded.name)[1].lower()
                            tmp_path = None
                            try:
                                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                                    shutil.copyfileobj(uploaded, tmp)
                                    tmp_path = tmp.name
                                progress = st.progress(0, text="Loading…")
                                data = load_file_udl(tmp_path, encoding)
                                load_and_persist(data, progress, dataset_name=uploaded.name)
                                st.success(f"Loaded '{uploaded.name}'")
                            except Exception as exc:
                                st.error(str(exc))
                            finally:
                                if tmp_path:
                                    os.unlink(tmp_path)

    elif source == "BigQuery":
        st.caption(
            "Authentication uses your environment credentials "
            "(`gcloud auth application-default login` or GOOGLE_APPLICATION_CREDENTIALS) "
            "— no secrets are stored in the app."
        )
        table_ref = st.text_input(
            "BigQuery table",
            placeholder="project_id.dataset_id.table_id",
            help="Full table reference. 3 parts = project.dataset.table; "
                 "2 parts = dataset.table (uses your default GCP project).",
        ).strip()

        if st.button("Load table", type="primary", disabled=not table_ref):
            parts = [p.strip() for p in table_ref.split(".") if p.strip()]
            if len(parts) not in (2, 3):
                st.error(
                    "Enter the table as 'project_id.dataset_id.table_id' "
                    "(or 'dataset_id.table_id')."
                )
            else:
                pid = parts[0] if len(parts) == 3 else None
                did = parts[-2]
                tid = parts[-1]
                with st.spinner("Loading from BigQuery…"):
                    try:
                        progress = st.progress(0, text="Loading from BigQuery…")
                        data = UniversalDataLoader(
                            project_id=pid, dataset_id=did, table_id=tid
                        ).load()
                        load_and_persist(data, progress, dataset_name=f"{did}.{tid}")
                        st.success(f"Loaded {pid + '.' if pid else ''}{did}.{tid}")
                    except ImportError:
                        st.error(
                            "google-cloud-bigquery not installed. Run: "
                            "pip install google-cloud-bigquery"
                        )
                    except Exception as exc:
                        st.error(
                            f"BigQuery load failed: {exc}\n\n"
                            "Tip: authenticate with `gcloud auth application-default login` "
                            "or set GOOGLE_APPLICATION_CREDENTIALS."
                        )

        with st.expander("Browse BigQuery (optional discovery)"):
            st.caption("Lists datasets/tables using your environment credentials.")
            bpid = st.text_input("GCP Project ID (for browsing)", key="bq_browse_pid")
            if st.button("List datasets", disabled=not bpid):
                try:
                    from google.cloud import bigquery

                    client = bigquery.Client(project=bpid)
                    st.session_state["bq_datasets"] = sorted(
                        d.dataset_id for d in client.list_datasets()
                    )
                except ImportError:
                    st.error("google-cloud-bigquery not installed: pip install google-cloud-bigquery")
                except Exception as exc:
                    st.error(f"Browse failed: {exc}")
            datasets = st.session_state.get("bq_datasets") or []
            if datasets:
                did = st.selectbox("Dataset", datasets, key="bq_browse_did")
                if st.button("List tables"):
                    try:
                        from google.cloud import bigquery

                        client = bigquery.Client(project=bpid)
                        st.session_state["bq_tables"] = sorted(
                            t.table_id for t in client.list_tables(f"{bpid}.{did}")
                        )
                    except Exception as exc:
                        st.error(f"Browse failed: {exc}")
                tables = st.session_state.get("bq_tables") or []
                if tables:
                    tid = st.selectbox("Table", tables, key="bq_browse_tid")
                    if st.button("Preview (first 1000 rows)"):
                        try:
                            from google.cloud import bigquery

                            client = bigquery.Client(project=bpid)
                            st.session_state["bq_preview"] = client.query(
                                f"SELECT * FROM `{bpid}.{did}.{tid}` LIMIT 1000"
                            ).to_dataframe()
                        except Exception as exc:
                            st.error(f"Preview failed: {exc}")

        preview = st.session_state.get("bq_preview")
        if preview is not None:
            with st.expander("BigQuery streaming preview (first 1000 rows)"):
                st.dataframe(preview, height=300, width='stretch')

    else:  # Sample Datasets
        samples = find_sample_datasets()
        if samples:
            name = st.selectbox("Quick-start dataset", list(samples))
            fp = samples[name]
            st.caption(f"Found at: `{fp}`")
            if st.button(f"Load {name}", type="primary"):
                with st.spinner("Loading sample…"):
                    try:
                        progress = st.progress(0, text="Loading…")
                        data = UniversalDataLoader(file_path=fp).load()
                        load_and_persist(data, progress, dataset_name=name)
                        st.success(f"Loaded sample: {name}")
                    except Exception as exc:
                        st.error(str(exc))
        else:
            st.warning("No sample datasets found. Drop `bank-full.csv` / `train.csv` into `./data/`.")
            manual = st.text_input("Or provide a path manually", placeholder="./data/bank-full.csv")
            if manual and st.button("Load Sample", type="primary", disabled=not os.path.exists(manual)):
                with st.spinner("Loading…"):
                    try:
                        progress = st.progress(0, text="Loading…")
                        data = UniversalDataLoader(file_path=manual).load()
                        load_and_persist(data, progress, dataset_name=os.path.basename(manual))
                        st.success("Loaded")
                    except Exception as exc:
                        st.error(str(exc))

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state["loaded"]:
    st.info("Load a dataset from the sidebar to get started.")
    st.stop()

tab_preview, tab_quality, tab_meta, tab_target = st.tabs(
    ["Preview", "Quality Report", "Column Metadata", "Target Selection"]
)

# ── Tab 1: Preview ────────────────────────────────────────────────────────────
with tab_preview:
    st.subheader("Preview — first 100 rows")
    df = db_query("SELECT * FROM udl_data LIMIT 100")
    st.caption(f"DuckDB: `{DB_FILE}`")
    st.dataframe(df, height=360, width='stretch')


# ── Shared profiling helpers ──────────────────────────────────────────────────
def effective_types(summ):
    overrides = st.session_state.get("type_overrides") or {}
    out = {}
    for _, row in summ.iterrows():
        inferred = "NUMERIC" if is_num(row.get("column_type", "")) else "CATEGORICAL"
        ov = overrides.get(str(row["column_name"]), "AUTO")
        out[str(row["column_name"])] = inferred if ov == "AUTO" else ov
    return out


def build_metadata(summ):
    eff = effective_types(summ)
    stamp = f"{os.path.getmtime(DB_FILE)}:{db_scalar('SELECT COUNT(*) FROM udl_data')}"

    @st.cache_data(ttl=600, show_spinner=False)
    def top5(col, _stamp):
        return db_query(
            f'SELECT CAST("{col}" AS VARCHAR) AS value, COUNT(*) AS cnt '
            f'FROM udl_data GROUP BY 1 ORDER BY 2 DESC LIMIT 5'
        )

    rows = []
    n = len(summ)
    prog = st.progress(0, text="Profiling columns…")
    for i, (_, row) in enumerate(summ.iterrows()):
        col = str(row["column_name"])
        etype = eff[col]
        inferred = "NUMERIC" if is_num(row.get("column_type", "")) else "CATEGORICAL"
        null_pct = float(row.get("null_percentage") or 0)
        unique = int(row.get("approx_unique") or 0)

        if etype == "NUMERIC":
            avg = row.get("avg")
            dist = f"min={row.get('min', '—')}  max={row.get('max', '—')}"
            if avg is not None:
                dist += f"  mean={float(avg):.4g}"
        else:
            t5 = top5(col, stamp)
            dist = ", ".join(f"{str(v)}:{c:,}" for v, c in zip(t5["value"], t5["cnt"])) or "—"

        warn = "✓"
        try:
            if etype == "CATEGORICAL":
                float(str(row.get("min", "")))
                float(str(row.get("max", "")))
                warn = "⚠ looks numeric but has text"
        except (ValueError, TypeError):
            pass
        if etype == "NUMERIC" and unique <= 5:
            warn = "⚠ low cardinality"
        if null_pct > 20:
            warn = "⚠ high nulls (>20%)"

        rows.append({
            "Column": col,
            "Type": inferred if etype == inferred else f"{inferred} → {etype}",
            "Cardinality": unique,
            "Null %": f"{null_pct:.1f}%",
            "Distribution": dist,
            "Warning": warn,
        })
        prog.progress((i + 1) / n)
    prog.empty()
    return pd.DataFrame(rows)


# ── Tab 2: Quality Report ─────────────────────────────────────────────────────
with tab_quality:
    st.subheader("Data Quality Report")

    n_rows = db_scalar("SELECT COUNT(*) FROM udl_data")
    types = db_query(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name='udl_data' ORDER BY ordinal_position"
    )
    summ = db_query("SUMMARIZE udl_data")
    eff = effective_types(summ)
    db_mb = os.path.getsize(DB_FILE) / 1024 / 1024
    n_num = sum(1 for t in eff.values() if t == "NUMERIC")
    n_cat = len(eff) - n_num

    with st.spinner("Computing memory footprint…"):
        full = db_query("SELECT * FROM udl_data")
    mem_mb = full.memory_usage(deep=True).sum() / 1e6

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", f"{len(eff)}")
    c3.metric("Numeric / Categorical", f"{n_num} / {n_cat}")
    c4.metric("Memory footprint", f"{mem_mb:.1f} MB")

    st.divider()

    null_df = (summ[["column_name", "null_percentage"]]
               .assign(null_pct=summ["null_percentage"].astype(float))
               .query("null_pct > 0")
               .sort_values("null_pct", ascending=False)
               .rename(columns={"column_name": "Column", "null_pct": "Null %"}))
    if null_df.empty:
        st.success("☑️ No missing values — data ready for segmentation ✓")
    else:
        st.warning(f"⚠️ {len(null_df)} column(s) have missing values (worst offenders first)")
        null_df["Null %"] = null_df["Null %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(null_df[["Column", "Null %"]].reset_index(drop=True), width='stretch')

    warns = build_metadata(summ)
    warns = warns[warns["Warning"] != "✓"]
    if warns.empty:
        st.success("☑️ No type warnings — data ready for segmentation ✓")
    else:
        st.warning(f"⚠️ {len(warns)} column(s) flagged with type/cardinality warnings")

    st.caption(f"DuckDB file size: {db_mb:.1f} MB")


# ── Tab 3: Column Metadata ────────────────────────────────────────────────────
with tab_meta:
    st.subheader("Column Metadata")
    summ = db_query("SUMMARIZE udl_data")

    with st.expander("Manual type overrides"):
        base = pd.DataFrame({
            "Column": summ["column_name"],
            "Override": [st.session_state["type_overrides"].get(c, "AUTO") for c in summ["column_name"]],
        })
        try:
            edited = st.data_editor(
                base,
                column_config={
                    "Column": st.column_config.TextColumn(disabled=True),
                    "Override": st.column_config.SelectboxColumn(options=["AUTO", "NUMERIC", "CATEGORICAL"]),
                },
                hide_index=True, width='stretch',
            )
        except AttributeError:
            edited = st.data_editor(base, hide_index=True, width='stretch')
        if st.button("Apply type overrides"):
            st.session_state["type_overrides"] = dict(zip(edited["Column"], edited["Override"]))
            st.success("Overrides stored — click 'Apply metadata' below to materialize them.")

    st.divider()
    st.subheader("Materialize modified dataset")
    st.caption(
        "Write a transformed copy (`module1_data_modified.duckdb`) with the type "
        "overrides applied and the target converted to 1/0. Downstream modules "
        "(Workbench, Execution, Results) read this copy automatically."
    )
    if st.button("Apply metadata & create modified dataset", type="primary"):
        materialize_modified()
    if os.path.exists(DB_FILE_MOD):
        st.success("☑️ Modified dataset is active.")
        if st.button("Discard modified dataset (revert to raw)"):
            try:
                os.remove(DB_FILE_MOD)
            except Exception:
                pass
            st.session_state["data_modified"] = False
            rerun()

    meta = build_metadata(summ)
    st.dataframe(meta, height=420, width='stretch', hide_index=True)


# ── Tab 4: Target Selection ───────────────────────────────────────────────────
with tab_target:
    st.subheader("Target Column Selection & Validation")

    summ = db_query("SUMMARIZE udl_data")
    col_names = summ["column_name"].tolist()

    only_binary = st.checkbox("Only show likely binary columns (≤ 2 unique values)", value=True)

    def label(c):
        u = int(summ.loc[summ["column_name"] == c, "approx_unique"].iloc[0] or 99)
        return f"{c}  ★" if u <= 2 else c

    display_names = [
        label(c) for c in col_names
        if (not only_binary or int(summ.loc[summ["column_name"] == c, "approx_unique"].iloc[0] or 99) <= 2)
    ]
    name_map = {label(c): c for c in col_names}

    if not display_names:
        st.warning("No binary candidates found — uncheck the filter to see all columns.")
        display_names = [label(c) for c in col_names]

    preselect = st.session_state.get("target_col")
    default_idx = 0
    if preselect in col_names:
        plabel = label(preselect)
        if plabel in display_names:
            default_idx = display_names.index(plabel)

    sel_display = st.selectbox("Target column  (★ = ≤ 2 unique values — likely binary)", display_names, index=default_idx)
    sel_col = name_map[sel_display]

    if st.button("Validate Target", type="primary"):
        with st.spinner("Validating…"):
            n_total = db_scalar("SELECT COUNT(*) FROM udl_data")
            n_notnull = db_scalar(f'SELECT COUNT("{sel_col}") FROM udl_data')
            n_distinct = db_scalar(f'SELECT COUNT(DISTINCT "{sel_col}") FROM udl_data')
            dist = db_query(
                f'SELECT CAST("{sel_col}" AS VARCHAR) AS val, COUNT(*) AS cnt '
                f'FROM udl_data GROUP BY "{sel_col}" ORDER BY cnt DESC LIMIT 10'
            )
            vals_nonnull = [v for v in dist["val"].tolist() if v is not None]
            BMAPS = [
                ({"0", "1"}, "0 / 1"),
                ({"true", "false"}, "True / False"),
                ({"yes", "no"}, "Yes / No"),
                ({"y", "n"}, "Y / N"),
            ]
            blabel = next((l for s, l in BMAPS if {str(v).lower() for v in vals_nonnull} == s), None)
            er = None
            if blabel:
                pos = dist.loc[dist["val"].astype(str).str.lower().isin({"1", "true", "yes", "y"}), "cnt"].sum()
                er = int(pos) / n_total if n_total else 0
            st.session_state["tinfo"] = dict(
                col=sel_col, n_distinct=n_distinct, is_binary=blabel is not None,
                binary_label=blabel, event_rate=er,
                null_count=n_total - n_notnull,
                null_pct=(n_total - n_notnull) / n_total * 100 if n_total else 0,
                dist_df=dist, n_total=n_total,
            )
            st.session_state["target_col"] = sel_col

    info = st.session_state.get("tinfo")
    if info:
        st.divider()
        if info["is_binary"]:
            st.success(f"☑️ Binary column — encoding: **{info['binary_label']}**")
            er = info["event_rate"]
            if er is not None:
                bad = er < 0.01 or er > 0.99
                col_a, col_b = st.columns(2)
                col_a.metric("Event Rate", f"{er:.2%}")
                if bad:
                    col_b.warning("⚠️ Severe class imbalance — outside 1%–99%")
                else:
                    col_b.success("☑️ Event rate in healthy range")
        else:
            st.warning(f"⚠️ Multi-class — {info['n_distinct']} distinct values. Use the binarization helper below.")

        with st.expander("Binarization helper"):
            st.caption(
                "Multi-class target? Pick the positive value to create a 0/1 column "
                "(also applies type overrides and writes the modified dataset)."
            )
            dist = info["dist_df"]
            options = [str(v) for v in dist["val"].tolist() if v is not None][:10]
            pos = st.selectbox("Positive (event) value", options)
            if st.button("Binarize into 0/1 column"):
                materialize_modified(positive_value=pos)

        if info["null_pct"] > 0:
            st.error(f"⚠️ Target column has {info['null_pct']:.1f}% nulls ({info['null_count']:,} rows)")

        dist = info["dist_df"]
        colors = ["#6366f1", "#f59e0b", "#22c55e", "#ef4444", "#3b82f6",
                  "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4"]
        fig = go.Figure(go.Bar(
            x=dist["val"].astype(str), y=dist["cnt"],
            marker_color=colors[:len(dist)],
            text=dist["cnt"].apply(lambda n: f"{n:,}"),
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Class distribution — {info['col']}",
            xaxis_title="Value", yaxis_title="Count",
            height=350, margin=dict(l=40, r=20, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width='stretch')


# ── Actions ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Actions")

info = st.session_state.get("tinfo")
ready = bool(info and info["is_binary"])

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    if st.button("Proceed to Workbench", type="primary", disabled=not ready, width='stretch'):
        st.session_state["workbench_ready"] = True
        st.switch_page("pages/2_Workbench.py")
    if not ready:
        st.caption("Enabled after a binary target is validated.")
with c2:
    if st.button("Upload Different File", width='stretch'):
        reset_dataset()
with c3:
    summ = db_query("SUMMARIZE udl_data")
    report_df = build_metadata(summ)
    st.download_button(
        "Download Profiling Report (CSV)",
        report_df.to_csv(index=False).encode("utf-8"),
        file_name="profiling_report.csv", mime="text/csv", width='stretch',
    )
    report_json = json.dumps(
        {"columns": report_df.to_dict(orient="records"),
         "rows": int(db_scalar("SELECT COUNT(*) FROM udl_data")),
         "target": info["col"] if info else None},
        indent=2,
    )
    st.download_button(
        "Download Profiling Report (JSON)",
        report_json.encode("utf-8"),
        file_name="profiling_report.json", mime="application/json", width='stretch',
    )