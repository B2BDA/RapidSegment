"""
RapidSegment — Universal Data Loader
Run with:  streamlit run udl_app.py
"""
import os, tempfile
import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
from rapidsegment.utils.data_loader import UniversalDataLoader

DB_FILE = os.path.join(tempfile.gettempdir(), "udl_rapidsegment.duckdb")
NUMERIC = {"INTEGER","BIGINT","DOUBLE","FLOAT","DECIMAL","REAL",
           "SMALLINT","TINYINT","HUGEINT","INT","UBIGINT","UINTEGER"}

def is_num(t): return any(k in str(t).upper() for k in NUMERIC)

# ── DuckDB helpers (short-lived connections) ───────────────────────────────────
def db_write(arrow_table):
    con = duckdb.connect(DB_FILE)
    con.execute("DROP TABLE IF EXISTS udl_data")
    con.execute("CREATE TABLE udl_data AS SELECT * FROM arrow_table")
    con.close()

def db_query(sql, read_only=True):
    con = duckdb.connect(DB_FILE, read_only=read_only)
    result = con.execute(sql).df()
    con.close()
    return result

def db_scalar(sql):
    con = duckdb.connect(DB_FILE, read_only=True)
    result = con.execute(sql).fetchone()[0]
    con.close()
    return result

# ── Load via UDL ───────────────────────────────────────────────────────────────
def load_and_persist(arrow_table):
    db_write(arrow_table)
    st.session_state["loaded"] = True
    st.session_state["tinfo"]  = None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RapidSegment — UDL", layout="wide")
st.title("RapidSegment — Universal Data Loader")

if "loaded" not in st.session_state:
    st.session_state["loaded"] = False
if "tinfo" not in st.session_state:
    st.session_state["tinfo"] = None

# ── Sidebar: source selection ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    source = st.radio("Select source", ["Local File", "BigQuery", "Upload / Fallback"])
    st.divider()

    if source == "Local File":
        fp = st.text_input("File path", placeholder="/path/to/file.csv")
        st.caption("Supported: CSV · Parquet · Arrow / Feather · Excel")
        if st.button("Load File", type="primary", disabled=not fp):
            with st.spinner("Reading file…"):
                try:
                    data = UniversalDataLoader(file_path=fp).load()
                    load_and_persist(data)
                    st.success("Loaded from file")
                except Exception as e:
                    st.error(str(e))

    elif source == "BigQuery":
        pid = st.text_input("GCP Project ID")
        did = st.text_input("Dataset ID")
        tid = st.text_input("Table ID")
        st.caption("Requires `google-cloud-bigquery` + GCP credentials")
        if st.button("Load from BigQuery", type="primary", disabled=not (pid and did and tid)):
            with st.spinner("Connecting to BigQuery…"):
                try:
                    data = UniversalDataLoader(project_id=pid, dataset_id=did, table_id=tid).load()
                    load_and_persist(data)
                    st.success("Loaded from BigQuery")
                except Exception as e:
                    st.error(str(e))

    elif source == "Upload / Fallback":
        uploaded = st.file_uploader("Drop file here", type=["csv","parquet","arrow","feather","xlsx","xls"])
        st.caption("Passed as `fallback_data` to UDL (highest priority)")
        if uploaded and st.button("Load Uploaded File", type="primary"):
            with st.spinner(f"Loading '{uploaded.name}'…"):
                try:
                    ext = os.path.splitext(uploaded.name)[-1]
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    raw  = UniversalDataLoader(file_path=tmp_path).load()
                    data = UniversalDataLoader().load(fallback_data=raw)
                    os.unlink(tmp_path)
                    load_and_persist(data)
                    st.success(f"Loaded '{uploaded.name}' via fallback")
                except Exception as e:
                    st.error(str(e))

# ── Main area: tabs ────────────────────────────────────────────────────────────
if not st.session_state["loaded"]:
    st.info("Load a dataset from the sidebar to get started.")
    st.stop()

tab_preview, tab_quality, tab_meta, tab_target = st.tabs(
    ["Preview", "Quality Report", "Column Metadata", "Target Selection"]
)

# ── Tab 1: Preview ─────────────────────────────────────────────────────────────
with tab_preview:
    st.subheader("Preview — first 20 rows")
    df = db_query("SELECT * FROM udl_data LIMIT 20")
    st.caption(f"DuckDB: `{DB_FILE}`")
    st.dataframe(df, use_container_width=True)

# ── Tab 2: Quality Report ──────────────────────────────────────────────────────
with tab_quality:
    st.subheader("Data Quality Report")

    n_rows = db_scalar("SELECT COUNT(*) FROM udl_data")
    types  = db_query(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name='udl_data' ORDER BY ordinal_position"
    )
    summ   = db_query("SUMMARIZE udl_data")
    db_mb  = os.path.getsize(DB_FILE) / 1024 / 1024
    n_num  = int(types["data_type"].apply(is_num).sum())
    n_cat  = len(types) - n_num

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", f"{len(types)}")
    c3.metric("Numeric / Categorical", f"{n_num} / {n_cat}")
    c4.metric("DuckDB file size", f"{db_mb:.1f} MB")

    st.divider()

    null_df = (summ[["column_name","null_percentage"]]
               .assign(null_pct=summ["null_percentage"].astype(float))
               .query("null_pct > 0")
               .sort_values("null_pct", ascending=False)
               .rename(columns={"column_name":"Column","null_pct":"Null %"}))

    if null_df.empty:
        st.success("✅ No missing values — data ready for segmentation")
    else:
        st.warning(f"⚠️ {len(null_df)} column(s) have missing values")
        null_df["Null %"] = null_df["Null %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(null_df[["Column","Null %"]].reset_index(drop=True), use_container_width=True)

# ── Tab 3: Column Metadata ─────────────────────────────────────────────────────
with tab_meta:
    st.subheader("Column Metadata")
    summ = db_query("SUMMARIZE udl_data")

    rows = []
    for _, row in summ.iterrows():
        ct  = str(row.get("column_type",""))
        avg = row.get("avg")
        if is_num(ct):
            dist = f"min={row.get('min','—')}  max={row.get('max','—')}  mean={float(avg):.4g}" if avg is not None else "—"
        else:
            dist = f"'{row.get('min','—')}' … '{row.get('max','—')}'"

        warn = "✓"
        try:
            if "VARCHAR" in ct.upper():
                float(str(row.get("min",""))); float(str(row.get("max","")))
                warn = "⚠ looks numeric"
        except (ValueError, TypeError):
            pass
        if is_num(ct) and int(row.get("approx_unique") or 99) <= 5:
            warn = "⚠ low cardinality"

        rows.append({
            "Column":       row["column_name"],
            "Type":         ct,
            "Cardinality":  int(row.get("approx_unique") or 0),
            "Null %":       f"{float(row.get('null_percentage') or 0):.1f}%",
            "Distribution": dist,
            "Warning":      warn,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Tab 4: Target Selection ────────────────────────────────────────────────────
with tab_target:
    st.subheader("Target Column Selection & Validation")

    summ     = db_query("SUMMARIZE udl_data")
    col_names = summ["column_name"].tolist()

    # Pre-flag binary candidates
    def label(c):
        u = int(summ.loc[summ["column_name"]==c, "approx_unique"].iloc[0] or 99)
        return f"{c}  ★" if u <= 2 else c

    display_names = [label(c) for c in col_names]
    name_map      = {label(c): c for c in col_names}

    sel_display = st.selectbox("Target column  (★ = ≤ 2 unique values — likely binary)", display_names)
    sel_col     = name_map[sel_display]

    if st.button("Validate Target", type="primary"):
        with st.spinner("Validating…"):
            n_total   = db_scalar("SELECT COUNT(*) FROM udl_data")
            n_notnull = db_scalar(f'SELECT COUNT("{sel_col}") FROM udl_data')
            dist      = db_query(
                f'SELECT CAST("{sel_col}" AS VARCHAR) AS val, COUNT(*) AS cnt '
                f'FROM udl_data GROUP BY "{sel_col}" ORDER BY cnt DESC LIMIT 10'
            )
            vals   = dist["val"].tolist()
            BMAPS  = [({"0","1"},"0 / 1"),({"true","false"},"True / False"),
                      ({"yes","no"},"Yes / No"),({"y","n"},"Y / N")]
            blabel = next((l for s,l in BMAPS if {v.lower() for v in vals}==s), None)
            er     = None
            if blabel:
                pos = dist.loc[dist["val"].str.lower().isin({"1","true","yes","y"}),"cnt"].sum()
                er  = int(pos) / n_total if n_total else 0
            st.session_state["tinfo"] = dict(
                col=sel_col, n_distinct=len(vals), is_binary=blabel is not None,
                binary_label=blabel, event_rate=er,
                null_count=n_total-n_notnull,
                null_pct=(n_total-n_notnull)/n_total*100 if n_total else 0,
                dist_df=dist, n_total=n_total
            )

    info = st.session_state["tinfo"]
    if info:
        st.divider()

        # Binary / multi-class verdict
        if info["is_binary"]:
            st.success(f"✅ Binary column — encoding: **{info['binary_label']}**")
            er = info["event_rate"]
            if er is not None:
                bad = er < 0.01 or er > 0.99
                col_a, col_b = st.columns(2)
                col_a.metric("Event Rate", f"{er:.2%}")
                if bad:
                    col_b.warning("⚠️ Severe class imbalance — outside 1%–99%")
                else:
                    col_b.success("✅ Event rate in healthy range")
        else:
            st.warning(f"⚠️ Multi-class — {info['n_distinct']} distinct values. Consider binarizing before segmentation.")

        if info["null_pct"] > 0:
            st.error(f"⚠️ Target column has {info['null_pct']:.1f}% nulls ({info['null_count']:,} rows)")

        # Distribution chart
        dist   = info["dist_df"]
        colors = ["#6366f1","#f59e0b","#22c55e","#ef4444","#3b82f6",
                  "#ec4899","#14b8a6","#f97316","#8b5cf6","#06b6d4"]
        fig = go.Figure(go.Bar(
            x=dist["val"].astype(str), y=dist["cnt"],
            marker_color=colors[:len(dist)],
            text=dist["cnt"].apply(lambda n: f"{n:,}"),
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Class distribution — {info['col']}",
            xaxis_title="Value", yaxis_title="Count",
            height=350, margin=dict(l=40,r=20,t=50,b=40),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)