"""
Module 1: Enhanced Data Source & Profiling
==========================================
Implements Module 1 of the RapidSegment UI_MVP.md spec.

Sections:
    1.1 Source selection      - Local File | BigQuery | Sample Datasets (radio buttons)
    1.2 File upload/parsing   - drag-and-drop, format + encoding auto-detection
    1.3 BigQuery integration  - project.dataset.table path with parsing helpers
    1.4 Data preview & profiling - first 100 rows + column metadata panel
        (type, cardinality, null %, distribution, type warnings)
    1.5 Target validation     - binary/multi-class detection, event rate,
        class imbalance warning, class distribution chart, binarization helper
    1.6 Data quality report   - quality score, missing summary, memory footprint,
        recommendation
    1.7 Actions               - Proceed to Workbench, reset, download report

Backend: DuckDBProfiler (DuckDB-native profiling, no pandas for aggregation).
UI:     Solara components.
"""
import base64
import html as _html
import json
from datetime import datetime
from pathlib import Path

import solara
import solara.lab

from .data_profiler_duckdb import DuckDBProfiler

# ---------------------------------------------------------------------------
# Module-level reactive state (shared across the Solara app)
# ---------------------------------------------------------------------------
source_mode = solara.reactive("local")          # "local" | "bigquery" | "sample"
local_path = solara.reactive("")
local_encoding = solara.reactive("auto")
sample_key = solara.reactive("")
bq_full_path = solara.reactive("")
bq_project = solara.reactive("")
bq_dataset = solara.reactive("")
bq_table = solara.reactive("")
data_table_name = solara.reactive("main_data")
primary_key = solara.reactive("")
target_col = solara.reactive("")
binarize_event = solara.reactive("")

profile_report = solara.reactive(None)
preview_df = solara.reactive(None)
columns_meta_df = solara.reactive(None)
quality_report = solara.reactive(None)
target_validation = solara.reactive(None)
binarized_target = solara.reactive(None)
load_status = solara.reactive("")
load_trigger = solara.reactive(0)

SOURCE_OPTIONS = [
    ("local", "Local File"),
    ("bigquery", "BigQuery"),
    ("sample", "Sample Datasets"),
]

ENCODING_OPTIONS = ["auto", "utf-8", "utf-8-sig", "latin-1"]


def get_sample_options() -> dict:
    """Sample datasets bundled in the repo ./Datasets dir."""
    return DuckDBProfiler.list_sample_datasets()


def _parse_bq_path(s: str):
    p = s.strip().replace("bq://", "").replace("bigquery://", "")
    parts = p.split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    return None, None, None


def _report_csv_content(report: dict) -> str:
    df = DuckDBProfiler.profiling_columns_df(report)
    return df.to_csv(index=False)


def _report_json_content(report: dict) -> str:
    return json.dumps(report, indent=2, default=str)


def _data_uri(content: str, mime: str) -> str:
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Small reusable UI primitives
# ---------------------------------------------------------------------------
@solara.component
def MiniBars(data, height: int = 16):
    """Lightweight horizontal bar chart built from HTML/CSS (no extra deps).
    data: list of (label, value, color)."""
    if not data:
        return
    maxv = max((v for _, v, _ in data if v is not None), default=1) or 1
    rows = []
    for label, value, color in data:
        value = value or 0
        pct = value / maxv * 100
        safe_label = _html.escape(str(label))
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<span style="width:150px;font-size:11px;color:#555;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">{safe_label}</span>'
            f'<div style="flex:1;background:#ececec;border-radius:4px;height:{height}px;">'
            f'<div style="width:{pct:.1f}%;background:{color};height:{height}px;border-radius:4px;"></div>'
            f'</div>'
            f'<span style="width:80px;font-size:11px;text-align:right;color:#222;">{value:,.0f}</span>'
            f'</div>'
        )
    solara.HTML(unsafe_innerHTML="".join(rows))


@solara.component
def StatCards(items):
    """items: list of (label, value)."""
    cards = []
    for label, value in items:
        cards.append(
            f'<div style="flex:1;min-width:110px;border:1px solid #ddd;border-radius:8px;'
            f'padding:10px 12px;text-align:center;background:#fafafa;">'
            f'<div style="font-size:20px;font-weight:600;color:#111;">{_html.escape(str(value))}</div>'
            f'<div style="font-size:11px;color:#666;margin-top:2px;">{_html.escape(str(label))}</div>'
            f'</div>'
        )
    solara.HTML(unsafe_innerHTML=f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{"".join(cards)}</div>')


@solara.component
def RecommendationBanner(quality):
    if not quality:
        return
    colors = {"ready": "#1b5e20", "warning": "#e65100", "attention": "#b71c1c"}
    bg = {"ready": "#e8f5e9", "warning": "#fff3e0", "attention": "#fdecea"}
    icon = {"ready": "OK", "warning": "!", "attention": "!"}
    color = colors.get(quality.get("status"), "#333")
    bgc = bg.get(quality.get("status"), "#eee")
    solara.HTML(
        unsafe_innerHTML=(
            f'<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:8px;'
            f'background:{bgc};border-left:4px solid {color};margin:8px 0;">'
            f'<span style="font-weight:700;color:{color};font-size:14px;">[{icon}]</span>'
            f'<span style="font-size:13px;color:#222;">Data quality score: '
            f'<b style="color:{color};">{quality.get("quality_score", "?")}/100</b>'
            f' &mdash; {_html.escape(quality.get("recommendation", ""))}</span>'
            f'</div>'
        )
    )


@solara.component
def SourceSelector():
    """1.1 Source selection rendered as radio-style buttons."""
    with solara.Row(style={"gap": "6px"}):
        for mode, label in SOURCE_OPTIONS:
            selected = source_mode.value == mode
            solara.Button(
                label=label,
                on_click=lambda m=mode: _set_source_mode(m),
                style=(
                    {"background": "#111111", "color": "#ffffff", "font-weight": "600"}
                    if selected
                    else {"color": "#333333"}
                ),
            )


def _set_source_mode(mode: str):
    source_mode.value = mode


# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
def _do_load(profiler: DuckDBProfiler):
    """Load the selected source into DuckDB and refresh all profiling state."""
    if load_trigger.value == 0:
        return
    mode = source_mode.value
    tbl = data_table_name.value
    profiler = profiler or DuckDBProfiler(db_path=".rapidsegment_suite/profiling.db")

    if mode == "local":
        fp = local_path.value.strip()
        if not fp:
            load_status.value = "Enter a local file path or drop a file below."
            return
        if not Path(fp).exists():
            load_status.value = f"File not found: {fp}"
            return
        fmt = DuckDBProfiler.detect_format(fp)
        enc = None if local_encoding.value == "auto" else local_encoding.value
        profiler.load_table(file_path=fp, table_name=tbl, encoding=enc, format=fmt)
    elif mode == "sample":
        key = sample_key.value
        if not key:
            load_status.value = "Pick a sample dataset first."
            return
        profiler.load_sample(key, table_name=tbl)
    elif mode == "bigquery":
        if bq_full_path.value:
            proj, ds, tb = _parse_bq_path(bq_full_path.value)
            if proj:
                bq_project.value, bq_dataset.value, bq_table.value = proj, ds, tb
            else:
                bq_dataset.value, bq_table.value = ds, tb
        if not bq_dataset.value or not bq_table.value:
            load_status.value = "Enter a BigQuery dataset.table path."
            return
        profiler.load_from_bq(
            bq_path=bq_full_path.value or None,
            project_id=bq_project.value or None,
            dataset_id=bq_dataset.value,
            table_id=bq_table.value,
            table_name=tbl,
        )
    else:
        load_status.value = f"Unknown source mode: {mode}"
        return

    # Refresh all derived state
    rep = profiler.profile(tbl)
    profile_report.value = rep
    preview_df.value = profiler.get_preview(tbl, n=100)
    columns_meta_df.value = DuckDBProfiler.profiling_columns_df(rep)
    quality_report.value = profiler.data_quality_report(tbl, rep)
    target_col.value = ""
    target_validation.value = None
    binarized_target.value = None
    binarize_event.value = ""
    load_status.value = (
        f"Loaded {rep['total_rows']:,} rows x {rep['total_columns']} cols via "
        f"rapidsegment UniversalDataLoader | {rep['estimated_data_size_mb']} MB in DuckDB"
    )


@solara.component
def LoadSection(profiler):
    """1.1-1.3 source config + load button + progress indicator."""
    task = solara.lab.use_task(
        lambda: _do_load(profiler),
        dependencies=[load_trigger.value],
        raise_error=False,
    )
    with solara.Column(style={"gap": "10px"}):
        SourceSelector()

        if source_mode.value == "local":
            with solara.Row():
                solara.InputText(
                    label="Local file path (CSV / Parquet / Excel / JSON / Arrow)",
                    value=local_path,
                    placeholder="/home/user/data.csv",
                    style="flex: 1;",
                )
                solara.Select(
                    label="Encoding",
                    values=ENCODING_OPTIONS,
                    value=local_encoding,
                    style="width: 160px;",
                )
            solara.FileDrop(
                label="Or drag & drop a file here (uploaded to ./uploaded_data)",
                on_file=_on_file_dropped,
                lazy=False,
            )
            with solara.Row(style={"gap": "4px", "align-items": "center"}):
                solara.Button(
                    label="Load & Profile",
                    on_click=lambda: load_trigger.set(load_trigger.value + 1),
                    style={"background": "#111111", "color": "#ffffff"},
                )
                solara.Button(
                    label="Scan folders",
                    text=True,
                    on_click=lambda: _scan_paths(),
                )
        elif source_mode.value == "sample":
            opts = get_sample_options()
            solara.Select(
                label="Sample dataset",
                values=sorted(opts.keys()),
                value=sample_key,
            )
            if sample_key.value:
                solara.Markdown(
                    f"`{opts.get(sample_key.value, '')}`  ({_fmt_size(opts.get(sample_key.value, ''))})"
                )
            solara.Button(
                label="Load & Profile",
                on_click=lambda: load_trigger.set(load_trigger.value + 1),
                style={"background": "#111111", "color": "#ffffff"},
            )
        else:  # bigquery
            solara.InputText(
                label="Full BigQuery path (project.dataset.table)",
                value=bq_full_path,
                placeholder="my-project.analytics.events",
                on_value=lambda v: _parse_bq_path(v),
            )
            with solara.Row():
                solara.InputText(label="Project", value=bq_project, style="flex: 1;")
                solara.InputText(label="Dataset", value=bq_dataset, style="flex: 1;")
                solara.InputText(label="Table", value=bq_table, style="flex: 1;")
            solara.Markdown(
                "Credentials are resolved via `rapidsegment[gcp]` OAuth. "
                "A one-time auth is cached in `.rapidsegment_suite/oauth_cache.json`."
            )
            solara.Button(
                label="Load & Profile",
                on_click=lambda: load_trigger.set(load_trigger.value + 1),
                style={"background": "#111111", "color": "#ffffff"},
            )

        if not task.finished and not task.cancelled:
            solara.ProgressLinear(True)
            solara.Text("Profiling data with DuckDB...")
        if task.error is not None:
            solara.Error(f"Load failed: {task.error}")
        elif load_status.value:
            if load_status.value.startswith("Load error"):
                solara.Warning(load_status.value)
            else:
                solara.Success(load_status.value)


def _on_file_dropped(info):
    if not info:
        return
    name = info.get("name", "")
    data = info.get("data")
    if data is None:
        load_status.value = f"No file content received for {name}."
        return
    out_dir = Path("uploaded_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name
    dest.write_bytes(data)
    local_path.value = str(dest)
    load_status.value = f"Uploaded {name} ({len(data):,} bytes) — click Load & Profile."


def _scan_paths():
    hints = []
    for cand in ["./data", "./datasets", str(Path.home() / "Downloads"), "./Data"]:
        p = Path(cand)
        if p.is_dir():
            for f in sorted(p.iterdir())[:8]:
                if f.is_file() and f.suffix.lower() in {".csv", ".parquet", ".xlsx", ".json"}:
                    hints.append(str(f))
    if hints:
        load_status.value = "Found files:\n" + "\n".join(hints)
    else:
        load_status.value = "No data files found in ./data, ./datasets, ~/Downloads."


def _fmt_size(path: str) -> str:
    try:
        size = Path(path).stat().st_size
    except Exception:
        return "?"
    if size > 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024:.0f} KB"


# ---------------------------------------------------------------------------
# Data preview & column profiling
# ---------------------------------------------------------------------------
@solara.component
def PreviewSection():
    if preview_df.value is None:
        return
    rep = profile_report.value
    with solara.Column(style={"gap": "6px"}):
        solara.Markdown("### Data Preview (first 100 rows)")
        solara.DataTable(preview_df.value, items_per_page=10)
        solara.Markdown("### Column Profiling")
        if columns_meta_df.value is not None:
            solara.DataTable(columns_meta_df.value, items_per_page=15)
        if rep:
            with solara.Row(style={"gap": "8px"}):
                for col in rep.get("columns", []):
                    if col.get("type_warnings"):
                        solara.Warning(f"{col['column']}: {' '.join(col['type_warnings'])}")


# ---------------------------------------------------------------------------
# Target selection & validation (1.5)
# ---------------------------------------------------------------------------
def _ordered_column_values(rep: dict) -> list:
    if not rep:
        return []
    binary = [c for c in rep.get("binary_candidates", [])]
    others = [c["column"] for c in rep.get("columns", []) if c["column"] not in binary]
    return binary + others


def _on_target_changed(profiler, value):
    target_col.value = value
    binarized_target.value = None
    if not value:
        target_validation.value = None
        binarize_event.value = ""
        return
    profiler = profiler or DuckDBProfiler(db_path=".rapidsegment_suite/profiling.db")
    tv = profiler.validate_target(data_table_name.value, value)
    target_validation.value = tv
    if tv.get("is_binary"):
        binarize_event.value = ""
    else:
        binarize_event.value = tv.get("suggested_event_value") or ""


def _do_binarize(profiler):
    profiler = profiler or DuckDBProfiler(db_path=".rapidsegment_suite/profiling.db")
    if not target_col.value or not binarize_event.value:
        load_status.value = "Choose an event value to binarize."
        return
    new_col = profiler.binarize_target(data_table_name.value, target_col.value, binarize_event.value)
    binarized_target.value = new_col
    tv = profiler.validate_target(data_table_name.value, new_col)
    target_validation.value = tv
    load_status.value = f"Binarized '{target_col.value}' -> '{new_col}' (event = {binarize_event.value})."
    # refresh profile columns to include the new column
    rep = profiler.profile(data_table_name.value)
    profile_report.value = rep
    columns_meta_df.value = DuckDBProfiler.profiling_columns_df(rep)


@solara.component
def TargetSection(profiler):
    if profile_report.value is None:
        return
    rep = profile_report.value
    values = _ordered_column_values(rep)
    with solara.Column(style={"gap": "8px"}):
        solara.Markdown("### Target Column Selection & Validation")
        with solara.Row():
            solara.Select(
                label="Target variable",
                values=values,
                value=target_col,
                on_value=lambda v: _on_target_changed(profiler, v),
                style="flex: 1;",
            )
            solara.Select(
                label="Primary key (optional, for scorecard)",
                values=[""] + [c["column"] for c in rep.get("columns", [])],
                value=primary_key,
                style="flex: 1;",
            )
        tv = target_validation.value
        if tv:
            _render_target_validation(tv, profiler)


@solara.component
def _render_target_validation(tv, profiler):
    total = tv.get("total") or 0
    if tv.get("error"):
        solara.Error(tv["error"])
        return
    color = "#1b5e20"
    if tv.get("is_binary"):
        label = "Binary target"
        if tv.get("class_imbalance"):
            color = "#b71c1c"
    elif tv.get("is_multiclass"):
        label = "Multi-class target (binarization recommended)"
        color = "#e65100"
    elif tv.get("is_constant"):
        label = "Constant column - not usable as target"
        color = "#b71c1c"
    else:
        label = "Categorical target"
        color = "#e65100"

    ev_pct = tv.get("event_rate_pct")
    ev_text = f"{ev_pct}%" if ev_pct is not None else "-"
    solara.HTML(
        unsafe_innerHTML=(
            f'<div style="display:flex;gap:12px;flex-wrap:wrap;padding:10px;border:1px solid #e0e0e0;'
            f'border-radius:8px;background:#fafafa;font-size:13px;">'
            f'<span><b>Status:</b> <span style="color:{color};font-weight:600;">{label}</span></span>'
            f'<span><b>Rows:</b> {total:,}</span>'
            f'<span><b>Distinct:</b> {tv.get("distinct_vals")}</span>'
            f'<span><b>Event value:</b> {_html.escape(str(tv.get("event_value", "?")))}</span>'
            f'<span><b>Events:</b> {tv.get("events", 0):,}</span>'
            f'<span><b>Event rate:</b> <span style="color:{color};font-weight:600;">{ev_text}</span></span>'
            f'</div>'
        )
    )

    # Class distribution mini chart
    dist = tv.get("class_distribution", [])
    if tv.get("is_binary"):
        ev = str(tv.get("event_value", "1"))
        chart_data = []
        for d in dist:
            if d["value"] == "__missing__":
                continue
            is_event = str(d["value"]).lower() == ev.lower()
            chart_data.append((f"Event ({d['value']})" if is_event else f"Non-event ({d['value']})",
                               d["count"], "#111111" if is_event else "#bdbdbd"))
        solara.Markdown("**Events vs Non-events**")
        MiniBars(chart_data, height=22)
    else:
        chart_data = [
            (d["value"], d["count"], "#111111" if i == 0 else "#9e9e9e")
            for i, d in enumerate(dist[:8]) if d["value"] != "__missing__"
        ]
        solara.Markdown("**Class distribution (top classes)**")
        MiniBars(chart_data, height=18)

    if tv.get("warnings"):
        for w in tv["warnings"]:
            solara.Warning(w)

    # Binarization helper for multi-class
    if tv.get("is_multiclass"):
        with solara.Row(style={"gap": "8px", "align-items": "center"}):
            solara.Select(
                label="Event value for binarization",
                values=[d["value"] for d in dist if d["value"] != "__missing__"],
                value=binarize_event,
            )
            solara.Button(
                label="Binarize target",
                on_click=lambda: _do_binarize(profiler),
                outlined=True,
            )
        if binarized_target.value:
            solara.Success(
                f"Binarized column '{binarized_target.value}' created. "
                f"Target for the experiment will be '{binarized_target.value}'."
            )


# ---------------------------------------------------------------------------
# Data quality report (1.6)
# ---------------------------------------------------------------------------
@solara.component
def QualitySection():
    rep = profile_report.value
    if rep is None:
        return
    quality = quality_report.value
    solara.Markdown("### Data Quality Report")
    StatCards([
        ("Rows", f"{rep['total_rows']:,}"),
        ("Columns", rep["total_columns"]),
        ("Numeric", rep["num_numeric"]),
        ("Categorical", rep["num_categorical"]),
        ("Size in DuckDB", f"{rep['estimated_data_size_mb']} MB"),
    ])
    if quality:
        RecommendationBanner(quality)
        missing = quality.get("missing_summary", [])[:6]
        if missing:
            solara.Markdown("**Missing values (worst offenders)**")
            MiniBars(
                [(c["column"], c["null_pct"], "#e65100" if c["null_pct"] > 20 else "#bdbdbd")
                 for c in missing],
                height=14,
            )
        if quality.get("warnings"):
            with solara.Column(style={"gap": "2px"}):
                for w in quality["warnings"][:8]:
                    solara.Warning(w)


# ---------------------------------------------------------------------------
# Actions (1.7)
# ---------------------------------------------------------------------------
@solara.component
def ActionsSection(on_proceed):
    can_proceed = False
    reason = "Load a dataset and select a target column to continue."
    if profile_report.value is not None:
        tv = target_validation.value
        if tv is not None and tv.get("is_binary") and tv.get("events", 0) > 0:
            can_proceed = True
            reason = ""
        elif tv is not None and tv.get("is_binary") and tv.get("events", 0) == 0:
            reason = "Target has no positive events - choose another target or event value."
        elif tv is not None and tv.get("is_multiclass"):
            reason = "Multi-class target - binarize it before proceeding."
        elif tv is not None and tv.get("is_constant"):
            reason = "Target column is constant - not usable."
        else:
            reason = "Select a target column to continue."

    rep = profile_report.value
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_uri = _data_uri(_report_csv_content(rep), "text/csv") if rep else ""
    json_uri = _data_uri(_report_json_content(rep), "application/json") if rep else ""

    with solara.Column(style={"gap": "10px"}):
        if not can_proceed and reason:
            solara.Info(reason)
        with solara.Row(style={"gap": "10px", "align-items": "center", "margin-top": "6px"}):
            solara.Button(
                label="Proceed to Workbench",
                disabled=not can_proceed,
                on_click=lambda: _proceed(on_proceed),
                style={"background": "#111111", "color": "#ffffff", "font-weight": "600"},
            )
            solara.Button(
                label="Upload different file",
                text=True,
                on_click=lambda: _reset_flow(),
            )
        if rep:
            with solara.Row(style={"gap": "10px"}):
                solara.HTML(
                    unsafe_innerHTML=(
                        f'<a href="{csv_uri}" download="profiling_report_{ts}.csv" '
                        f'style="color:#111111;">Download Profiling Report (CSV)</a>'
                    )
                )
                solara.HTML(
                    unsafe_innerHTML=(
                        f'<a href="{json_uri}" download="profiling_report_{ts}.json" '
                        f'style="color:#111111;">Download Profiling Report (JSON)</a>'
                    )
                )


def _proceed(on_proceed):
    if not on_proceed:
        load_status.value = "No callback wired to the Workbench tab."
        return
    tgt = binarized_target.value or target_col.value
    if not tgt:
        load_status.value = "Select a target column first."
        return
    on_proceed(
        target=tgt,
        table_name=data_table_name.value,
        primary_key=primary_key.value or None,
    )


def _reset_flow():
    """'Upload different file' - reset the whole profiling flow."""
    load_trigger.value = 0
    profile_report.value = None
    preview_df.value = None
    columns_meta_df.value = None
    quality_report.value = None
    target_validation.value = None
    target_col.value = ""
    binarized_target.value = None
    binarize_event.value = ""
    primary_key.value = ""
    local_path.value = ""
    sample_key.value = ""
    bq_full_path.value = ""
    bq_project.value = ""
    bq_dataset.value = ""
    bq_table.value = ""
    load_status.value = "Flow reset. Choose a data source to begin."


# ---------------------------------------------------------------------------
# Top-level Module 1 component
# ---------------------------------------------------------------------------
@solara.component
def DataSourceModule(profiler=None, on_proceed=None):
    """Module 1: Enhanced Data Source & Profiling.

    Args:
        profiler: optional DuckDBProfiler instance (reused by the app).
        on_proceed: callback(target, table_name, primary_key) fired when the
                    user clicks "Proceed to Workbench".
    """
    with solara.Column(style={"gap": "12px"}):
        solara.Markdown("## Module 1 - Data Source & Profiling")
        LoadSection(profiler)
        QualitySection()
        PreviewSection()
        TargetSection(profiler)
        ActionsSection(on_proceed)
