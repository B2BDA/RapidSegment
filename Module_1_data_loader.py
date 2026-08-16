"""
RapidSegment — Universal Data Loader UI
Run with:  solara run udl_app.py
"""
import tempfile
import os
import solara
import duckdb
from rapidsegment.utils.data_loader import UniversalDataLoader

# ── Reactive state ─────────────────────────────────────────────────────────────
source     = solara.reactive("Local File")  # "Local File" | "BigQuery" | "Upload / Fallback"
project_id = solara.reactive("")
dataset_id = solara.reactive("")
table_id   = solara.reactive("")
file_path  = solara.reactive("")
status     = solara.reactive("")
db_path    = solara.reactive(None)   # path to persisted DuckDB file — no in-memory table
preview_df = solara.reactive(None)   # 20-row pandas slice for UI only

SOURCES = ["Local File", "BigQuery", "Upload / Fallback"]
DB_FILE = os.path.join(tempfile.gettempdir(), "udl_rapidsegment.duckdb")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _persist_and_preview(arrow_table, label: str) -> None:
    """Write PyArrow table to DuckDB on disk, then discard it from memory."""
    # Persist to disk
    con = duckdb.connect(DB_FILE)
    con.execute("DROP TABLE IF EXISTS udl_data")
    con.execute("CREATE TABLE udl_data AS SELECT * FROM arrow_table")
    n_rows = con.execute("SELECT COUNT(*) FROM udl_data").fetchone()[0]
    n_cols = con.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name='udl_data'").fetchone()[0]
    preview = con.execute("SELECT * FROM udl_data LIMIT 20").df()
    con.close()

    # arrow_table goes out of scope here — no reference kept
    db_path.set(DB_FILE)
    preview_df.set(preview)
    status.set(f"✓ {label} — {n_rows:,} rows × {n_cols} cols  (persisted to DuckDB)")


# ── Components ─────────────────────────────────────────────────────────────────
@solara.component
def StatusBar():
    s = status.value
    if not s:
        return
    color = "#22c55e" if s.startswith("✓") else "#ef4444"
    solara.Text(s, style=f"color:{color};font-weight:500;margin-top:10px;")


@solara.component
def DataPreview():
    df = preview_df.value
    if df is None:
        return
    solara.Markdown(f"**Preview — first 20 rows**  ·  DuckDB: `{db_path.value}`")
    solara.DataFrame(df)


# ── Main Page ──────────────────────────────────────────────────────────────────
@solara.component
def Page():
    solara.Title("RapidSegment — Universal Data Loader")

    with solara.Card("Universal Data Loader", style="max-width:760px;margin:auto;"):

        solara.Markdown("#### Select data source")
        solara.ToggleButtonsSingle(
            value=source.value,
            values=SOURCES,
            on_value=source.set,
        )
        solara.Text("")

        # ── BigQuery ──────────────────────────────────────────────────────────
        if source.value == "BigQuery":
            solara.Markdown("_Requires `google-cloud-bigquery` and GCP credentials configured._")
            solara.InputText("GCP Project ID", value=project_id.value, on_value=project_id.set)
            solara.InputText("Dataset ID",     value=dataset_id.value, on_value=dataset_id.set)
            solara.InputText("Table ID",       value=table_id.value,   on_value=table_id.set)

            def _load_bq():
                try:
                    status.set("Connecting to BigQuery…")
                    data = UniversalDataLoader(
                        project_id=project_id.value or None,
                        dataset_id=dataset_id.value or None,
                        table_id=table_id.value or None,
                    ).load()
                    _persist_and_preview(data, "Loaded from BigQuery")
                except Exception as exc:
                    status.set(f"Error: {exc}")

            solara.Button("Load from BigQuery", on_click=_load_bq, color="primary")

        # ── Local File ────────────────────────────────────────────────────────
        elif source.value == "Local File":
            solara.Markdown("_Supported: **CSV · Parquet · Arrow / Feather · Excel**_")
            solara.InputText("File path", value=file_path.value, on_value=file_path.set)

            def _load_file():
                try:
                    status.set("Reading file…")
                    data = UniversalDataLoader(file_path=file_path.value or None).load()
                    _persist_and_preview(data, "Loaded from file")
                except Exception as exc:
                    status.set(f"Error: {exc}")

            solara.Button("Load File", on_click=_load_file, color="primary")

        # ── Upload / Fallback ─────────────────────────────────────────────────
        elif source.value == "Upload / Fallback":
            solara.Markdown(
                "_Drop a file — UDL receives it as `fallback_data` (highest priority path). "
                "Supported: **CSV · Parquet · Arrow / Feather · Excel**_"
            )

            def _on_file(f: dict):
                try:
                    name = f["name"]
                    ext  = os.path.splitext(name)[-1]
                    status.set(f"Loading '{name}'…")
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(f["data"])
                        tmp_path = tmp.name
                    raw  = UniversalDataLoader(file_path=tmp_path).load()
                    data = UniversalDataLoader().load(fallback_data=raw)
                    os.unlink(tmp_path)
                    _persist_and_preview(data, f"Loaded via fallback ('{name}')")
                except Exception as exc:
                    status.set(f"Error: {exc}")

            solara.FileDrop(
                label="Drop file here or click to browse",
                on_file=_on_file,
                lazy=False,
            )

        StatusBar()
        DataPreview()