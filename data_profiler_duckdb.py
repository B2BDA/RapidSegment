"""
Data Profiler - DuckDB ONLY for profiling (disk-persisted)
- Supports local file path (CSV/Parquet/JSON/Excel/Arrow) via DuckDB native readers
- Supports BigQuery table path via UniversalDataLoader -> ingested into DuckDB file
- Supports sample datasets bundled in the repo (./Datasets)
- Module 1 metrics: null rates, cardinality, distributions (numeric + categorical),
  type warnings, memory footprint, target validation, data quality report
"""
import duckdb
import pandas as pd
import pyarrow as pa
from pathlib import Path

SAMPLE_DATASETS_DIR = Path(__file__).resolve().parent.parent / "Datasets"
VALID_EXTS = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".arrow", ".feather", ".txt"}

class DuckDBProfiler:
    def __init__(self, db_path: str = ".rapidsegment_suite/profiling.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        try:
            self.con.execute("PRAGMA memory_limit='2GB'")
        except:
            pass

    @staticmethod
    def detect_format(file_path: str) -> str:
        """Auto-infer file format from extension."""
        ext = Path(file_path).suffix.lower()
        if ext in [".csv", ".txt"]:
            return "csv"
        if ext in [".parquet", ".pq"]:
            return "parquet"
        if ext in [".xlsx", ".xls"]:
            return "excel"
        if ext in [".json", ".jsonl", ".ndjson"]:
            return "json"
        if ext in [".arrow", ".feather"]:
            return "arrow"
        return "unknown"

    @staticmethod
    def detect_encoding(file_path: str, sample_bytes: int = 8192) -> str:
        """Best-effort encoding detection: UTF-8 (with/without BOM) or Latin-1."""
        try:
            with open(file_path, "rb") as f:
                raw = f.read(sample_bytes)
        except Exception:
            return "utf-8"
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        try:
            raw.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"

    @staticmethod
    def list_sample_datasets() -> dict:
        """Scan the bundled Datasets/ dir and return {label: file_path}."""
        found = {}
        if SAMPLE_DATASETS_DIR.is_dir():
            for p in sorted(SAMPLE_DATASETS_DIR.iterdir()):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    found[p.stem] = str(p)
        return found

    def load_sample(self, sample_key: str, table_name: str = "main_data"):
        """Load a bundled sample dataset into a DuckDB table."""
        samples = self.list_sample_datasets()
        if sample_key not in samples:
            raise ValueError(f"Sample '{sample_key}' not found. Available: {sorted(samples)}")
        return self.load_table(file_path=samples[sample_key], table_name=table_name)

    def _parse_bq_path(self, bq_path: str):
        p = bq_path.strip().replace("bq://","").replace("bigquery://","")
        parts = p.split(".")
        if len(parts) == 3:
            return {"project_id": parts[0], "dataset_id": parts[1], "table_id": parts[2]}
        elif len(parts) == 2:
            return {"project_id": None, "dataset_id": parts[0], "table_id": parts[1]}
        else:
            raise ValueError(f"Invalid BQ path {bq_path}. Expected project.dataset.table")

    def load_from_bq(self, bq_path: str = None, project_id: str = None, dataset_id: str = None, table_id: str = None, table_name: str = "main_data"):
        try:
            from rapidsegment import UniversalDataLoader
        except ImportError:
            raise ImportError("rapidsegment not installed: pip install rapidsegment needed for BigQuery")

        if bq_path and not (project_id and dataset_id and table_id):
            parsed = self._parse_bq_path(bq_path)
            project_id = project_id or parsed["project_id"]
            dataset_id = dataset_id or parsed["dataset_id"]
            table_id = table_id or parsed["table_id"]

        if not dataset_id or not table_id:
            raise ValueError("dataset_id and table_id required")

        result = UniversalDataLoader(project_id=project_id, dataset_id=dataset_id, table_id=table_id).load()

        if isinstance(result, pa.Table):
            self._ingest_arrow(result, table_name)
        elif isinstance(result, str):
            # UniversalDataLoader returned a DuckDB bigquery_scan macro string
            # (google-cloud-bigquery not installed).
            try:
                self.con.execute("INSTALL bigquery; LOAD bigquery;")
            except Exception as e:
                raise RuntimeError(
                    "BigQuery requires 'google-cloud-bigquery' (pip install rapidsegment[gcp]) "
                    f"or the DuckDB bigquery extension ({e})"
                )
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {result}")
        else:
            self.con.register(f"{table_name}_tmp", result)
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_tmp")
            self.con.unregister(f"{table_name}_tmp")
        self._checkpoint()
        return table_name

    def _ingest_arrow(self, arrow_tbl: pa.Table, table_name: str):
        """Ingest a PyArrow table into a DuckDB table (via DuckDB's arrow scan)."""
        if not isinstance(arrow_tbl, pa.Table):
            raise TypeError(f"Expected a PyArrow Table, got {type(arrow_tbl).__name__}")
        self.con.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.con.register(f"{table_name}_arrow", arrow_tbl)
        try:
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_arrow")
        finally:
            self.con.unregister(f"{table_name}_arrow")

    def _checkpoint(self):
        try:
            self.con.execute("CHECKPOINT")
        except:
            pass

    def load_table(self, file_path: str = None, table_name: str = "main_data", df=None, encoding: str = None, format: str = None):
        try:
            self.con.execute(f"DROP TABLE IF EXISTS {table_name}")
        except:
            pass

        if df is not None:
            self.con.register(f"{table_name}_tmp", df)
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_tmp")
            self.con.unregister(f"{table_name}_tmp")
            self._checkpoint()
            return table_name

        if not file_path:
            raise ValueError("file_path or df required")

        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"{file_path} not found")

        fmt = (format or self.detect_format(file_path)).lower()
        enc = encoding or self.detect_encoding(file_path)

        # Primary path: repurpose RapidSegment's UniversalDataLoader so the
        # no-code UI shares the library's ingestion and numeric normalisation.
        # UniversalDataLoader supports csv/parquet/arrow/feather/excel.
        if fmt != "json":
            try:
                from rapidsegment import UniversalDataLoader
                arrow_tbl = UniversalDataLoader(file_path=str(file_path)).load()
                self._ingest_arrow(arrow_tbl, table_name)
                self._checkpoint()
                return table_name
            except Exception:
                # Fall back to DuckDB native readers (encoding variants, etc.)
                pass

        self._load_native(file_path, table_name, fmt, enc)
        self._checkpoint()
        return table_name

    def _load_native(self, file_path: str, table_name: str, fmt: str, enc: str):
        """DuckDB native readers - fallback path when UniversalDataLoader
        cannot parse the file (e.g. JSON, non-UTF-8 CSV)."""
        ext = Path(file_path).suffix.lower()
        if fmt == "csv":
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv('{file_path}', AUTO_DETECT=TRUE, HEADER=TRUE, SAMPLE_SIZE=100000, encoding='{enc}')")
        elif fmt in ["parquet", "pq"]:
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')")
        elif fmt == "json":
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_json('{file_path}', format='auto')")
        elif fmt == "arrow":
            import pyarrow.feather as feather
            import pyarrow.ipc as ipc
            try:
                tbl = feather.read_table(file_path)
            except Exception:
                with open(file_path, "rb") as fh:
                    tbl = ipc.open_file(fh).read_all()
            self.con.register(f"{table_name}_tmp", tbl)
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_tmp")
            self.con.unregister(f"{table_name}_tmp")
        elif fmt == "excel":
            try:
                self.con.execute("INSTALL excel; LOAD excel;")
                self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_xlsx('{file_path}')")
            except Exception:
                df_tmp = pd.read_excel(file_path)
                self.con.register("tmp_excel", df_tmp)
                self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_excel")
                self.con.unregister("tmp_excel")
        else:
            try:
                if ext in [".xlsx", ".xls"]:
                    df_tmp = pd.read_excel(file_path)
                else:
                    df_tmp = pd.read_csv(file_path, encoding=enc)
                self.con.register("tmp_generic", df_tmp)
                self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_generic")
                self.con.unregister("tmp_generic")
            except Exception:
                raise ValueError(f"Unsupported file format: {ext}")

    def _get_columns_meta(self, table_name: str):
        rows = self.con.execute(f"DESCRIBE {table_name}").fetchall()
        cols = []
        for r in rows:
            col_name = r[0]
            col_type = r[1]
            cols.append({"name": col_name, "type": col_type})
        return cols

    def _is_numeric_type(self, duckdb_type: str) -> bool:
        t = duckdb_type.upper()
        return any(tok in t for tok in ["INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL", "BIGINT", "SMALLINT", "TINYINT", "UBIGINT", "HUGEINT"])

    def _is_string_type(self, duckdb_type: str) -> bool:
        t = duckdb_type.upper()
        return any(tok in t for tok in ["VARCHAR", "CHAR", "TEXT", "STRING", "UUID", "ENUM"])

    def get_db_size_info(self):
        file_size = 0
        if self.db_path != ":memory:" and Path(self.db_path).exists():
            file_size = Path(self.db_path).stat().st_size
        pragma_info = {}
        try:
            rows = self.con.execute("SELECT * FROM pragma_database_size()").fetchall()
            if rows:
                cols = [d[0] for d in self.con.description] if self.con.description else []
                pragma_info = dict(zip(cols, rows[0])) if cols else {"raw": rows[0]}
        except Exception:
            try:
                rows = self.con.execute("PRAGMA database_size").fetchall()
                pragma_info = {"pragma_raw": rows}
            except:
                pragma_info = {}
        return {
            "file_path": str(self.db_path),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024*1024), 3),
            "file_size_kb": round(file_size / 1024, 1),
            "pragma": pragma_info
        }

    @staticmethod
    def _safe_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def get_preview(self, table_name: str, n: int = 100) -> pd.DataFrame:
        """First n rows as a pandas DataFrame for the preview table."""
        return self.con.execute(f"SELECT * FROM {table_name} LIMIT {n}").df()

    def _numeric_stats(self, table_name: str, escaped: str) -> dict:
        q = f"""
            SELECT MIN({escaped}) AS min_val, MAX({escaped}) AS max_val,
                   AVG({escaped}) AS mean_val, MEDIAN({escaped}) AS median_val,
                   STDDEV({escaped}) AS std_val,
                   PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {escaped}) AS q1,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {escaped}) AS q3,
                   COUNT({escaped}) AS non_null
            FROM {table_name}
        """
        try:
            row = self.con.execute(q).fetchone()
            cols = [d[0] for d in self.con.description]
            stats = dict(zip(cols, row))
            out = {}
            for k, v in stats.items():
                try:
                    out[k] = round(float(v), 4) if v is not None else None
                except (TypeError, ValueError):
                    out[k] = None
            return out
        except Exception:
            return {"min_val": None, "max_val": None, "mean_val": None, "median_val": None,
                    "std_val": None, "q1": None, "q3": None, "non_null": None}

    def _top_values(self, table_name: str, escaped: str, total_rows: int, n: int = 5, missing_token: str = "__missing__") -> list:
        q = f"""
            SELECT CASE WHEN {escaped} IS NULL OR CAST({escaped} AS VARCHAR) IN ('', 'None', 'nan', 'NaN', 'NULL', 'null')
                        THEN '{missing_token}' ELSE CAST({escaped} AS VARCHAR) END AS val,
                   COUNT(*) AS cnt
            FROM {table_name}
            GROUP BY 1 ORDER BY cnt DESC LIMIT {n}
        """
        try:
            rows = self.con.execute(q).fetchall()
        except Exception:
            return []
        return [{"value": v, "count": int(c), "pct": round(float(c) / total_rows * 100, 2) if total_rows else 0.0}
                for v, c in rows]

    def _looks_numeric_ratio(self, table_name: str, escaped: str):
        q = f"""
            SELECT COUNT(*) FILTER (WHERE {escaped} IS NOT NULL AND TRY_CAST({escaped} AS DOUBLE) IS NOT NULL) AS castable,
                   COUNT({escaped}) AS non_null
            FROM {table_name}
        """
        try:
            castable, non_null = self.con.execute(q).fetchone()
            return int(castable or 0), int(non_null or 0)
        except Exception:
            return 0, 0

    def _column_memory_bytes(self, table_name: str, col_name: str, col_type: str, total_rows: int) -> int:
        escaped = self._safe_ident(col_name)
        try:
            if self._is_string_type(col_type):
                row = self.con.execute(f"SELECT COALESCE(SUM(LENGTH(CAST({escaped} AS VARCHAR)) + 8), 0) FROM {table_name}").fetchone()[0]
                return int(row) if row is not None else 0
            elif self._is_numeric_type(col_type) or "BOOL" in col_type.upper():
                return int(total_rows) * 8
            else:
                return int(total_rows) * 16
        except Exception:
            return int(total_rows) * 16

    def profile(self, table_name: str, target_col: str = None):
        total_rows = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        total_rows = int(total_rows) if total_rows else 0
        cols_meta = self._get_columns_meta(table_name)
        num_cols = len(cols_meta)

        per_col = []
        binary_candidates = []
        total_mem_bytes = 0
        for col in cols_meta:
            col_name = col["name"]
            col_type = col["type"]
            escaped = self._safe_ident(col_name)

            is_numeric = self._is_numeric_type(col_type)
            is_string = self._is_string_type(col_type)
            is_bool = "BOOL" in col_type.upper()

            if is_string:
                null_q = f"""
                    SELECT 
                        SUM(CASE WHEN {escaped} IS NULL OR CAST({escaped} AS VARCHAR) IN ('', 'None', 'nan', 'NaN', 'NULL', 'null') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) * 100 as null_pct,
                        COUNT(DISTINCT {escaped}) as distinct_cnt,
                        COUNT(*) - COUNT({escaped}) as null_count_raw
                    FROM {table_name}
                """
            else:
                null_q = f"""
                    SELECT 
                        SUM(CASE WHEN {escaped} IS NULL THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) * 100 as null_pct,
                        COUNT(DISTINCT {escaped}) as distinct_cnt,
                        COUNT(*) - COUNT({escaped}) as null_count_raw
                    FROM {table_name}
                """
            try:
                null_pct, distinct_cnt, null_raw = self.con.execute(null_q).fetchone()
                null_pct = float(null_pct) if null_pct is not None else 0.0
                distinct_cnt = int(distinct_cnt) if distinct_cnt is not None else 0
            except Exception:
                null_pct, distinct_cnt, null_raw = 0.0, 0, 0

            # Distribution
            if is_numeric:
                distribution = self._numeric_stats(table_name, escaped)
                distribution["kind"] = "numeric"
            elif is_bool:
                distribution = {"kind": "categorical", "top_values": self._top_values(table_name, escaped, total_rows)}
            else:
                distribution = {"kind": "categorical", "top_values": self._top_values(table_name, escaped, total_rows)}

            # Type warnings
            type_warnings = []
            if is_string:
                castable, non_null = self._looks_numeric_ratio(table_name, escaped)
                if non_null and castable / non_null >= 0.95 and distinct_cnt > 2:
                    type_warnings.append("Looks numeric but stored as text — consider casting.")
                if distinct_cnt > 1000 and distinct_cnt / max(1, total_rows - int(null_raw)) > 0.9:
                    type_warnings.append("High cardinality string — looks like an ID or free-text.")
            if is_numeric and distinct_cnt == 2:
                type_warnings.append("Binary indicator (two distinct values).")

            if 0 < distinct_cnt <= 2:
                binary_candidates.append(col_name)

            mem_bytes = self._column_memory_bytes(table_name, col_name, col_type, total_rows)
            total_mem_bytes += mem_bytes

            per_col.append({
                "column": col_name,
                "type": col_type,
                "is_numeric": is_numeric,
                "is_categorical": is_string or is_bool,
                "is_string": is_string,
                "is_bool": is_bool,
                "dtype": "numeric" if is_numeric else ("bool" if is_bool else "categorical"),
                "null_pct": round(null_pct, 2),
                "null_count": int(null_raw) if null_raw else 0,
                "distinct_count": distinct_cnt,
                "distribution": distribution,
                "type_warnings": type_warnings,
                "estimated_bytes": mem_bytes
            })

        num_numeric = sum(1 for c in per_col if c["is_numeric"])
        num_categorical = sum(1 for c in per_col if c["is_categorical"])
        num_other = num_cols - num_numeric - num_categorical

        event_info = None
        target_validation = None
        if target_col and target_col in [c["name"] for c in cols_meta]:
            target_validation = self.validate_target(table_name, target_col)
            event_info = {
                "target": target_col,
                "total": target_validation.get("total"),
                "events": target_validation.get("events"),
                "event_rate": target_validation.get("event_rate"),
                "event_rate_pct": target_validation.get("event_rate_pct"),
                "distinct_vals": target_validation.get("distinct_vals"),
                "is_binary": target_validation.get("is_binary"),
                "is_multiclass": target_validation.get("is_multiclass"),
            }

        size_info = self.get_db_size_info()

        report = {
            "table": table_name,
            "total_rows": total_rows,
            "total_columns": int(num_cols),
            "num_numeric": int(num_numeric),
            "num_categorical": int(num_categorical),
            "num_string": int(num_categorical),
            "num_other": int(num_other),
            "estimated_data_size_mb": round(total_mem_bytes / (1024 * 1024), 3),
            "columns": per_col,
            "binary_candidates": binary_candidates,
            "event_info": event_info,
            "target_validation": target_validation,
            "size_info": size_info,
            "generated_at": __import__("datetime").datetime.now().isoformat()
        }
        return report

    def validate_target(self, table_name: str, target_col: str):
        """Classify the target column: binary vs multi-class, event rate, imbalance."""
        escaped = self._safe_ident(target_col)
        col_types = {c["name"]: c["type"] for c in self._get_columns_meta(table_name)}
        if target_col not in col_types:
            return {"target": target_col, "error": "not found in table"}
        col_type = col_types[target_col]
        is_string = self._is_string_type(col_type)

        if is_string:
            vals_q = f"""
                SELECT CASE WHEN {escaped} IS NULL OR CAST({escaped} AS VARCHAR) IN ('', 'None', 'nan', 'NaN', 'NULL', 'null')
                            THEN '__missing__' ELSE CAST({escaped} AS VARCHAR) END AS val, COUNT(*) AS cnt
                FROM {table_name} GROUP BY 1 ORDER BY cnt DESC
            """
        else:
            vals_q = f"""
                SELECT CASE WHEN {escaped} IS NULL THEN '__missing__' ELSE CAST({escaped} AS VARCHAR) END AS val, COUNT(*) AS cnt
                FROM {table_name} GROUP BY 1 ORDER BY cnt DESC
            """
        rows = self.con.execute(vals_q).fetchall()
        # Normalize values (e.g. UDL casts numerics to float64, so 0/1 arrive
        # as "0.0"/"1.0") so binary detection works for 0.0/1.0 columns too.
        def _norm_key(k):
            k = k.strip().lower() if isinstance(k, str) else str(k)
            if len(k) > 2 and k.endswith(".0") and k[:-2].isdigit():
                return k[:-2]
            return k
        class_distribution = [{"value": _norm_key(v), "count": int(c)} for v, c in rows]
        total = sum(c for _, c in rows)
        non_missing = [c for v, c in rows if _norm_key(v) != "__missing__"]
        distinct_vals = len(non_missing)

        # Normalize values for binary detection
        norm = {}
        for v, c in rows:
            key = _norm_key(v)
            norm[key] = norm.get(key, 0) + c

        binary_map = None
        if distinct_vals == 2:
            keys = {str(k) for k in norm.keys()}
            if keys == {"0", "1"}:
                binary_map = {"event": "1", "non_event": "0"}
            elif keys == {"true", "false"}:
                binary_map = {"event": "true", "non_event": "false"}
            elif keys == {"yes", "no"}:
                binary_map = {"event": "yes", "non_event": "no"}
            elif keys == {"y", "n"}:
                binary_map = {"event": "y", "non_event": "n"}

        is_binary = binary_map is not None
        is_multiclass = distinct_vals > 2
        is_constant = distinct_vals <= 1

        event_value = binary_map["event"] if binary_map else None
        events = norm.get(_norm_key(event_value), 0) if event_value else 0
        # For 0/1 numeric stored as string, events = count of "1"
        event_rate = events / total if total else 0.0
        event_rate_pct = round(event_rate * 100, 2)

        warnings = []
        if is_constant:
            warnings.append("Column has a single value — not usable as a target.")
        elif is_multiclass:
            warnings.append("Multi-class target — binarization recommended before segmentation.")
        elif is_binary:
            class_imbalance = event_rate < 0.01 or event_rate > 0.99
            if class_imbalance:
                warnings.append(f"Severe class imbalance detected ({event_rate_pct:.1f}% event rate).")
        if events == 0:
            warnings.append("No positive events detected for the default event value.")

        # Suggested event value for multi-class: rarest class (common anomaly/default view)
        suggested_event = None
        if is_multiclass and non_missing:
            counts = sorted(rows, key=lambda r: r[1])
            for v, c in counts:
                if _norm_key(v) != "__missing__":
                    suggested_event = _norm_key(v)
                    break

        return {
            "target": target_col,
            "dtype": col_type,
            "is_string": is_string,
            "total": total,
            "events": events,
            "non_events": total - events,
            "event_rate": round(event_rate, 6),
            "event_rate_pct": event_rate_pct,
            "distinct_vals": distinct_vals,
            "is_binary": is_binary,
            "is_multiclass": is_multiclass,
            "is_constant": is_constant,
            "event_value": event_value,
            "class_imbalance": is_binary and (event_rate < 0.01 or event_rate > 0.99),
            "class_distribution": class_distribution,
            "suggested_event_value": suggested_event,
            "warnings": warnings
        }

    def binarize_target(self, table_name: str, target_col: str, event_value, new_name: str = None) -> str:
        """Create a binary 0/1 column from a target (handles multi-class binarization)."""
        new_name = new_name or f"{target_col}__bin"
        escaped_old = self._safe_ident(target_col)
        escaped_new = self._safe_ident(new_name)
        if event_value is None:
            raise ValueError("event_value required to binarize the target")
        try:
            self.con.execute(f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {escaped_new}")
        except Exception:
            pass
        self.con.execute(f"ALTER TABLE {table_name} ADD COLUMN {escaped_new} BIGINT")
        # UDL casts numerics to float64, so "0" may be stored as "0.0": match both forms.
        ev = str(event_value)
        ev_alt = ev if ev.endswith(".0") else f"{ev}.0"
        self.con.execute(
            f"UPDATE {table_name} SET {escaped_new} = CASE "
            f"WHEN TRIM(CAST({escaped_old} AS VARCHAR)) = ? OR TRIM(CAST({escaped_old} AS VARCHAR)) = ? "
            f"THEN 1 ELSE 0 END",
            [ev, ev_alt]
        )
        self._checkpoint()
        return new_name

    def data_quality_report(self, table_name: str, report: dict = None) -> dict:
        """Overall data quality score, missing summary and recommendations."""
        report = report or self.profile(table_name)
        warnings = []
        cols = report.get("columns", [])

        missing_summary = sorted([c for c in cols if c["null_pct"] > 0], key=lambda c: -c["null_pct"])
        worst_offenders = [c for c in missing_summary if c["null_pct"] > 20]
        for c in worst_offenders:
            warnings.append(f"Column '{c['column']}' has {c['null_pct']:.1f}% missing values.")

        for c in cols:
            for w in c.get("type_warnings", []):
                warnings.append(f"Column '{c['column']}': {w}")

        # Score heuristic (0-100)
        score = 100
        score -= min(30, len(worst_offenders) * 5)
        score -= min(10, sum(1 for c in cols if c["distinct_count"] > 100000) * 2)
        if report.get("total_rows", 0) == 0:
            score = 0
        score = max(0, min(100, score))

        if score >= 85:
            status = "ready"
            recommendation = "Data ready for segmentation."
        elif score >= 60:
            status = "warning"
            recommendation = "Data usable, but review the flagged columns below."
        else:
            status = "attention"
            recommendation = "Data needs cleanup before segmentation."

        return {
            "quality_score": int(score),
            "status": status,
            "recommendation": recommendation,
            "warnings": warnings,
            "missing_summary": missing_summary,
            "worst_offenders": worst_offenders,
            "total_rows": report.get("total_rows"),
            "total_columns": report.get("total_columns"),
            "num_numeric": report.get("num_numeric"),
            "num_categorical": report.get("num_categorical"),
            "estimated_data_size_mb": report.get("estimated_data_size_mb")
        }

    @staticmethod
    def profiling_columns_df(report: dict) -> pd.DataFrame:
        """Column-level profiling report as a DataFrame (for the metadata panel + export)."""
        rows = []
        for c in report.get("columns", []):
            d = c.get("distribution", {})
            if d.get("kind") == "numeric":
                distrib = (f"min {d.get('min_val')} | max {d.get('max_val')} | mean {d.get('mean_val')} | "
                           f"median {d.get('median_val')}")
            else:
                tops = "; ".join(f"{t['value']} ({t['count']})" for t in d.get("top_values", [])[:3])
                distrib = tops or "-"
            rows.append({
                "column": c["column"],
                "type": c["type"],
                "dtype": c["dtype"],
                "cardinality": c["distinct_count"],
                "null_pct": c["null_pct"],
                "distribution": distrib,
                "warnings": "; ".join(c.get("type_warnings", [])) or "-"
            })
        return pd.DataFrame(rows)

    def get_sample(self, table_name: str, n: int = 10):
        rows = self.con.execute(f"SELECT * FROM {table_name} LIMIT {n}").fetchall()
        cols = [c[0] for c in self.con.execute(f"DESCRIBE {table_name}").fetchall()]
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        try:
            self.con.close()
        except:
            pass
