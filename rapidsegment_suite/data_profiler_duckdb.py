"""
Data Profiler - DuckDB ONLY for profiling (disk-persisted)
- Supports local file path (CSV/Parquet/JSON/Excel) via DuckDB native readers
- Supports BigQuery table path via UniversalDataLoader -> ingested into DuckDB file
- Metrics: null rates, num categorical/numerical, size on disk using persisted DuckDB file
"""
import duckdb
from pathlib import Path

class DuckDBProfiler:
    def __init__(self, db_path: str = ".rapidsegment_suite/profiling.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        try:
            self.con.execute("PRAGMA memory_limit='2GB'")
        except:
            pass

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

        loader = UniversalDataLoader(project_id=project_id, dataset_id=dataset_id, table_id=table_id)
        arrow_tbl = loader.load() if hasattr(loader, 'load') else loader
        df = arrow_tbl.to_pandas() if hasattr(arrow_tbl, 'to_pandas') else arrow_tbl

        self.con.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.con.register(f"{table_name}_tmp", df)
        self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_tmp")
        self.con.unregister(f"{table_name}_tmp")
        try:
            self.con.execute("CHECKPOINT")
        except:
            pass
        return table_name

    def load_table(self, file_path: str = None, table_name: str = "main_data", df=None):
        try:
            self.con.execute(f"DROP TABLE IF EXISTS {table_name}")
        except:
            pass

        if df is not None:
            self.con.register(f"{table_name}_tmp", df)
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_tmp")
            self.con.unregister(f"{table_name}_tmp")
            try:
                self.con.execute("CHECKPOINT")
            except:
                pass
            return table_name

        if not file_path:
            raise ValueError("file_path or df required")

        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"{file_path} not found")

        ext = p.suffix.lower()

        if ext == ".csv":
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv('{file_path}', AUTO_DETECT=TRUE, HEADER=TRUE, SAMPLE_SIZE=100000)")
        elif ext in [".parquet", ".pq"]:
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')")
        elif ext in [".json", ".jsonl", ".ndjson"]:
            self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_json('{file_path}')")
        else:
            try:
                self.con.execute("INSTALL excel; LOAD excel;")
                self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_xlsx('{file_path}')")
            except Exception:
                import pandas as pd
                if ext in [".xlsx", ".xls"]:
                    df_tmp = pd.read_excel(file_path)
                else:
                    df_tmp = pd.read_csv(file_path)
                self.con.register("tmp_excel", df_tmp)
                self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_excel")
                self.con.unregister("tmp_excel")

        try:
            self.con.execute("CHECKPOINT")
        except:
            pass
        return table_name

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

    def profile(self, table_name: str, target_col: str = None):
        total_rows = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        cols_meta = self._get_columns_meta(table_name)
        num_cols = len(cols_meta)

        num_numeric = sum(1 for c in cols_meta if self._is_numeric_type(c["type"]))
        num_categorical = sum(1 for c in cols_meta if self._is_string_type(c["type"]))
        num_other = num_cols - num_numeric - num_categorical

        per_col = []
        for col in cols_meta:
            col_name = col["name"]
            col_type = col["type"]
            escaped = f'"{col_name}"'

            if self._is_string_type(col_type):
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
                null_pct = 0.0
                distinct_cnt = 0
                null_raw = 0

            per_col.append({
                "column": col_name,
                "type": col_type,
                "is_numeric": self._is_numeric_type(col_type),
                "is_categorical": self._is_string_type(col_type),
                "is_string": self._is_string_type(col_type),
                "null_pct": round(null_pct, 2),
                "null_count": int(null_raw) if 'null_raw' in locals() else 0,
                "distinct_count": distinct_cnt
            })

        event_info = None
        if target_col and target_col in [c["name"] for c in cols_meta]:
            escaped_target = f'"{target_col}"'
            try:
                q = f"""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN {escaped_target} IS NULL THEN 0 ELSE CAST({escaped_target} AS DOUBLE) END) as events,
                        AVG(CAST({escaped_target} AS DOUBLE)) as event_rate,
                        COUNT(DISTINCT {escaped_target}) as distinct_vals,
                        MIN(CAST({escaped_target} AS DOUBLE)) as min_val,
                        MAX(CAST({escaped_target} AS DOUBLE)) as max_val
                    FROM {table_name}
                    WHERE {escaped_target} IS NOT NULL
                """
                row = self.con.execute(q).fetchone()
                if row:
                    total, events, event_rate, distinct_vals, min_val, max_val = row
                    event_info = {
                        "target": target_col,
                        "total": int(total) if total else 0,
                        "events": float(events) if events else 0,
                        "event_rate": float(event_rate) if event_rate is not None else 0.0,
                        "event_rate_pct": round(float(event_rate)*100, 2) if event_rate is not None else 0.0,
                        "distinct_vals": int(distinct_vals) if distinct_vals else 0,
                        "min": float(min_val) if min_val is not None else None,
                        "max": float(max_val) if max_val is not None else None,
                        "is_binary": distinct_vals == 2 if distinct_vals else False
                    }
            except Exception:
                try:
                    q2 = f"""SELECT COUNT(*) as total, COUNT(DISTINCT {escaped_target}) as distinct_vals FROM {table_name}"""
                    total, distinct_vals = self.con.execute(q2).fetchone()
                    event_info = {
                        "target": target_col,
                        "total": int(total),
                        "distinct_vals": int(distinct_vals),
                        "event_rate": None,
                        "note": "Categorical target"
                    }
                except Exception as e2:
                    event_info = {"target": target_col, "error": str(e2)}

        size_info = self.get_db_size_info()

        report = {
            "table": table_name,
            "total_rows": int(total_rows),
            "total_columns": int(num_cols),
            "num_numeric": int(num_numeric),
            "num_categorical": int(num_categorical),
            "num_string": int(num_categorical),
            "num_other": int(num_other),
            "columns": per_col,
            "event_info": event_info,
            "size_info": size_info,
            "generated_at": __import__("datetime").datetime.now().isoformat()
        }
        return report

    def get_sample(self, table_name: str, n: int = 10):
        rows = self.con.execute(f"SELECT * FROM {table_name} LIMIT {n}").fetchall()
        cols = [c[0] for c in self.con.execute(f"DESCRIBE {table_name}").fetchall()]
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        try:
            self.con.close()
        except:
            pass
