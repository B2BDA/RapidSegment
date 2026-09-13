"""
Unified Data Ingestion Layer
============================
Multi‑format data loader supporting Local Files (CSV, Parquet, Arrow, Excel),
In‑Memory PyArrow Tables, and Google Cloud BigQuery Storage API streams.

Author: Bishwarup Biswas + Gemini + DeepSeek
Python Version: 3.9+
"""

import logging
import os
import re
import tempfile
import uuid
from typing import Any, Optional, Union

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_pq

logger = logging.getLogger("StrategicEngine.DataLoader")


class UniversalDataLoader:
    """
    Handles multi‑source data ingestion, normalising inputs into highly optimised
    in‑memory PyArrow Tables suitable for vectorised downstream compute engines.

    The loader automatically detects the source type based on constructor arguments.
    If a `fallback_data` object is passed to `load()`, it takes precedence.

    Args:
        project_id: (Optional) GCP project ID for BigQuery.
        dataset_id: (Optional) BigQuery dataset ID.
        table_id: (Optional) BigQuery table ID.
        file_path: (Optional) Local file path (CSV, Parquet, Arrow/Feather, Excel).
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.file_path = file_path

    def load(self, fallback_data: Optional[Any] = None) -> Union[pa.Table, str]:
        """
        Auto‑detects the source configuration and loads the dataset.

        Priority order:
            1. If `fallback_data` is provided, it is returned (with type normalisation).
            2. If BigQuery identifiers are set, load from BigQuery.
            3. If a local `file_path` is provided, load from file.

        Args:
            fallback_data: Optional pre‑loaded data (e.g., a PyArrow Table or any
                           object that can be passed to DuckDB directly).

        Returns:
            A PyArrow Table, or a DuckDB scan macro string (when BigQuery client
            is not available and fallback is not provided).
        """
        # Scenario 1: Direct in‑memory object
        if fallback_data is not None:
            if isinstance(fallback_data, pa.Table):
                logger.info("📥 Ingesting directly provided in‑memory PyArrow Table.")
                return self._cast_table_numerics_to_float(fallback_data)
            logger.info("📥 Using provided fallback data (non‑Arrow) as‑is.")
            return fallback_data

        # Scenario 2: BigQuery
        if self.dataset_id and self.table_id:
            return self._load_from_bigquery()

        # Scenario 3: Local file
        if self.file_path:
            return self._load_from_file()

        raise ValueError(
            "Invalid Configuration: You must provide either a valid `file_path`, "
            "BigQuery identifiers, or pass an explicit `fallback_data` object."
        )

    @staticmethod
    def _cast_table_numerics_to_float(table: pa.Table) -> pa.Table:
        """
        Casts all numeric columns in a PyArrow Table to float64.

        This ensures consistent numerical precision across downstream operations.

        Args:
            table: Input PyArrow Table.

        Returns:
            A new table with numeric columns cast to float64.
        """
        if not isinstance(table, pa.Table):
            return table

        new_columns = []
        new_fields = []

        for i, field in enumerate(table.schema):
            # Check if the type is integer, floating, or decimal
            if (
                pa.types.is_integer(field.type)
                or pa.types.is_floating(field.type)
                or pa.types.is_decimal(field.type)
            ):
                try:
                    casted_col = pc.cast(
                        table.column(i), pa.float64(), safe=False
                    )
                    new_columns.append(casted_col)
                    new_fields.append(
                        pa.field(field.name, pa.float64(), nullable=field.nullable)
                    )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to cast column {field.name} to float64. Reason: {e}"
                    )
                    new_columns.append(table.column(i))
                    new_fields.append(field)
            else:
                new_columns.append(table.column(i))
                new_fields.append(field)

        return pa.Table.from_arrays(new_columns, schema=pa.schema(new_fields))

    def _load_from_file(self) -> pa.Table:
        """
        Parses a local file using high‑performance C++ Arrow readers.

        Supports: .parquet, .csv, .arrow / .feather, .xlsx / .xls.

        Returns:
            PyArrow Table with numeric columns cast to float64.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at: {self.file_path}")

        ext = os.path.splitext(self.file_path)[-1].lower()
        logger.info(f"📂 Loading file: {self.file_path} (extension: {ext})")

        if ext == ".parquet":
            table = pa_pq.read_table(self.file_path)
        elif ext == ".csv":
            table = pa_csv.read_csv(self.file_path)
        elif ext in (".arrow", ".feather"):
            with pa.memory_map(self.file_path, "r") as source:
                table = pa.ipc.open_file(source).read_all()
        elif ext in (".xlsx", ".xls"):
            table = self._load_excel_to_arrow()
        else:
            raise ValueError(f"Unsupported file format: '{ext}'.")

        return self._cast_table_numerics_to_float(table)

    def _load_excel_to_arrow(self) -> pa.Table:
        """
        Parses an Excel file using openpyxl with positional column tracking.

        Returns:
            PyArrow Table.
        """
        logger.info("📊 Parsing Excel spreadsheet via positional column tracking...")
        try:
            import openpyxl

            wb = openpyxl.load_workbook(
                self.file_path, data_only=True, read_only=True
            )
            sheet = wb.active
            rows = sheet.iter_rows(values_only=True)

            headers = next(rows)
            if not headers:
                raise ValueError("The Excel file appears to be empty.")

            # Use column indices to prevent header‑shift corruption
            column_names = [
                f"{h}" if h is not None else f"_col_{i}"
                for i, h in enumerate(headers)
            ]
            data_columns = {name: [] for name in column_names}

            for row in rows:
                for i, name in enumerate(column_names):
                    val = row[i] if i < len(row) else None
                    data_columns[name].append(val)

            wb.close()
            return pa.Table.from_pydict(data_columns)

        except ImportError:
            raise ImportError(
                "Dependency missing: `pip install openpyxl` required for Excel files."
            )

    def _load_from_bigquery(self) -> Union[pa.Table, str]:
        """
        Resolves BigQuery ingestion using cost‑optimised metadata inspection.

        If the `google‑cloud‑bigquery` library is available, streams the table
        as a PyArrow Table. Otherwise, returns a DuckDB scan macro string for
        later execution (requires DuckDB's BigQuery extension).

        Returns:
            PyArrow Table or a DuckDB macro string.
        """
        full_bq_path = (
            f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            if self.project_id
            else f"{self.dataset_id}.{self.table_id}"
        )
        logger.info(f"☁️ Initialising BigQuery client for: {full_bq_path}")

        try:
            from google.cloud import bigquery

            bq_client = bigquery.Client(project=self.project_id)
            full_table_ref = (
                f"{self.project_id or bq_client.project}."
                f"{self.dataset_id}.{self.table_id}"
            )

            # Fetch schema via get_table (cheaper than INFORMATION_SCHEMA)
            table = bq_client.get_table(full_table_ref)

            select_clauses = []
            for field in table.schema:
                # Cast numeric types to FLOAT64 for consistency
                if field.field_type in (
                    "NUMERIC",
                    "BIGNUMERIC",
                    "DECIMAL",
                    "INTEGER",
                    "INT64",
                    "FLOAT",
                    "FLOAT64",
                ):
                    select_clauses.append(
                        f"SAFE_CAST(`{field.name}` AS FLOAT64) AS `{field.name}`"
                    )
                else:
                    select_clauses.append(f"`{field.name}`")

            query = f"SELECT {', '.join(select_clauses)} FROM `{full_table_ref}`"
            arrow_table = bq_client.query(query).to_arrow()
            logger.info(f"✅ Loaded {len(arrow_table):,} rows from BigQuery.")
            return arrow_table

        except ImportError:
            logger.warning(
                "⚠️ google‑cloud‑bigquery not found. Returning DuckDB scan macro. "
                "This macro requires the BigQuery extension to be loaded in DuckDB."
            )
            # Return a string that DuckDB can interpret as a table reference
            # (assuming the extension is loaded)
            return f"bigquery_scan('{self.project_id or 'default'}', '{self.dataset_id}', '{self.table_id}')"

    @staticmethod
    def _validate_identifier(name: str, arg: str) -> str:
        """Ensures a caller-supplied table name is a safe SQL identifier."""
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"{arg} must be a valid SQL identifier; got {name!r}")
        return name

    @staticmethod
    def _drop_object_if_exists(con: duckdb.DuckDBPyConnection, name: str) -> None:
        """Drops a table or view by its exact catalog kind so creation is idempotent."""
        rows = con.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchall()
        for (table_type,) in rows:
            if table_type.upper() == "VIEW":
                con.execute(f'DROP VIEW IF EXISTS "{name}"')
            else:
                con.execute(f'DROP TABLE IF EXISTS "{name}"')

    @staticmethod
    def _cast_numeric_columns_to_double(
        con: duckdb.DuckDBPyConnection, table_name: str
    ) -> None:
        """Memory-light on-disk cast of every numeric column to DOUBLE (float64)."""
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ?",
            [table_name],
        ).fetchall()
        for name, dtype in cols:
            up = dtype.upper()
            if (
                any(
                    k in up
                    for k in ("INT", "FLOAT", "DECIMAL", "REAL", "DOUBLE", "NUMERIC")
                )
                and "DOUBLE" not in up
            ):
                con.execute(
                    f'ALTER TABLE "{table_name}" ALTER COLUMN "{name}" TYPE DOUBLE'
                )

    def stream_to_duckdb(
        self,
        db_path: Optional[str] = None,
        path: Optional[str] = None,
        encoding: Optional[str] = None,
        data: Optional[Any] = None,
        table_name: str = "udl_data",
        scorer_view_name: str = "df",
    ) -> str:
        """Stream a data source straight into a persisted DuckDB database file.

        Handles local files (read straight from disk, memory-light), in-memory
        objects (PyArrow Table, pandas DataFrame, or any DuckDB-registrable
        relation), and BigQuery tables. Numeric columns are cast to DOUBLE on-disk
        to match the loader's float64 convention, and a ``df`` view is added over
        the data table so the returned path can be handed directly to the engine's
        builder, evaluator, and scorer.

        Supported files: .csv/.tsv (read_csv_auto), .parquet/.pq (read_parquet),
        .arrow/.feather (read_ipc). Excel is not streamed directly — call
        ``UniversalDataLoader(file_path=...).load()`` and pass the result as ``data=``.

        Args:
            db_path: Optional. Database name or file path to write to. A ``.duckdb``
                extension is appended automatically when missing and the parent
                directory is created as needed. When omitted, a unique database is
                auto-created under the system temp directory.
            path: Source file path (overrides the constructor's ``file_path`` when
                both are set).
            encoding: Optional encoding hint ("Latin-1" is honored for CSV/TSV).
            data: In-memory source object (PyArrow Table, pandas DataFrame, or any
                DuckDB-registrable relation) used instead of a file.
            table_name: Name of the persisted data table (default 'udl_data'). A
                custom name is fine — a ``udl_data`` view is always added as an
                alias, so `StrategicSegmentBuilder.extract_segments` and
                `evaluate_final_coverage` keep working when handed the returned
                path (both resolve 'udl_data' by that fixed name).
            scorer_view_name: Optional view name aliasing the data table so
                `StrategicSegmentScore.calculate_and_export_weights` can consume
                the same file (default 'df'). Pass None/'' to skip creating it.

        Returns:
            Absolute path to the created .duckdb file, usable directly by
            ``extract_segments``, ``evaluate_final_coverage``, and
            ``calculate_and_export_weights``.

        Examples:
            >>> from rapidsegment.utils.data_loader import UniversalDataLoader
            >>> out = UniversalDataLoader(file_path="bank_train.csv").stream_to_duckdb("bank_data")
            >>> builder.extract_segments(out)          # reads table/view 'udl_data'
            >>> builder.evaluate_final_coverage(out)   # reads table/view 'udl_data'
            >>> scorer.calculate_and_export_weights(out, "w.json")  # reads table/view 'df'
        """
        if not db_path:
            unique_id = uuid.uuid4().hex[:8]
            db_path = os.path.join(
                tempfile.gettempdir(), f"rapidsegment_udl_{unique_id}.duckdb"
            )
            logger.info(f"🗄️ No db_path given — created default database at: {db_path}")
        table_name = self._validate_identifier(table_name, "table_name")
        if scorer_view_name:
            scorer_view_name = self._validate_identifier(
                scorer_view_name, "scorer_view_name"
            )

        # Resolve the source, mirroring load()'s priority order.
        source_kind: str
        source_path: Optional[str] = None
        if data is not None:
            source_kind = "object"
        elif path is not None:
            source_kind = "file"
            source_path = os.path.abspath(path)
        elif self.file_path is not None:
            source_kind = "file"
            source_path = os.path.abspath(self.file_path)
        elif self.dataset_id and self.table_id:
            bq_result = self._load_from_bigquery()
            if isinstance(bq_result, str):
                raise ValueError(
                    "stream_to_duckdb needs the 'google-cloud-bigquery' library to "
                    "materialise BigQuery data; install it and retry."
                )
            data = bq_result
            source_kind = "object"
        else:
            raise ValueError(
                "No data source configured. Pass `path=`, constructor `file_path`, "
                "BigQuery identifiers (project_id/dataset_id/table_id), or in-memory `data=`."
            )

        if source_kind == "file" and source_path:
            ext = os.path.splitext(source_path)[1].lower()
            if ext not in (".csv", ".tsv", ".parquet", ".pq", ".arrow", ".feather"):
                raise ValueError(
                    f"stream_to_duckdb does not support '{ext}' files directly; "
                    f"call UniversalDataLoader(file_path='{source_path}').load() and "
                    "pass the result as `data=` instead."
                )

        # Normalize the target: absolute path, always ending in .duckdb.
        db_name = str(db_path)
        if not db_name.lower().endswith(".duckdb"):
            db_name = f"{db_name}.duckdb"
        db_abs = os.path.abspath(db_name)
        parent = os.path.dirname(db_abs)
        if parent:
            os.makedirs(parent, exist_ok=True)

        con = duckdb.connect(db_abs)
        try:
            self._drop_object_if_exists(con, table_name)
            if source_kind == "file" and source_path:
                if ext in (".csv", ".tsv"):
                    opts = "header=true, sample_size=-1"
                    if encoding == "Latin-1":
                        opts += ", encoding='LATIN-1'"
                    con.execute(
                        f'CREATE TABLE "{table_name}" AS SELECT * FROM read_csv_auto(?, {opts})',
                        [source_path],
                    )
                elif ext in (".parquet", ".pq"):
                    con.execute(
                        f'CREATE TABLE "{table_name}" AS SELECT * FROM read_parquet(?)',
                        [source_path],
                    )
                else:  # .arrow / .feather
                    con.execute(
                        f'CREATE TABLE "{table_name}" AS SELECT * FROM read_ipc(?)',
                        [source_path],
                    )
            else:
                con.register("__rs_stream_src", data)
                con.execute(
                    f'CREATE TABLE "{table_name}" AS SELECT * FROM __rs_stream_src'
                )
            # float64 convention, memory-light (on-disk ALTER per numeric column)
            self._cast_numeric_columns_to_double(con, table_name)
            # Engine-compatibility aliases: keep a `udl_data` view (builder /
            # evaluator resolve it by that fixed name) and a `df` view (scorer).
            # Views are catalog metadata only — a few bytes, never a data copy.
            engine_base = "udl_data"
            if table_name != engine_base:
                self._drop_object_if_exists(con, engine_base)
                con.execute(
                    f'CREATE VIEW "{engine_base}" AS SELECT * FROM "{table_name}"'
                )
            if scorer_view_name and scorer_view_name != table_name:
                self._drop_object_if_exists(con, scorer_view_name)
                con.execute(
                    f'CREATE VIEW "{scorer_view_name}" AS SELECT * FROM "{table_name}"'
                )
        finally:
            con.close()
        return db_abs

    @staticmethod
    def duckdb_to_arrow(
        db: Union[str, os.PathLike, duckdb.DuckDBPyConnection],
        table_name: str = "udl_data",
    ) -> pa.Table:
        """Read a table from a DuckDB file (or an open DuckDB connection) into PyArrow.

        Args:
            db: A DuckDB database file path or an already-open ``duckdb`` connection.
            table_name: Table or view to read (default 'udl_data'; artifacts written
                by ``stream_to_duckdb`` also carry the 'df' scorer view).

        Returns:
            PyArrow Table of the requested table.

        Raises:
            ValueError: If ``table_name`` does not exist in the DuckDB source.
        """
        owned = False
        if isinstance(db, duckdb.DuckDBPyConnection):
            con = db
        else:
            db_file = str(db)
            try:
                con = duckdb.connect(db_file, read_only=True)
            except duckdb.Error:
                # A same-process connection with a different (read-write) config
                # may already hold the file; match it. Never create a missing DB.
                if not os.path.exists(db_file):
                    raise
                con = duckdb.connect(db_file)
            owned = True
        try:
            exists = con.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            if not exists:
                available = ", ".join(
                    r[0]
                    for r in con.execute(
                        "SELECT table_name FROM information_schema.tables ORDER BY 1"
                    ).fetchall()
                )
                raise ValueError(
                    f"Table '{table_name}' not found in DuckDB source. "
                    f"Available tables: {available or '(none)'}"
                )
            return UniversalDataLoader._fetch_arrow_table(
                con.execute(f'SELECT * FROM "{table_name}"')
            )
        finally:
            if owned:
                con.close()

    @staticmethod
    def _fetch_arrow_table(result: Any) -> pa.Table:
        """Normalises duckdb's varying result hand-off into a concrete pa.Table."""
        try:
            return result.fetch_arrow_table()
        except AttributeError:  # duckdb < 1.x exposes .arrow() -> RecordBatchReader
            return result.arrow().read_all()


def duckdb_to_arrow(
    db: Union[str, os.PathLike, duckdb.DuckDBPyConnection],
    table_name: str = "udl_data",
) -> pa.Table:
    """Module-level convenience wrapper for ``UniversalDataLoader.duckdb_to_arrow``."""
    return UniversalDataLoader.duckdb_to_arrow(db, table_name=table_name)