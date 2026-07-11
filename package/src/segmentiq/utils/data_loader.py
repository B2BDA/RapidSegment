"""
Unified Data Ingestion Layer for Strategic Analytics
===================================================
A multi-format data loader abstraction supporting Local Files (CSV, Parquet, Arrow, Excel),
In-Memory PyArrow Tables, and Google Cloud BigQuery Storage API streams.
"""

import os
import logging
from typing import Any, Optional, Union, List
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_pq

logger = logging.getLogger("StrategicEngine.DataLoader")


class UniversalDataLoader:
    """Handles multi-source data ingestion, normalizing inputs into highly optimized
    in-memory structures compatible with vectorized down-stream compute engines.
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
        """Auto-detects the source configuration parameters and loads the dataset.

        Returns:
            An ingestion asset (pa.Table or DuckDB macro string) ready for downstream consumption.
        """
        # Scenario 1: Direct In-Memory Object Check
        if fallback_data is not None:
            if isinstance(fallback_data, pa.Table):
                logger.info("Ingesting directly provided in-memory PyArrow Table.")
                return fallback_data
            return fallback_data

        # Scenario 2: BigQuery Parameters Detected
        if self.dataset_id and self.table_id:
            return self._load_from_bigquery()

        # Scenario 3: Local File Path Detection
        if self.file_path:
            return self._load_from_file()

        raise ValueError(
            "Invalid Configuration: You must provide either a valid `file_path`, "
            "BigQuery identifiers, or pass an explicit `fallback_data` object."
        )

    def _load_from_file(self) -> pa.Table:
        """Parses local files using high-performance C++ Arrow readers."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at: {self.file_path}")

        ext = os.path.splitext(self.file_path)[-1].lower()
        
        if ext == ".parquet":
            return pa_pq.read_table(self.file_path)
        elif ext == ".csv":
            return pa_csv.read_csv(self.file_path)
        elif ext in [".arrow", ".feather"]:
            with pa.memory_map(self.file_path, "r") as source:
                return pa.ipc.open_file(source).read_all()
        elif ext in [".xlsx", ".xls"]:
            return self._load_excel_to_arrow()
        
        raise ValueError(f"Unsupported format: '{ext}'.")

    def _load_excel_to_arrow(self) -> pa.Table:
        """Parses Excel using positional tracking to prevent column misalignment."""
        logger.info("Parsing Excel spreadsheet via positional column tracking...")
        try:
            import openpyxl
            wb = openpyxl.load_workbook(self.file_path, data_only=True, read_only=True)
            sheet = wb.active
            rows = sheet.iter_rows(values_only=True)
            
            headers = next(rows)
            if not headers:
                raise ValueError("The Excel file appears to be empty.")
            
            # Fix: Track column names by index to prevent header-shift corruption
            column_names = [f"{h}" if h else f"_col_{i}" for i, h in enumerate(headers)]
            data_columns = {name: [] for name in column_names}

            for row in rows:
                for i, name in enumerate(column_names):
                    val = row[i] if i < len(row) else None
                    data_columns[name].append(val)
            
            wb.close()
            return pa.Table.from_pydict(data_columns)
        except ImportError:
            raise ImportError("Dependency missing: `pip install openpyxl` required.")

    def _load_from_bigquery(self) -> Union[pa.Table, str]:
        """Resolves BigQuery ingestion using cost-optimized metadata inspection."""
        try:
            from google.cloud import bigquery
            bq_client = bigquery.Client(project=self.project_id)
            full_table_ref = f"{self.project_id or bq_client.project}.{self.dataset_id}.{self.table_id}"
            
            # Optimization: Use get_table() instead of INFORMATION_SCHEMA to avoid slot costs
            table = bq_client.get_table(f"{self.project_id or bq_client.project}.{self.dataset_id}.{self.table_id}")
            
            select_clauses = []
            for field in table.schema:
                # Type normalization: Casting numeric types to FLOAT64 for compatibility
                if field.field_type in ("NUMERIC", "BIGNUMERIC", "DECIMAL"):
                    select_clauses.append(f"SAFE_CAST(`{field.name}` AS FLOAT64) AS `{field.name}`")
                else:
                    select_clauses.append(f"`{field.name}`")
            
            query = f"SELECT {', '.join(select_clauses)} FROM `{full_table_ref}`"
            return bq_client.query(query).to_arrow()

        except ImportError:
            # Fallback for environments without the GCP library installed
            if not self.dataset_id or not self.table_id:
                raise ValueError("BigQuery coordinates (dataset_id, table_id) required for scan.")
            
            logger.warning("google-cloud-bigquery not found. Returning native DuckDB scan macro.")
            return f"bigquery_scan('{self.project_id or 'default'}', '{self.dataset_id}', '{self.table_id}')"