"""
Data Loader wrapper - connects to UniversalDataLoader from rapidsegment
Falls back to pandas if library not available
Uses DuckDB for fast preview (pandas view only)
"""
import pandas as pd
from pathlib import Path

class SuiteDataLoader:
    def __init__(self):
        self._loader = None
        try:
            from rapidsegment import UniversalDataLoader
            self._loader_class = UniversalDataLoader
            self.has_rapidsegment_loader = True
        except ImportError:
            self._loader_class = None
            self.has_rapidsegment_loader = False

    def load(self, file_path: str = None, df: pd.DataFrame = None, project_id=None, dataset_id=None, table_id=None) -> pd.DataFrame:
        """
        Priority: df > file_path > BigQuery via UniversalDataLoader
        Returns pandas DataFrame for suite (light viewing via pandas)
        """
        if df is not None:
            print(f"Using provided DataFrame: {df.shape}")
            return df.copy()

        if file_path:
            p = Path(file_path)
            if p.suffix.lower() == '.csv':
                return pd.read_csv(file_path)
            elif p.suffix.lower() in ['.parquet', '.pq']:
                return pd.read_parquet(file_path)
            elif p.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                # Try UniversalDataLoader for generic
                if self.has_rapidsegment_loader:
                    loader = self._loader_class(file_path=file_path)
                    # UniversalDataLoader returns PyArrow Table - convert
                    try:
                        tbl = loader.load() if hasattr(loader, 'load') else None
                        if tbl is not None:
                            return tbl.to_pandas()
                    except Exception as e:
                        print(f"UniversalDataLoader failed, falling back to pandas: {e}")
                return pd.read_csv(file_path)

        if project_id and dataset_id and table_id and self.has_rapidsegment_loader:
            loader = self._loader_class(project_id=project_id, dataset_id=dataset_id, table_id=table_id)
            tbl = loader.load()
            return tbl.to_pandas()

        raise ValueError("Provide df or file_path (csv/parquet/excel) or BQ params")

    def preview(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Pandas light preview"""
        # DuckDB can be used for profiling but view via pandas
        return df.head(n)

    def profile(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Quick profile using pandas + duckdb for counts"""
        try:
            import duckdb
            # DuckDB in-memory for fast stats
            stats = duckdb.query(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN {target}=1 THEN 1 ELSE 0 END) as events,
                    AVG(CASE WHEN {target}=1 THEN 1 ELSE 0 END) as baseline_rate
                FROM df
            """).df()
            return stats
        except Exception:
            total = len(df)
            events = df[target].sum() if target in df.columns else 0
            return pd.DataFrame([{"total": total, "events": events, "baseline_rate": events/total if total else 0}])
