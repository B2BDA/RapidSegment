"""
SuiteDB - DuckDB ONLY for persistence
All reads return pandas DataFrames (light data viewing)
"""
import json
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid

DEFAULT_ROOT = Path(".rapidsegment_suite")
DEFAULT_DB = DEFAULT_ROOT / "suite_data.db"
DEFAULT_ARTIFACTS = DEFAULT_ROOT / "artifacts"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    exp_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    created_at TIMESTAMP,
    status VARCHAR,
    target VARCHAR,
    params_json VARCHAR,
    segments_json VARCHAR,
    metrics_json VARCHAR,
    coverage_json VARCHAR,
    artifact_path VARCHAR
);
"""

class SuiteDB:
    def __init__(self, db_path: Path = DEFAULT_DB, artifacts_root: Path = DEFAULT_ARTIFACTS):
        self.db_path = Path(db_path)
        self.artifacts_root = Path(artifacts_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return duckdb.connect(str(self.db_path))

    def _init_db(self):
        with self._connect() as con:
            con.execute(SCHEMA)

    def create_experiment(self, name: str, params: dict, target: str = "", status: str = "pending") -> str:
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        artifact_path = self.artifacts_root / exp_id
        artifact_path.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                "INSERT INTO experiments (exp_id, name, created_at, status, target, params_json, segments_json, metrics_json, coverage_json, artifact_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [exp_id, name, datetime.now(), status, target, json.dumps(params), json.dumps([]), json.dumps({}), json.dumps([]), str(artifact_path)]
            )
        return exp_id

    def update_experiment(self, exp_id: str, segments: list = None, metrics: dict = None, coverage: list = None, status: str = None):
        with self._connect() as con:
            if segments is not None:
                con.execute("UPDATE experiments SET segments_json = ? WHERE exp_id = ?", [json.dumps(segments, default=str), exp_id])
            if metrics is not None:
                con.execute("UPDATE experiments SET metrics_json = ? WHERE exp_id = ?", [json.dumps(metrics, default=str), exp_id])
            if coverage is not None:
                con.execute("UPDATE experiments SET coverage_json = ? WHERE exp_id = ?", [json.dumps(coverage, default=str), exp_id])
            if status is not None:
                con.execute("UPDATE experiments SET status = ? WHERE exp_id = ?", [status, exp_id])

    def get_experiment(self, exp_id: str) -> dict:
        with self._connect() as con:
            df = con.execute("SELECT * FROM experiments WHERE exp_id = ?", [exp_id]).df()
        if df.empty:
            raise ValueError(f"{exp_id} not found")
        rec = df.iloc[0].to_dict()
        rec['params'] = json.loads(rec['params_json']) if rec['params_json'] else {}
        rec['segments'] = json.loads(rec['segments_json']) if rec['segments_json'] else []
        rec['metrics'] = json.loads(rec['metrics_json']) if rec['metrics_json'] else {}
        rec['coverage'] = json.loads(rec['coverage_json']) if rec['coverage_json'] else []
        return rec

    def list_experiments_df(self) -> pd.DataFrame:
        """Pandas only - light viewing"""
        with self._connect() as con:
            df = con.execute("""
                SELECT exp_id, name, created_at, status, target, metrics_json, segments_json
                FROM experiments ORDER BY created_at DESC
            """).df()
        if df.empty:
            return df

        def _parse_metrics(mj, sj):
            try:
                m = json.loads(mj) if mj else {}
                segs = json.loads(sj) if sj else []
                return pd.Series({
                    'segment_count': len(segs),
                    'max_lift': max([s.get('lift',0) for s in segs], default=0),
                    'avg_lift': sum([s.get('lift',0) for s in segs])/len(segs) if segs else 0,
                    'total_count': sum([s.get('count',0) for s in segs]),
                    'baseline_rate': m.get('baseline_rate', m.get('baseline_event_rate')),
                    'total_coverage_pct': m.get('total_coverage_pct'),
                    'active_population_pct': m.get('active_population_pct'),
                })
            except Exception:
                return pd.Series({'segment_count':0,'max_lift':0,'avg_lift':0,'total_count':0})

        parsed = df.apply(lambda r: _parse_metrics(r['metrics_json'], r['segments_json']), axis=1)
        out = pd.concat([df[['exp_id','name','created_at','status','target']], parsed], axis=1)
        out['created_at'] = pd.to_datetime(out['created_at'])
        return out

    def list_params_for_clone(self) -> dict:
        with self._connect() as con:
            rows = con.execute("SELECT exp_id, name, params_json FROM experiments ORDER BY created_at DESC").fetchall()
        return {r[0]: {"name": r[1], "params": json.loads(r[2])} for r in rows}

    def delete_all(self):
        with self._connect() as con:
            con.execute("DELETE FROM experiments")
