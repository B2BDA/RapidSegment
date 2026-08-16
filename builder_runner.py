"""
Builder Runner - The core connector to rapidsegment library modules
- StrategicSegmentBuilder
- StrategicSegmentScore
- Uses DuckDB for scoring_df creation (as per library quickstart)
- Writes artifacts to filesystem, metadata to DuckDB
"""
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import traceback

from .db import SuiteDB

# Try importing real library
try:
    from rapidsegment import StrategicSegmentBuilder, StrategicSegmentScore
    HAS_RAPIDSEGMENT = True
except ImportError:
    HAS_RAPIDSEGMENT = False
    StrategicSegmentBuilder = None
    StrategicSegmentScore = None


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
    def emit(self, record):
        self.logs.append(self.format(record))


class RapidSegmentRunner:
    def __init__(self, db: SuiteDB = None):
        self.db = db or SuiteDB()

    def _build_builder(self, params: dict):
        """Map suite params -> StrategicSegmentBuilder kwargs"""
        if not HAS_RAPIDSEGMENT:
            raise ImportError("rapidsegment not installed. pip install rapidsegment")

        # params from workbench are flat dict matching builder signature
        allowed = [
            "target", "n_jobs", "min_sample_size", "min_lift", "min_events",
            "top_n_vars", "max_segments", "max_feature_reuse", "enable_diversity",
            "enable_1way", "enable_2way", "enable_3way",
            "feature_groups", "ignore_features", "param_grid",
            "sort_priority", "binning_method", "naive_bins",
            "max_expansion_hops", "selection_metric", "expand_log_mode",
            "db_path", "db_temp_dir"
        ]
        builder_kwargs = {k: v for k, v in params.items() if k in allowed}
        # defaults if not provided
        builder_kwargs.setdefault("target", params.get("target", "target"))
        return StrategicSegmentBuilder(**builder_kwargs)

    def run(self, name: str, params: dict, data: pd.DataFrame, primary_key: str = None, build_scorecard: bool = True) -> str:
        """
        Executes full RapidSegment pipeline:
        1. Create experiment in DuckDB
        2. Extract segments via StrategicSegmentBuilder
        3. Evaluate coverage
        4. Build scorecard via StrategicSegmentScore (if requested)
        5. Save artifacts + update DuckDB
        """
        exp_id = self.db.create_experiment(name=name, params=params, target=params.get("target",""), status="running")
        rec = self.db.get_experiment(exp_id)
        artifact_path = Path(rec['artifact_path'])
        logs_path = artifact_path / "logs.txt"
        sql_path = artifact_path / "query.sql"
        segments_path = artifact_path / "segments.json"
        coverage_path = artifact_path / "coverage.json"
        scorecard_path = artifact_path / "scorecard.json"
        metrics_path = artifact_path / "metrics.json"

        logs = []
        def log(msg, level="INFO"):
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {level}: {msg}"
            logs.append(line)
            print(line)

        try:
            if not HAS_RAPIDSEGMENT:
                # Mock fallback for demo without lib
                log("rapidsegment not installed - running MOCK builder for UI demo", "WARNING")
                import random, hashlib
                seed = int(hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8], 16) % 10000
                random.seed(seed)
                mock_segments = []
                for i in range(params.get("max_segments",3)):
                    mock_segments.append({
                        "segment_id": i+1,
                        "rule_string": f"feature_{i} in bin_{i}",
                        "sql_filter": f"{params.get('target','target')} IS NOT NULL AND feature_{i} > {random.random():.2f}",
                        "count": random.randint(500,5000),
                        "rate": round(random.random()*0.2,4),
                        "lift": round(1.5+random.random()*3,2),
                        "meta_applied_sample_size": params.get("min_sample_size",1000),
                        "meta_applied_min_lift": params.get("min_lift",1.5)
                    })
                segments = mock_segments
                coverage = [{"segment_id": s["segment_id"], "coverage_pct": 10.0} for s in segments]
                scorecard = {"model_metadata": {"total_training_population": len(data)}, "segment_weights": {}, "decile_min_thresholds": {}}
                metrics = {
                    "baseline_rate": data[params.get("target")].mean() if params.get("target") in data.columns else 0.05,
                    "total_coverage_pct": 35.5,
                    "segment_count": len(segments),
                    "active_population_pct": 40.0
                }
            else:
                log(f"Initializing StrategicSegmentBuilder target={params.get('target')}")
                builder = self._build_builder(params)

                # Capture internal logs if builder uses logging
                log_capture = LogCapture()
                log_capture.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logging.getLogger().addHandler(log_capture)

                log("Starting extract_segments...")
                segments = builder.extract_segments(data)
                log(f"Extracted {len(segments)} segments")

                # Coverage
                try:
                    coverage = builder.evaluate_final_coverage(data)
                    log(f"Coverage evaluated: {len(coverage)} rows")
                except Exception as e:
                    log(f"Coverage eval failed: {e}", "WARNING")
                    coverage = []

                # Scorecard
                scorecard = {}
                metrics = {}
                if build_scorecard and segments:
                    log("Building scoring DataFrame via DuckDB")
                    try:
                        import duckdb
                        # Determine primary key
                        pk = primary_key or (data.columns[0] if len(data.columns)>0 else "id")
                        if pk not in data.columns:
                            # create synthetic pk
                            data = data.copy()
                            data["_suite_id"] = [f"ID_{i}" for i in range(len(data))]
                            pk = "_suite_id"

                        scoring_df = data[[pk, params['target']]].copy() if params['target'] in data.columns else data[[pk]].copy()
                        if params['target'] not in scoring_df.columns:
                            scoring_df[params['target']] = 0

                        segment_cols = []
                        for seg in segments:
                            col = f"SEG_{seg['segment_id']}"
                            sql_filter = seg['sql_filter']
                            try:
                                # Use DuckDB to apply filter as per library quickstart
                                matched = duckdb.query(f"SELECT {pk} FROM data WHERE {sql_filter}").df()[pk]
                                scoring_df[col] = scoring_df[pk].isin(matched).astype(int)
                            except Exception as e:
                                log(f"Failed to apply filter {sql_filter}: {e}", "ERROR")
                                scoring_df[col] = 0
                            segment_cols.append(col)

                        log(f"Scoring DF built: {scoring_df.shape}, cols={segment_cols}")

                        if segment_cols:
                            scorer = StrategicSegmentScore(
                                target_col=params['target'],
                                primary_key=pk,
                                segment_cols=segment_cols
                            )
                            log("Calculating scorecard weights...")
                            model = scorer.calculate_and_export_weights(
                                data=scoring_df,
                                export_path=str(scorecard_path)
                            )
                            scorecard = model
                            metrics = model.get("model_metadata", {})
                            # Add aggregated metrics
                            metrics["segment_count"] = len(segments)
                            metrics["max_lift"] = max([s.get("lift",0) for s in segments], default=0)
                            metrics["avg_lift"] = sum([s.get("lift",0) for s in segments])/len(segments) if segments else 0
                            metrics["total_count"] = sum([s.get("count",0) for s in segments])
                            # coverage pct from scorer if available
                            log(f"Scorecard built: {metrics}")
                        else:
                            log("No segment columns for scorecard", "WARNING")

                    except Exception as e:
                        log(f"Scorecard failed: {e}\n{traceback.format_exc()}", "ERROR")
                        scorecard = {"error": str(e)}
                        metrics = {"segment_count": len(segments)}
                else:
                    metrics = {
                        "segment_count": len(segments),
                        "max_lift": max([s.get("lift",0) for s in segments], default=0),
                        "avg_lift": sum([s.get("lift",0) for s in segments])/len(segments) if segments else 0,
                    }

                # Diagnostics
                try:
                    if hasattr(builder, 'explain_no_segments') and len(segments)==0:
                        no_seg_reason = builder.explain_no_segments()
                        log(f"No segments reason: {no_seg_reason}", "WARNING")
                except Exception:
                    pass

            # Save artifacts
            (artifact_path / "logs.txt").write_text("\n".join(logs + (log_capture.logs if 'log_capture' in locals() else [])))
            (artifact_path / "query.sql").write_text("\n\n".join([f"-- Segment {s['segment_id']}\n{s['sql_filter']};" for s in segments]))
            (artifact_path / "segments.json").write_text(json.dumps(segments, indent=2, default=str))
            (artifact_path / "coverage.json").write_text(json.dumps(coverage, indent=2, default=str))
            if 'scorecard' in locals() and scorecard:
                if not scorecard_path.exists():
                    scorecard_path.write_text(json.dumps(scorecard, indent=2, default=str))
            (artifact_path / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
            (artifact_path / "params.json").write_text(json.dumps(params, indent=2, default=str))

            # Update DB
            self.db.update_experiment(exp_id, segments=segments, metrics=metrics, coverage=coverage, status="completed")
            log(f"Experiment {exp_id} completed")

        except Exception as e:
            err = f"Failed: {e}\n{traceback.format_exc()}"
            log(err, "ERROR")
            logs_path.write_text("\n".join(logs))
            self.db.update_experiment(exp_id, metrics={"error": str(e)}, status=f"failed: {e}")
            raise

        return exp_id
