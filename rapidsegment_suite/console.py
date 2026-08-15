"""
Artifact Console - Connected to real RapidSegment outputs
Split-pane: Log Terminal + SQL Inspector + Segments table + Scorecard
"""
from pathlib import Path
from IPython.display import display, HTML
import pandas as pd
import json

from .db import SuiteDB

class ArtifactConsole:
    def __init__(self, db: SuiteDB = None):
        self.db = db or SuiteDB()

    def show(self, exp_id: str, log_level_filter: str = "ALL"):
        rec = self.db.get_experiment(exp_id)
        artifact_path = Path(rec['artifact_path'])

        logs_text = (artifact_path / "logs.txt").read_text() if (artifact_path / "logs.txt").exists() else "No logs"
        sql_text = (artifact_path / "query.sql").read_text() if (artifact_path / "query.sql").exists() else "-- No SQL"
        segments = rec.get('segments', [])
        coverage = rec.get('coverage', [])
        metrics = rec.get('metrics', {})

        if log_level_filter != "ALL":
            logs_text = "\n".join([l for l in logs_text.splitlines() if log_level_filter in l])

        # Pandas views (light data)
        segments_df = pd.DataFrame(segments) if segments else pd.DataFrame()
        coverage_df = pd.DataFrame(coverage) if coverage else pd.DataFrame()
        metrics_df = pd.DataFrame([metrics]) if metrics else pd.DataFrame()

        # Scorecard JSON
        scorecard_path = artifact_path / "scorecard.json"
        scorecard_text = ""
        if scorecard_path.exists():
            try:
                scorecard_json = json.loads(scorecard_path.read_text())
                # Decile thresholds
                deciles = scorecard_json.get('decile_min_thresholds', {})
                weights = scorecard_json.get('segment_weights', {})
                scorecard_text = f"Deciles: {deciles}"
                # weights df
                weights_df = pd.DataFrame.from_dict(weights, orient='index') if weights else pd.DataFrame()
            except Exception as e:
                scorecard_text = f"Error reading scorecard: {e}"
                weights_df = pd.DataFrame()
        else:
            weights_df = pd.DataFrame()

        # HTML split-pane
        html = f"""
        <div style="display:flex; gap:12px; font-family: monospace; height:460px;">
          <div style="flex:1; background:#0e0e0e; color:#d4d4d4; padding:12px; overflow:auto; border-radius:8px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
              <b style="color:#fff;">Log Terminal - {exp_id}</b>
              <span style="font-size:11px; color:#888;">{rec['status']} | {rec['name']}</span>
            </div>
            <pre style="white-space:pre-wrap; font-size:11px; line-height:1.4;">{logs_text[:8000]}</pre>
          </div>
          <div style="flex:1.2; background:#fafafa; border:1px solid #e5e5e5; padding:12px; overflow:auto; border-radius:8px;">
            <b>SQL Inspector (ANSI SQL filters from StrategicSegmentBuilder)</b>
            <pre style="white-space:pre-wrap; font-size:11px; background:white; padding:8px; border-radius:4px; border:1px solid #eee; max-height:180px; overflow:auto;"><code>{sql_text[:6000]}</code></pre>
            <div style="margin-top:12px;">
              <b>Segments (pandas view)</b>
              {segments_df.head(10).to_html(index=False) if not segments_df.empty else '<i>No segments</i>'}
            </div>
            <div style="margin-top:12px;">
              <b>Scorecard Weights</b>
              {weights_df.head(10).to_html() if not weights_df.empty else '<i>No scorecard yet</i>'}
            </div>
            <div style="margin-top:12px; font-size:11px; color:#666;">
              {scorecard_text}
            </div>
          </div>
        </div>
        """

        display(HTML(html))

        print("\n--- Metrics (pandas) ---")
        if not metrics_df.empty:
            display(metrics_df)
        if not segments_df.empty:
            print("\n--- Segments Detail ---")
            display(segments_df.style.bar(subset=['lift'], color='black').hide(axis="index"))
        if not coverage_df.empty:
            print("\n--- Coverage Report (evaluate_final_coverage) ---")
            display(coverage_df.head(20))

        print(f"\nExport Hub: {artifact_path}")
        print(f"  logs.txt: {artifact_path / 'logs.txt'}")
        print(f"  query.sql: {artifact_path / 'query.sql'}")
        print(f"  segments.json: {artifact_path / 'segments.json'}")
        print(f"  scorecard.json: {artifact_path / 'scorecard.json'}")

        return {
            "logs": logs_text,
            "sql": sql_text,
            "segments_df": segments_df,
            "coverage_df": coverage_df,
            "metrics": metrics,
            "paths": str(artifact_path)
        }
