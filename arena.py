"""
Arena - 1v1 Comparison (Connected)
KPI Face-Off + Param Diff + SQL Diff + Segment Overlap
DuckDB JOIN + difflib + pandas
"""
import difflib
import json
import pandas as pd
from pathlib import Path
from IPython.display import display, HTML

from .db import SuiteDB

class Arena:
    def __init__(self, db: SuiteDB = None):
        self.db = db or SuiteDB()

    def _param_diff(self, p1: dict, p2: dict) -> pd.DataFrame:
        rows = []
        def flatten(d, prefix=""):
            out = {}
            for k,v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    out.update(flatten(v, key))
                else:
                    out[key] = v
            return out
        f1 = flatten(p1)
        f2 = flatten(p2)
        for k in sorted(set(f1.keys()) | set(f2.keys())):
            v1 = f1.get(k)
            v2 = f2.get(k)
            if v1 != v2:
                rows.append({"param": k, "A": v1, "B": v2})
        return pd.DataFrame(rows)

    def compare(self, exp_a: str, exp_b: str):
        # DuckDB fetch
        with self.db._connect() as con:
            df = con.execute("SELECT exp_id FROM experiments WHERE exp_id IN (?, ?)", [exp_a, exp_b]).df()
        if len(df) < 2:
            raise ValueError("Both exp_ids must exist")

        rec_a = self.db.get_experiment(exp_a)
        rec_b = self.db.get_experiment(exp_b)

        # KPI face-off from segments
        segs_a = rec_a['segments']
        segs_b = rec_b['segments']
        metrics_a = rec_a['metrics']
        metrics_b = rec_b['metrics']

        kpi_rows = []
        for key in ['segment_count','max_lift','avg_lift','total_count','baseline_rate','total_coverage_pct']:
            va = metrics_a.get(key) if isinstance(metrics_a, dict) else None
            vb = metrics_b.get(key) if isinstance(metrics_b, dict) else None
            # fallback compute from segments
            if va is None and segs_a:
                if key == 'max_lift': va = max([s.get('lift',0) for s in segs_a], default=0)
                if key == 'avg_lift': va = sum([s.get('lift',0) for s in segs_a])/len(segs_a) if segs_a else 0
            if vb is None and segs_b:
                if key == 'max_lift': vb = max([s.get('lift',0) for s in segs_b], default=0)
                if key == 'avg_lift': vb = sum([s.get('lift',0) for s in segs_b])/len(segs_b) if segs_b else 0
            delta = (vb - va) if isinstance(va,(int,float)) and isinstance(vb,(int,float)) else None
            kpi_rows.append({"metric": key, "A": va, "B": vb, "delta (B-A)": delta})

        kpi_df = pd.DataFrame(kpi_rows)

        # Segment-level comparison (pandas)
        seg_a_df = pd.DataFrame(segs_a)[['segment_id','count','rate','lift','sql_filter']] if segs_a else pd.DataFrame()
        seg_b_df = pd.DataFrame(segs_b)[['segment_id','count','rate','lift','sql_filter']] if segs_b else pd.DataFrame()

        param_diff_df = self._param_diff(rec_a['params'], rec_b['params'])

        # SQL diff
        sql_a = "\n".join([s.get('sql_filter','') for s in segs_a])
        sql_b = "\n".join([s.get('sql_filter','') for s in segs_b])
        sql_diff = "\n".join(difflib.unified_diff(sql_a.splitlines(), sql_b.splitlines(), fromfile=f"{exp_a}", tofile=f"{exp_b}", lineterm=''))

        # Render
        html = f"""
        <div style="font-family: Inter, sans-serif; max-width:1100px;">
          <h3>⚔️ Arena: {exp_a} vs {exp_b}</h3>
          <p style="color:#666;">{rec_a['name']} (target={rec_a.get('target')}) vs {rec_b['name']}</p>
          <div style="display:flex; gap:16px;">
            <div style="flex:1; border:2px solid black; border-radius:8px; padding:12px; background:#fafafa;">
              <b>A: {rec_a['name']}</b><br><span style="font-size:12px;">{len(segs_a)} segments | max lift {kpi_df[kpi_df['metric']=='max_lift']['A'].values[0] if not kpi_df.empty else ''}</span>
            </div>
            <div style="flex:1; border:1px solid #ccc; border-radius:8px; padding:12px;">
              <b>B: {rec_b['name']}</b><br><span style="font-size:12px;">{len(segs_b)} segments | max lift {kpi_df[kpi_df['metric']=='max_lift']['B'].values[0] if not kpi_df.empty else ''}</span>
            </div>
          </div>
        </div>
        """
        display(HTML(html))

        print("KPI Face-Off:")
        display(kpi_df.style.bar(subset=['delta (B-A)'], color='black').hide(axis="index"))

        print("\nParameter Diff (only differing):")
        if param_diff_df.empty:
            print("No param diff - identical builder configs")
        else:
            display(param_diff_df.style.hide(axis="index"))

        print("\nSegment A:")
        display(seg_a_df.head(10).style.bar(subset=['lift'], color='black').hide(axis="index") if not seg_a_df.empty else seg_a_df)
        print("\nSegment B:")
        display(seg_b_df.head(10).style.bar(subset=['lift'], color='black').hide(axis="index") if not seg_b_df.empty else seg_b_df)

        print("\nSQL Diff (GitHub-style):")
        diff_html = f"<pre style='background:#fafafa; border:1px solid #eee; padding:12px; font-size:11px; max-height:300px; overflow:auto;'>{sql_diff or 'No diff'}</pre>"
        display(HTML(diff_html))

        return {
            "kpi": kpi_df,
            "param_diff": param_diff_df,
            "seg_a": seg_a_df,
            "seg_b": seg_b_df,
            "sql_diff": sql_diff
        }
