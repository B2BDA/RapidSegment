"""
Leaderboard - History & Performance (DuckDB + Pandas)
Ranked Data Grid with black bars
"""
import pandas as pd
from IPython.display import display
from .db import SuiteDB

class Leaderboard:
    def __init__(self, db: SuiteDB = None):
        self.db = db or SuiteDB()

    def get_df(self, sort_by: str = "max_lift", ascending: bool = False) -> pd.DataFrame:
        df = self.db.list_experiments_df()
        if df.empty:
            return df
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending, na_position_last=True)
        return df

    def show(self, sort_by: str = "max_lift", top_n: int = None, filter_status: str = "completed") -> pd.DataFrame:
        df = self.get_df(sort_by=sort_by)
        if filter_status:
            df = df[df['status'] == filter_status]
        if top_n:
            df = df.head(top_n)

        if df.empty:
            print("No experiments yet. Run wb.run() first.")
            return df

        # Pandas styling - solid black as spec
        styled = df.style\
            .bar(subset=['max_lift'], color='black', vmin=0)\
            .bar(subset=['avg_lift'], color='#111111')\
            .bar(subset=['total_count'], color='#222222')\
            .bar(subset=['segment_count'], color='#333333')\
            .format({
                'max_lift': '{:.2f}x',
                'avg_lift': '{:.2f}x',
                'total_count': '{:,.0f}',
                'baseline_rate': '{:.2%}',
                'total_coverage_pct': '{:.1f}%'
            }, na_rep="-")\
            .set_caption(f"Leaderboard - Ranked by {sort_by} (North Star: Lift)")\
            .hide(axis="index")

        display(styled)
        print("\nRow actions: wb.clone_from_history(exp_id) | arena.compare(exp_a, exp_b) | console.show(exp_id)")
        return df

    def compare_top_two(self):
        df = self.get_df()
        if len(df) < 2:
            print("Need 2 experiments")
            return None
        from .arena import Arena
        arena = Arena(db=self.db)
        a = df.iloc[0]['exp_id']
        b = df.iloc[1]['exp_id']
        return arena.compare(a,b)
