# update_weekly_data.py
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

DATA_ROOT = Path(r"C:\Users\stuff\nfl_export")
OUT_DIR   = DATA_ROOT / "data_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE  = OUT_DIR / "weekly.parquet"

SEASONS = list(range(2022, 2026))  # auto-includes 2025

df = nfl.import_weekly_data(SEASONS, downcast=True)

df = df.rename(columns={
    "player_display_name": "PLAYER",
    "position": "POS",
    "recent_team": "TEAM",
    "opponent_team": "OPP",
    "season": "SEASON",
    "week": "WEEK",
})

if OUT_FILE.exists():
    old = pd.read_parquet(OUT_FILE)
    df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["PLAYER","POS","TEAM","OPP","SEASON","WEEK"], keep="last")

df.to_parquet(OUT_FILE, index=False)
print(f"[ok] wrote {OUT_FILE}  |  rows={len(df):,}")
