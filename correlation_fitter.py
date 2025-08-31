# correlation_fitter.py
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

def _pick_or_none(df, pos, k=1):
    d = df[df["POS"]==pos].sort_values("FPTS", ascending=False)
    vals = d["FPTS"].values[:k]
    return list(vals) + [np.nan]*(k-len(vals))

def _corr(a: pd.Series, b: pd.Series):
    m = (~a.isna()) & (~b.isna())
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0,1]), int(m.sum())

def _shrink(r, n, k=50):
    if np.isnan(r) or n is None: return np.nan
    w = n / (n + k)
    return float(np.clip(r * w, -0.95, 0.95))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Folder containing data_2022_2024/weekly.parquet or weekly.csv")
    ap.add_argument("--out", default="correlation_table.json", help="Output json path")
    args = ap.parse_args()

    root = Path(args.data_root)
    weekly_pq = root / "data_2022_2024" / "weekly.parquet"
    weekly_csv = root / "data_2022_2024" / "weekly.csv"

    if weekly_pq.exists():
        df = pd.read_parquet(weekly_pq)
    elif weekly_csv.exists():
        df = pd.read_csv(weekly_csv)
    else:
        raise FileNotFoundError("Could not find weekly.parquet or weekly.csv under data_2022_2024")

    # Normalize columns
    df = df.copy()
    def ucget(*names):
        for c in names:
            if c in df.columns: return c
        return None
    POS = ucget("POS","position")
    TEAM = ucget("TEAM","recent_team","team")
    OPP = ucget("OPP","opponent_team","Opp")
    PTS = ucget("fantasy_points_ppr","fantasy_points")
    SEASON = ucget("SEASON","season")
    WEEK = ucget("WEEK","week")

    if any(c is None for c in (POS, TEAM, OPP, PTS, SEASON, WEEK)):
        raise ValueError("Missing required columns after normalization.")

    df["POS"] = df[POS].astype(str).str.upper().str.strip()
    df["TEAM"] = df[TEAM].astype(str).str.upper().str.strip()
    df["OPP"]  = df[OPP].astype(str).str.upper().str.strip()
    df["FPTS"] = pd.to_numeric(df[PTS], errors="coerce")
    df["SEASON"] = pd.to_numeric(df[SEASON], errors="coerce").astype("Int64")
    df["WEEK"]   = pd.to_numeric(df[WEEK], errors="coerce").astype("Int64")

    df = df[df["POS"].isin(["QB","RB","WR","TE"])].copy()

    # Build per-game team table of top roles
    rows = []
    for (season, week, team, opp), g in df.groupby(["SEASON","WEEK","TEAM","OPP"]):
        qb1 = _pick_or_none(g, "QB", 1)[0]
        rb1 = _pick_or_none(g, "RB", 1)[0]
        wr1, wr2 = _pick_or_none(g, "WR", 2)
        te1 = _pick_or_none(g, "TE", 1)[0]
        rows.append({
            "SEASON": season, "WEEK": week, "TEAM": team, "OPP": opp,
            "QB1": qb1, "RB1": rb1, "WR1": wr1, "WR2": wr2, "TE1": te1
        })
    team_df = pd.DataFrame(rows)

    # Same-team correlations (across games)
    def series(name): return team_df[name]
    pairs_same = {
        "QB-WR":  _corr(series("QB1"), series("WR1")),
        "QB-TE":  _corr(series("QB1"), series("TE1")),
        "WR-WR":  _corr(series("WR1"), series("WR2")),
        "RB-QB":  _corr(series("RB1"), series("QB1")),
        "RB-WR":  _corr(series("RB1"), series("WR1")),
        "RB-TE":  _corr(series("RB1"), series("TE1")),
    }
    same_team = { k: _shrink(r, n) for k,(r,n) in pairs_same.items() }

    # Bring-back (opponents): merge A vs B with B vs A
    opp_df = team_df.merge(
        team_df,
        left_on=["SEASON","WEEK","TEAM","OPP"],
        right_on=["SEASON","WEEK","OPP","TEAM"],
        suffixes=("_A","_B")
    )
    def oc(nameA, nameB): return _corr(opp_df[nameA], opp_df[nameB])

    pairs_opp = {
        "QB-WR":  oc("QB1_A","WR1_B"),
        "WR-WR":  oc("WR1_A","WR1_B"),
        "QB-QB":  oc("QB1_A","QB1_B"),
        "RB-PASS": oc("RB1_A","WR1_B"),  # often mildly negative
        "GENERIC": oc("WR1_A","TE1_B"),  # fallback
    }
    bring_back = { k: _shrink(r, n) for k,(r,n) in pairs_opp.items() }

    table = {"same_team": same_team, "bring_back": bring_back}

    out = Path(args.out)
    out.write_text(json.dumps(table, indent=2))
    print(f"[ok] wrote {out.resolve()}")
    print(json.dumps(table, indent=2))

if __name__ == "__main__":
    main()
