#!/usr/bin/env python
"""
evaluate_accuracy.py (hardened)
- Robust TEAM/OPP derivation and safe fallback to 'UNK'
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import List

POS_LIST = ["QB","RB","WR","TE"]

def dk_points_from_weekly(row: pd.Series) -> float:
    pass_yds = row.get("passing_yards", 0.0) or 0.0
    pass_td  = row.get("passing_tds", 0.0) or 0.0
    ints     = row.get("interceptions", 0.0) or 0.0
    rush_yds = row.get("rushing_yards", 0.0) or 0.0
    rush_td  = row.get("rushing_tds", 0.0) or 0.0
    rec      = row.get("receptions", 0.0) or 0.0
    rec_yds  = row.get("receiving_yards", 0.0) or 0.0
    rec_td   = row.get("receiving_tds", 0.0) or 0.0
    fum_lost = row.get("fumbles_lost", 0.0) or 0.0
    pts = 0.0
    pts += pass_yds * 0.04; pts += pass_td * 4.0; pts += ints * -1.0
    pts += rush_yds * 0.1;  pts += rush_td * 6.0
    pts += rec * 1.0;       pts += rec_yds * 0.1; pts += rec_td * 6.0
    pts += fum_lost * -1.0
    if pass_yds >= 300: pts += 3.0
    if rush_yds >= 100: pts += 3.0
    if rec_yds  >= 100: pts += 3.0
    return float(pts)

def find_col(df: pd.DataFrame, aliases: List[str]):
    norm = {c.lower().replace(" ", "").replace("_",""): c for c in df.columns}
    for a in aliases:
        key = a.lower().replace(" ", "").replace("_","")
        if key in norm: return norm[key]
    return None

def _safe_upper(x) -> str:
    try: return str(x).upper().strip()
    except: return ""

def _parse_game_to_teams(df: pd.DataFrame, gcol: str):
    tmp = df[gcol].astype(str).str.replace("@","-")
    parts = tmp.str.split("-", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("GAME/MATCHUP column could not be parsed into TEAM/OPP (expected like NE@BUF).")
    team = parts[0].map(_safe_upper)
    opp  = parts[1].map(_safe_upper)
    return team, opp

def _ensure_team_opp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TEAM" not in df.columns:
        tcol = find_col(df, ["team","recent_team","posteam","team_abbr","team_code"])
        if tcol is not None:
            df["TEAM"] = df[tcol].map(_safe_upper)
    if "OPP" not in df.columns:
        ocol = find_col(df, ["opponent","opp","defteam","opponent_team","opp_team","opponent_abbr","opponent_code"])
        if ocol is not None:
            df["OPP"] = df[ocol].map(_safe_upper)
    if ("TEAM" not in df.columns) or ("OPP" not in df.columns):
        gcol = find_col(df, ["game","matchup"])
        if gcol is not None:
            team, opp = _parse_game_to_teams(df, gcol)
            if "TEAM" not in df.columns: df["TEAM"] = team
            if "OPP"  not in df.columns: df["OPP"]  = opp
    if "TEAM" not in df.columns:
        print("[warn] TEAM column not found anywhere — setting TEAM='UNK'")
        df["TEAM"] = "UNK"
    if "OPP" not in df.columns:
        print("[warn] OPP column not found anywhere — setting OPP='UNK'")
        df["OPP"] = "UNK"
    return df

def normalize_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pos_col = find_col(df, ["position","pos","player_position"])
    if pos_col is None: raise ValueError("Missing position column")
    df["POS"] = df[pos_col].astype(str).str.upper().replace({"D":"DST"})
    name_col = find_col(df, ["player","player_name","full_name"]) or "player"
    df["PLAYER"] = df[name_col].astype(str)
    df = _ensure_team_opp(df)
    scol = find_col(df, ["season","year"]); wcol = find_col(df, ["week","game_week"])
    if scol is None or wcol is None: raise ValueError("Missing season/week columns")
    df["SEASON"] = pd.to_numeric(df[scol], errors="coerce").astype(int)
    df["WEEK"]   = pd.to_numeric(df[wcol], errors="coerce").astype(int)
    return df

def attach_optional(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    usage_path = data_dir / "player_usage.parquet"
    ctx_path   = data_dir / "team_context.parquet"
    def_path   = data_dir / "defense_adjust.parquet"
    def find_and_merge(dfL: pd.DataFrame, path: Path) -> pd.DataFrame:
        if not path.exists(): return dfL
        t = pd.read_parquet(path)
        key_map = {}
        for k in ["PLAYER","player","player_name","full_name"]:
            if k in t.columns: key_map["PLAYER"] = k; break
        for k in ["TEAM","team","recent_team","posteam"]:
            if k in t.columns: key_map["TEAM"] = k; break
        for k in ["SEASON","season","year"]:
            if k in t.columns: key_map["SEASON"] = k; break
        for k in ["WEEK","week","game_week"]:
            if k in t.columns: key_map["WEEK"] = k; break
        t2 = t.rename(columns={v:k for k,v in key_map.items()})
        return dfL.merge(t2, on=[c for c in ["PLAYER","TEAM","SEASON","WEEK"] if c in t2.columns], how="left")
    df = find_and_merge(df, usage_path)
    df = find_and_merge(df, ctx_path)
    if def_path.exists():
        d = pd.read_parquet(def_path)
        cand = [c for c in d.columns if c.upper() in ("OPP","TEAM","DEFTEAM","DEF_TEAM","OPPONENT_TEAM")]
        if cand:
            oppc = cand[0]
            d["_OPP"] = d[oppc].astype(str).str.upper()
            left = df.copy(); left["_OPP"] = left["OPP"]
            df = left.merge(d.drop(columns=[oppc]).rename(columns={"_OPP":"OPP"}), left_on="_OPP", right_on="OPP", how="left").drop(columns=["_OPP"])
    return df

def crps_normal(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    phi = np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
    crps = sigma * (z * (2*Phi - 1) + 2*phi - 1/np.sqrt(np.pi))
    return crps

def spearman_corr(a, b):
    ar = pd.Series(a).rank(method="average").values
    br = pd.Series(b).rank(method="average").values
    return float(np.corrcoef(ar, b)[0,1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--models-dir", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.data_root)
    data_dir = root / "data_2022_2024"
    models_dir = Path(args.models_dir)
    out_csv = models_dir / "predictions_eval_2022_2024.csv"

    weekly_path = data_dir / "weekly.parquet"
    if not weekly_path.exists():
        raise SystemExit(f"Missing {weekly_path}")

    weekly = pd.read_parquet(weekly_path)
    print("[info] weekly columns:", list(weekly.columns))
    weekly = normalize_core(weekly)
    print("[info] normalized columns:", list(weekly.columns))
    weekly = attach_optional(weekly, data_dir)
    weekly = weekly[weekly["POS"].isin(POS_LIST)].reset_index(drop=True)

    rows = []
    for pos in POS_LIST:
        bundle_path = models_dir / f"lgbm_{pos}.pkl"
        if not bundle_path.exists():
            print(f"[{pos}] Missing model bundle at {bundle_path}. Skipping.")
            continue
        bundle = joblib.load(bundle_path)
        models = bundle["models"]; feat_names = bundle["feature_names"]
        delta80 = bundle["calib"]["delta80"]
        dfp = weekly[weekly["POS"] == pos].copy()

        dfp = dfp.sort_values(["PLAYER","SEASON","WEEK"]).reset_index(drop=True)
        for col in ["targets","receptions","receiving_yards","rushing_yards","rushing_attempts","passing_yards","passing_tds"]:
            if col in dfp.columns and f"{col}_r3" not in dfp.columns:
                r = dfp.groupby("PLAYER")[col].shift(1)
                dfp[f"{col}_r3"] = r.rolling(3, min_periods=1).mean()

        # Ensure TEAM/OPP exist
        if "TEAM" not in dfp.columns: dfp["TEAM"] = "UNK"
        if "OPP"  not in dfp.columns: dfp["OPP"]  = "UNK"

        X = dfp.reindex(columns=feat_names, fill_value=0.0).astype(float)
        y_true = dfp.apply(dk_points_from_weekly, axis=1).values

        p10 = models["p10"].predict(X); p50 = models["p50"].predict(X); p90 = models["p90"].predict(X)
        lo80 = p50 - delta80; hi80 = p50 + delta80

        mae = float(np.mean(np.abs(p50 - y_true)))
        cov80 = float(np.mean((y_true >= lo80) & (y_true <= hi80)))
        left_tail = float(np.mean(y_true < p10)); right_tail = float(np.mean(y_true > p90))
        sigma = (p90 - p50) / 1.2816; crps = float(np.mean(crps_normal(y_true, p50, sigma)))
        rows.append({"POS": pos, "N": len(y_true), "MAE": mae, "Cov80": cov80, "Left< p10": left_tail, "Right> p90": right_tail, "CRPS": crps})

    if rows:
        dfm = pd.DataFrame(rows); print("\n=== Summary ==="); print(dfm.to_string(index=False))
        print(f"\nPredictions saved to: {out_csv}")

if __name__ == "__main__":
    main()
