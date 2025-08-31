#!/usr/bin/env python
"""
train_models.py (final alignment fix)
- Compute y AFTER sorting so X and y share the exact same index
- Remove y = y.loc[df.index]
- Keep earlier robustness (TEAM/OPP derivation, GAME fallback, calibration fallback)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List
import joblib
import lightgbm as lgb

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

    scol = find_col(df, ["season","year"])
    wcol = find_col(df, ["week","game_week"])
    if scol is None or wcol is None: raise ValueError("Missing season/week columns")
    df["SEASON"] = pd.to_numeric(df[scol], errors="coerce").astype(int)
    df["WEEK"]   = pd.to_numeric(df[wcol], errors="coerce").astype(int)
    return df

def attach_usage(df: pd.DataFrame, usage_path: Path) -> pd.DataFrame:
    if not usage_path.exists(): return df
    use = pd.read_parquet(usage_path)
    key_map = {}
    for k in ["PLAYER","player","player_name","full_name"]:
        if k in use.columns: key_map["PLAYER"] = k; break
    for k in ["TEAM","team","recent_team","posteam"]:
        if k in use.columns: key_map["TEAM"] = k; break
    for k in ["SEASON","season","year"]:
        if k in use.columns: key_map["SEASON"] = k; break
    for k in ["WEEK","week","game_week"]:
        if k in use.columns: key_map["WEEK"] = k; break
    right = use.rename(columns={v:k for k,v in key_map.items()})
    on = [c for c in ["PLAYER","TEAM","SEASON","WEEK"] if c in right.columns]
    return df.merge(right, on=on, how="left") if on else df

def attach_def(df: pd.DataFrame, def_path: Path) -> pd.DataFrame:
    if not def_path.exists(): return df
    d = pd.read_parquet(def_path)
    cand = [c for c in d.columns if c.upper() in ("OPP","TEAM","DEFTEAM","DEF_TEAM","OPPONENT_TEAM")]
    if not cand: return df
    oppc = cand[0]
    d["_OPP"] = d[oppc].astype(str).str.upper()
    left = df.copy(); left["_OPP"] = left["OPP"]
    return left.merge(d.drop(columns=[oppc]).rename(columns={"_OPP":"OPP"}), left_on="_OPP", right_on="OPP", how="left").drop(columns=["_OPP"])

def attach_team_context(df: pd.DataFrame, ctx_path: Path) -> pd.DataFrame:
    if not ctx_path.exists(): return df
    c = pd.read_parquet(ctx_path)
    if "TEAM" not in c.columns and "team" in c.columns:
        c = c.rename(columns={"team":"TEAM"})
    if "TEAM" not in c.columns: return df
    c["TEAM"] = c["TEAM"].astype(str).str.upper()
    return df.merge(c, on=[col for col in ["TEAM","SEASON","WEEK"] if col in c.columns], how="left")

def build_features(base: pd.DataFrame):
    # Return X, y, features, and the sorted frame used to create them (for aligned masks)
    df = base.copy()
    # Ensure TEAM/OPP exist even if upstream missed them
    if "TEAM" not in df.columns: df["TEAM"] = "UNK"
    if "OPP"  not in df.columns: df["OPP"]  = "UNK"

    # Sort first so everything that follows uses this exact index
    df = df.sort_values(["PLAYER","SEASON","WEEK"]).reset_index(drop=True)

    # Rolling recency features
    for col in ["targets","receptions","receiving_yards","rushing_yards","rushing_attempts","passing_yards","passing_tds"]:
        if col in df.columns:
            r = df.groupby("PLAYER")[col].shift(1)
            df[f"{col}_r3"] = r.rolling(3, min_periods=1).mean()

    # One-hot
    for pos in POS_LIST:
        df[f"is_{pos}"] = (df["POS"] == pos).astype(int)
    team_dummies = pd.get_dummies(df["TEAM"], prefix="tm", dtype=int)
    opp_dummies  = pd.get_dummies(df["OPP"],  prefix="opp", dtype=int)
    df = pd.concat([df, team_dummies, opp_dummies], axis=1)

    # Feature set
    feat_cols = []
    candidates = [
        "target_share","rush_share","route_rate",
        "targets_r3","receptions_r3","receiving_yards_r3",
        "rushing_yards_r3","rushing_attempts_r3","passing_yards_r3","passing_tds_r3",
        "pass_def_mult","rush_def_mult",
        "ou","total","game_total","spread","line","proe","pace",
        *[c for c in df.columns if c.startswith("tm_") or c.startswith("opp_")],
        *[c for c in df.columns if c.startswith("is_")],
    ]
    for c in candidates:
        if c in df.columns: feat_cols.append(c)

    X = df[feat_cols].fillna(0.0)
    y = df.apply(dk_points_from_weekly, axis=1)  # compute AFTER sorting
    return X, y, feat_cols, df

def compute_cqr_delta(y_true, q_low, q_high, alpha: float = 0.2) -> float:
    if len(y_true) == 0:
        return 0.0
    s = np.maximum(q_low - y_true, y_true - q_high)
    s = np.sort(s); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(s[k])

def train_quantile_models_for_pos(df_pos: pd.DataFrame, pos: str, out_dir: Path) -> None:
    X, y, feat_cols, df_sorted = build_features(df_pos)

    # Time-based splits computed on the SAME index as X,y
    idx_train = (df_sorted["SEASON"] <= 2023)
    if (df_sorted["SEASON"] == 2024).any():
        weeks_2024 = sorted(df_sorted.loc[df_sorted["SEASON"] == 2024, "WEEK"].unique().tolist())
        cutoff = weeks_2024[len(weeks_2024)//2] if len(weeks_2024) >= 6 else weeks_2024[-1]
        idx_cal = (df_sorted["SEASON"] == 2024) & (df_sorted["WEEK"] <= cutoff)
        idx_test = (df_sorted["SEASON"] == 2024) & (df_sorted["WEEK"] > cutoff)
    else:
        weeks_2023 = sorted(df_sorted.loc[df_sorted["SEASON"] == 2023, "WEEK"].unique().tolist())
        last_weeks = weeks_2023[-6:] if len(weeks_2023) >= 6 else weeks_2023
        idx_cal = (df_sorted["SEASON"] == 2023) & (df_sorted["WEEK"].isin(last_weeks))
        idx_test = (df_sorted["SEASON"] == 2023) & (~df_sorted["WEEK"].isin(last_weeks))

    X_train, y_train = X[idx_train], y[idx_train]
    X_cal, y_cal     = X[idx_cal],   y[idx_cal]

    base_params = dict(
        boosting_type="gbdt", learning_rate=0.05, num_leaves=64, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, max_depth=-1,
        n_estimators=1200, verbose=-1, force_row_wise=True,
    )

    models = {}
    for q, alpha in [("p10", 0.10), ("p50", 0.50), ("p90", 0.90)]:
        params = base_params | {"objective": "quantile", "alpha": alpha}
        model = lgb.LGBMRegressor(**params)
        eval_set = [(X_cal, y_cal)] if len(X_cal) else []
        model.fit(X_train, y_train, eval_set=eval_set if len(eval_set) else None, eval_metric="l1", callbacks=[lgb.log_evaluation(period=0)])
        models[q] = model

    if len(X_cal):
        ql_cal = models["p10"].predict(X_cal)
        qh_cal = models["p90"].predict(X_cal)
        delta80 = compute_cqr_delta(y_cal.values, ql_cal, qh_cal, alpha=0.20)
    else:
        ql_tr = models["p10"].predict(X_train)
        qh_tr = models["p90"].predict(X_train)
        delta80 = compute_cqr_delta(y_train.values, ql_tr, qh_tr, alpha=0.20)

    bundle = {"models": models, "calib": {"delta80": float(delta80)}, "feature_names": list(X.columns)}
    out_path = out_dir / f"lgbm_{pos}.pkl"
    joblib.dump(bundle, out_path)
    print(f"[{pos}] Saved {out_path.name}  |  delta80={delta80:.3f}  train={int(idx_train.sum())} cal={int(idx_cal.sum())}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    root = Path(args.data_root)
    data_dir = root / "data_2022_2024"
    out_dir = Path(args.out_dir) if args.out_dir else (root / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    weekly_path = data_dir / "weekly.parquet"
    if not weekly_path.exists():
        raise SystemExit(f"Missing {weekly_path}. Export or copy your weekly parquet first.")
    weekly = pd.read_parquet(weekly_path)
    print("[info] weekly columns:", list(weekly.columns))
    weekly = normalize_core(weekly)
    print("[info] normalized columns:", list(weekly.columns))

    usage_path = data_dir / "player_usage.parquet"
    ctx_path   = data_dir / "team_context.parquet"
    def_path   = data_dir / "defense_adjust.parquet"
    if usage_path.exists(): weekly = attach_usage(weekly, usage_path)
    if ctx_path.exists():   weekly = attach_team_context(weekly, ctx_path)
    if def_path.exists():   weekly = attach_def(weekly, def_path)

    weekly = weekly[weekly["POS"].isin(POS_LIST)].reset_index(drop=True)
    for pos in POS_LIST:
        df_pos = weekly[weekly["POS"] == pos].copy()
        if len(df_pos) < 500:
            print(f"[{pos}] Not enough rows to train ({len(df_pos)}). Skipping.")
            continue
        train_quantile_models_for_pos(df_pos, pos, out_dir)
    print("Done. Models are in:", out_dir)

if __name__ == "__main__":
    main()
