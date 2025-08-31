#!/usr/bin/env python
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import joblib

POS_LIST = ["QB","RB","WR","TE"]

def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    norm = {c.lower().replace(" ", "").replace("_",""): c for c in df.columns}
    for a in aliases:
        k = a.lower().replace(" ", "").replace("_","")
        if k in norm: return norm[k]
    return None

def _safe_upper(s):
    try: return str(s).upper().strip()
    except: return ""

def _normalize_core(upload_df: pd.DataFrame) -> pd.DataFrame:
    df = upload_df.copy()
    pos_col = _find_col(df, ["pos","position","player_position"])
    if pos_col is None:
        raise ValueError("Could not locate a POS/position column in the uploaded file.")
    df["POS"] = df[pos_col].astype(str).str.upper().replace({"D":"DST"})

    name_col = _find_col(df, ["player","player_name","full_name","name"])
    if name_col is None: name_col = pos_col
    df["PLAYER"] = df[name_col].astype(str)

    tcol = _find_col(df, ["team","recent_team","posteam"])
    ocol = _find_col(df, ["opp","opponent","defteam"])
    if tcol is None or ocol is None:
        gcol = _find_col(df, ["game","matchup"])
        if gcol is None:
            raise ValueError("Need TEAM/OPP or a GAME column in the uploaded file.")
        teams = df[gcol].astype(str).str.replace("@","-").str.split("-", n=1, expand=True)
        if teams.shape[1] != 2:
            raise ValueError("GAME column could not be parsed into TEAM/OPP.")
        df["TEAM"] = teams[0].map(_safe_upper)
        df["OPP"]  = teams[1].map(_safe_upper)
    else:
        df["TEAM"] = df[tcol].map(_safe_upper)
        df["OPP"]  = df[ocol].map(_safe_upper)

    scol = _find_col(df, ["salary","sal","dk_salary"])
    df["SAL"] = pd.to_numeric(df[scol], errors="coerce").fillna(0).astype(int) if scol else 0

    df["TEAM"] = df["TEAM"].astype(str).str.upper()
    df["OPP"]  = df["OPP"].astype(str).str.upper()
    df["POS"]  = df["POS"].astype(str).str.upper()
    return df

def _attach_aux(df: pd.DataFrame, aux_data_dir: Optional[str]) -> pd.DataFrame:
    if not aux_data_dir:
        return df
    root = Path(aux_data_dir)
    p = root / "player_usage.parquet"
    if p.exists():
        use = pd.read_parquet(p)
        keymap = {}
        for k in ["PLAYER","player","player_name","full_name"]:
            if k in use.columns: keymap["PLAYER"] = k; break
        for k in ["TEAM","team","recent_team","posteam"]:
            if k in use.columns: keymap["TEAM"] = k; break
        for k in ["SEASON","season","year"]:
            if k in use.columns: keymap["SEASON"] = k; break
        for k in ["WEEK","week","game_week"]:
            if k in use.columns: keymap["WEEK"] = k; break
        use2 = use.rename(columns={v:k for k,v in keymap.items()})
        on = [c for c in ["PLAYER","TEAM","SEASON","WEEK"] if c in use2.columns]
        df = df.merge(use2, on=on, how="left") if on else df

    p = root / "team_context.parquet"
    if p.exists():
        ctx = pd.read_parquet(p)
        if "TEAM" in ctx.columns:
            ctx["TEAM"] = ctx["TEAM"].astype(str).str.upper()
        on = [c for c in ["TEAM","SEASON","WEEK"] if c in ctx.columns and c in df.columns]
        df = df.merge(ctx, on=on, how="left") if on else df

    p = root / "defense_adjust.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        cand = [c for c in d.columns if c.upper() in ("OPP","TEAM","DEFTEAM","DEF_TEAM")]
        if cand:
            oppc = cand[0]
            d["_OPP"] = d[oppc].astype(str).str.upper()
            left = df.copy()
            left["_OPP"] = left["OPP"]
            df = left.merge(d.drop(columns=[oppc]).rename(columns={"_OPP":"OPP"}),
                            left_on="_OPP", right_on="OPP", how="left").drop(columns=["_OPP"])
    return df

def _build_features_for_pos(df_pos: pd.DataFrame, feat_names: List[str]) -> pd.DataFrame:
    X = df_pos.copy()
    for pos in POS_LIST:
        col = f"is_{pos}"
        if col in feat_names and col not in X.columns:
            X[col] = (X.get("POS","") == pos).astype(int)

    tm_prefixes = [c for c in feat_names if c.startswith("tm_")]
    if tm_prefixes:
        tm_dum = pd.get_dummies(X["TEAM"], prefix="tm", dtype=int)
        for c in tm_dum.columns:
            if c not in X.columns:
                X[c] = tm_dum[c]
    opp_prefixes = [c for c in feat_names if c.startswith("opp_")]
    if opp_prefixes:
        opp_dum = pd.get_dummies(X["OPP"], prefix="opp", dtype=int)
        for c in opp_dum.columns:
            if c not in X.columns:
                X[c] = opp_dum[c]

    X = X.reindex(columns=feat_names, fill_value=0.0).astype(float)
    return X

def load_quantile_bundles(models_dir: str) -> Dict[str, dict]:
    md = Path(models_dir)
    out = {}
    for pos in POS_LIST:
        path = md / f"lgbm_{pos}.pkl"
        if path.exists():
            out[pos] = joblib.load(path)
    if not out:
        raise FileNotFoundError(f"No model bundles found in {md}. Expected lgbm_QB.pkl, ...")
    return out

def score_current_week(upload_df: pd.DataFrame,
                       models_dir: str,
                       aux_data_dir: Optional[str] = None) -> pd.DataFrame:
    df = _normalize_core(upload_df)
    df = _attach_aux(df, aux_data_dir)

    bundles = load_quantile_bundles(models_dir)

    df["p10"] = np.nan
    df["p50"] = np.nan
    df["p90"] = np.nan
    df["lo80"] = np.nan
    df["hi80"] = np.nan

    for pos, bundle in bundles.items():
        dfp = df[df["POS"] == pos].copy()
        if dfp.empty: 
            continue
        feat_names = bundle["feature_names"]
        models = bundle["models"]
        delta80 = float(bundle.get("calib", {}).get("delta80", 0.0))

        X = _build_features_for_pos(dfp, feat_names)
        p10 = models["p10"].predict(X)
        p50 = models["p50"].predict(X)
        p90 = models["p90"].predict(X)

        df.loc[dfp.index, "p10"] = p10
        df.loc[dfp.index, "p50"] = p50
        df.loc[dfp.index, "p90"] = p90
        if delta80 > 0:
            df.loc[dfp.index, "lo80"] = p50 - delta80
            df.loc[dfp.index, "hi80"] = p50 + delta80

    return df
