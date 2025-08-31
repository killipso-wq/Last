# simulation_engine.py
# ML + (optional) Monte Carlo slate simulator with correlation-aware stacks.
# - Uses your LightGBM quantile models (p50/p90) if present in ./models
# - Analytic mode (default): no sampling; fast ceilings via covariance math
# - Monte Carlo mode: set n_sims >= 5000 in the UI; runs per-game Gaussian copula sims
# - Returns (players_df, game_tbl) as your app expects

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

MODEL_DIR = Path("models")
Z90 = 1.2815515655446  # Phi^-1(0.90)

# -----------------------------
# Basic normalization
# -----------------------------
def _upper_strip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PLAYER
    if "PLAYER" not in df.columns:
        for c in ("player", "player_name", "player_display_name", "Name"):
            if c in df.columns:
                df["PLAYER"] = _upper_strip(df[c]); break
    else:
        df["PLAYER"] = _upper_strip(df["PLAYER"])

    # POS
    if "POS" not in df.columns:
        for c in ("position", "Position"):
            if c in df.columns:
                df["POS"] = _upper_strip(df[c]); break
    else:
        df["POS"] = _upper_strip(df["POS"])

    # TEAM/OPP
    if "TEAM" not in df.columns:
        for c in ("team", "recent_team"):
            if c in df.columns:
                df["TEAM"] = _upper_strip(df[c]); break
    else:
        df["TEAM"] = _upper_strip(df["TEAM"])
    if "OPP" not in df.columns:
        for c in ("opponent_team", "Opp"):
            if c in df.columns:
                df["OPP"] = _upper_strip(df[c]); break
    else:
        df["OPP"] = _upper_strip(df["OPP"])

    # SAL
    if "SAL" not in df.columns:
        for c in ("Salary","salary"):
            if c in df.columns:
                df["SAL"] = pd.to_numeric(df[c], errors="coerce"); break

    # OWN (0–1)
    if "OWN" not in df.columns:
        for c in ("Ownership","OWN%","RST%","own"):
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce")
                df["OWN"] = vals/100.0 if vals.max() and vals.max() > 1.01 else vals
                break
    if "OWN" not in df.columns:
        df["OWN"] = 0.0

    # CSV_MED fallback
    if "CSV_MED" not in df.columns:
        for c in ("Proj","PROJ","fantasy_points_ppr","fantasy_points","MEDIAN","MED"):
            if c in df.columns:
                df["CSV_MED"] = pd.to_numeric(df[c], errors="coerce"); break
    if "CSV_MED" not in df.columns:
        df["CSV_MED"] = np.nan

    return df

# -----------------------------
# Load models & predict quantiles
# -----------------------------
def _load_models() -> Dict[str, object]:
    models = {}
    if joblib is None:
        return models
    for pos in ("QB","RB","WR","TE"):
        pkl = MODEL_DIR / f"lgbm_{pos}.pkl"
        if pkl.exists():
            try:
                models[pos] = joblib.load(pkl)
            except Exception:
                pass
    return models

def _expected_feat_names(model) -> List[str]:
    names = None
    if hasattr(model, "booster_") and model.booster_ is not None:
        try:
            names = list(model.booster_.feature_name())
        except Exception:
            names = None
    if names is None and hasattr(model, "feature_name_"):
        names = list(model.feature_name_)
    return names or []

def _build_feats_for_infer(df_pos: pd.DataFrame, feat_names: List[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df_pos.index)
    team_d = pd.get_dummies(df_pos["TEAM"], prefix="team", dtype=int)
    opp_d  = pd.get_dummies(df_pos["OPP"],  prefix="opp",  dtype=int)
    X = pd.concat([X, team_d, opp_d], axis=1)
    if "SAL" in df_pos.columns:
        X["SAL"] = pd.to_numeric(df_pos["SAL"], errors="coerce").fillna(0)
    if feat_names:
        for c in feat_names:
            if c not in X.columns:
                X[c] = 0
        X = X[feat_names]
    return X

def _predict_quantiles_for_pos(df: pd.DataFrame, pos: str, model) -> Tuple[pd.Series, pd.Series]:
    df_pos = df[df["POS"] == pos].copy()
    if df_pos.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    feat_names = _expected_feat_names(model)
    X = _build_feats_for_infer(df_pos, feat_names)

    p50, p90 = None, None
    try:
        if isinstance(model, dict):
            if "p50" in model:
                p50 = pd.Series(model["p50"].predict(X), index=df_pos.index)
            if "p90" in model:
                p90 = pd.Series(model["p90"].predict(X), index=df_pos.index)
    except Exception:
        pass

    if p50 is None:
        try:
            p50 = pd.Series(model.predict(X), index=df_pos.index)
        except Exception:
            p50 = pd.Series(df_pos.get("CSV_MED", 0.0), index=df_pos.index)

    if p90 is None:
        bump = 1.1 * np.sqrt(np.clip(p50.values, 0, None))
        p90 = pd.Series(p50.values + bump, index=df_pos.index)

    return p50.clip(lower=0), p90.clip(lower=0)

def _predict_all_positions(df: pd.DataFrame, models: Dict[str, object]) -> Tuple[pd.Series, pd.Series]:
    p50_all = pd.Series(index=df.index, dtype=float)
    p90_all = pd.Series(index=df.index, dtype=float)

    for pos in ("QB","RB","WR","TE"):
        if pos in models:
            p50, p90 = _predict_quantiles_for_pos(df, pos, models[pos])
        else:
            idx = df["POS"] == pos
            base = pd.to_numeric(df.loc[idx, "CSV_MED"], errors="coerce").fillna(0.0)
            p50, p90 = base, base + 1.1*np.sqrt(np.clip(base.values,0,None))
        p50_all.loc[p50.index] = p50
        p90_all.loc[p90.index] = p90

    mask_rest = ~df["POS"].isin(["QB","RB","WR","TE"])
    base = pd.to_numeric(df.loc[mask_rest, "CSV_MED"], errors="coerce").fillna(0.0)
    p50_all.loc[mask_rest] = base
    p90_all.loc[mask_rest] = base * 1.25

    return p50_all.fillna(0.0), p90_all.fillna(0.0)

# -----------------------------
# Uncertainty & correlation
# -----------------------------
def _sigma_from_quantiles(p50: pd.Series, p90: pd.Series) -> pd.Series:
    sig = (p90 - p50) / Z90
    return sig.clip(lower=0.25)  # floor

def _injury_widen(sig: pd.Series, df: pd.DataFrame) -> pd.Series:
    widen = np.ones_like(sig.values, dtype=float)
    status_cols = [c for c in df.columns if c.upper() in ("STATUS","INJURY","INJ")]
    if status_cols:
        s = df[status_cols[0]].astype(str).str.upper().fillna("")
        widen += 0.25 * s.str.contains("Q|DTD|LP|PROB").astype(float)
        widen += 0.50 * s.str.contains("D|OUT").astype(float)
    return sig * widen

def _rho_rule(a_pos: str, b_pos: str, same_team: bool, opponents: bool) -> float:
    a = a_pos.upper(); b = b_pos.upper()
    if same_team:
        if "QB" in (a,b) and (("WR" in (a+b)) or ("TE" in (a+b))):
            return 0.45
        if (a.startswith("WR") and b.startswith("WR")) or (set([a,b])==set(["WR","TE"])):
            return 0.20
        if ("RB" in (a,b)) and ("QB" in (a,b)):
            return 0.12
        if ("RB" in (a,b)) and ("WR" in (a,b)):
            return 0.08
        if ("TE" in (a,b)) and ("RB" in (a,b)):
            return 0.05
        return 0.05
    if opponents:
        if (("QB" in (a,b)) and (("WR" in (a+b)) or ("TE" in (a+b)))) or \
           (a.startswith("WR") and b.startswith("WR")):
            return 0.15
        if ("RB" in (a,b)) and (("QB" in (a,b)) or ("WR" in (a,b)) or ("TE" in (a,b))):
            return -0.10
        return 0.05
    return 0.0

# -----------------------------
# Analytic stack ceiling (no sampling)
# -----------------------------
def _stack_p90(players: pd.DataFrame, ids: List[int], rho_lookup) -> float:
    mu = players.loc[ids, "__mu__"].values
    sg = players.loc[ids, "__sig__"].values
    var = float(np.sum(sg**2))
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            rho = rho_lookup(ids[i], ids[j])
            var += 2.0 * rho * sg[i] * sg[j]
    var = max(var, 0.0)
    return float(np.sum(mu) + Z90 * np.sqrt(var))

# -----------------------------
# Monte Carlo per-game copula simulation
# -----------------------------
def _game_key(team: str, opp: str) -> tuple:
    # undirected game key (TEAM vs OPP considered one block)
    pair = tuple(sorted([str(team), str(opp)]))
    return pair

def _build_corr_block(g: pd.DataFrame) -> np.ndarray:
    n = len(g)
    C = np.eye(n, dtype=float)
    pos = g["POS"].values.astype(str)
    team = g["TEAM"].values.astype(str)
    opp  = g["OPP"].values.astype(str)
    for i in range(n):
        for j in range(i+1, n):
            same_team = (team[i] == team[j])
            opponents = (team[i] == opp[j]) and (opp[i] == team[j])
            rho = _rho_rule(pos[i], pos[j], same_team, opponents)
            rho = float(np.clip(rho, -0.95, 0.95))
            C[i, j] = C[j, i] = rho
    return C

def _simulate_block_gaussian(mu: np.ndarray, sig: np.ndarray, C: np.ndarray, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    # Ensure positive definite (add jitter if needed)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        eps = 1e-6
        for _ in range(6):
            try:
                L = np.linalg.cholesky(C + np.eye(C.shape[0])*eps)
                break
            except np.linalg.LinAlgError:
                eps *= 10
        else:
            # fallback to diag if still bad
            L = np.linalg.cholesky(np.eye(C.shape[0]))
    Z = rng.standard_normal(size=(C.shape[0], n_sims))
    G = L @ Z  # correlated standard normals
    samples = (mu[:, None] + sig[:, None] * G).astype(np.float32)
    # clamp at zero (fantasy points cannot go negative in most DK scoring contexts)
    np.maximum(samples, 0.0, out=samples)
    return samples  # shape (n_players_in_block, n_sims)

# -----------------------------
# Shared post-processing (players + game_tbl)
# -----------------------------
def _build_game_table(players: pd.DataFrame, rho_lookup=None, samples_by_block=None, block_indices=None, n_sims: int = 0) -> pd.DataFrame:
    rows = []
    # Build from analytic (rho_lookup) or MC (samples_by_block)
    # Group by directed TEAM/OPP to select QB + pass-catchers consistently
    for (tm, opp), g in players.groupby(["TEAM","OPP"], dropna=False):
        g2 = g[g["POS"].astype(str).str.upper() != "DST"].copy()
        if g2.empty:
            continue
        qb = g2[g2["POS"]=="QB"].sort_values("FPTS_p90", ascending=False)
        if qb.empty:
            continue
        qb_id = int(qb["_id"].iloc[0])
        pass_catch = g2[g2["POS"].isin(["WR","TE"])].sort_values("FPTS_p90", ascending=False)
        if pass_catch.empty:
            continue
        wr1_id = int(pass_catch["_id"].iloc[0])
        wr2_id = int(pass_catch["_id"].iloc[1]) if len(pass_catch) > 1 else None

        # Bring-back from opponent
        opp_pool = players[(players["TEAM"]==opp) & (players["OPP"]==tm) & (players["POS"].isin(["WR","TE"]))]
        br_id = int(opp_pool.sort_values("FPTS_p90", ascending=False)["_id"].iloc[0]) if not opp_pool.empty else None

        if samples_by_block is None:
            # Analytic ceiling
            p90_2 = _stack_p90(players, [qb_id, wr1_id], rho_lookup)
            p90_3 = _stack_p90(players, [qb_id, wr1_id] + ([wr2_id] if wr2_id is not None else []), rho_lookup)
            p90_3x1 = np.nan
            if br_id is not None:
                ids = [qb_id, wr1_id] + ([wr2_id] if wr2_id is not None else []) + [br_id]
                p90_3x1 = _stack_p90(players, ids, rho_lookup)
        else:
            # Monte Carlo ceiling from samples
            key = _game_key(tm, opp)
            if key not in samples_by_block:
                continue
            # map global row indices -> within-block row indices
            block_rows = block_indices[key]  # pd.Index of global rows in this block
            id_to_loc = {int(players.loc[r, "_id"]): k for k, r in enumerate(block_rows.tolist())}
            # ensure we have all ids in this block
            ids2 = [id_to_loc.get(qb_id, None), id_to_loc.get(wr1_id, None)]
            ids3 = ids2 + ([id_to_loc.get(wr2_id, None)] if wr2_id is not None else [])
            ids31 = ids3 + ([id_to_loc.get(br_id, None)] if br_id is not None else [])
            S = samples_by_block[key]  # shape (n_players_in_block, n_sims)

            def p90_sum(id_list):
                idxs = [i for i in id_list if i is not None]
                if not idxs:
                    return np.nan
                total = np.sum(S[idxs, :], axis=0)
                return float(np.percentile(total, 90))
            p90_2  = p90_sum(ids2)
            p90_3  = p90_sum(ids3)
            p90_3x1 = p90_sum(ids31) if br_id is not None else np.nan

        rows.append({
            "TEAM": tm, "OPP": opp,
            "Stack2_P90": round(p90_2, 2) if p90_2==p90_2 else np.nan,
            "Stack3_P90": round(p90_3, 2) if p90_3==p90_3 else np.nan,
            "Stack3x1_P90": round(p90_3x1, 2) if p90_3x1==p90_3x1 else np.nan,
            "QB": players.loc[players["_id"]==qb_id, "PLAYER"].iloc[0],
            "Primary": players.loc[players["_id"]==wr1_id, "PLAYER"].iloc[0],
            "Secondary": players.loc[players["_id"]==wr2_id, "PLAYER"].iloc[0] if wr2_id is not None else "",
            "BringBack": players.loc[players["_id"]==br_id, "PLAYER"].iloc[0] if br_id is not None else "",
            "players": len(g2)
        })

    game_tbl = pd.DataFrame(rows).sort_values(
        ["Stack3x1_P90","Stack3_P90","Stack2_P90"],
        ascending=False, na_position="last"
    ).reset_index(drop=True)
    return game_tbl

# -----------------------------
# Public API (used by app.py)
# -----------------------------
def run_simulation_with_best_effort(df_in: pd.DataFrame, n_sims: int = 0):
    """
    Returns (players_df, game_tbl).
    - If n_sims >= 5000 -> Monte Carlo mode (Gaussian copula per game)
    - Else -> Analytic mode (fast, no sampling)
    """
    df = _ensure_cols(df_in)
    models = _load_models()

    # Predict ML quantiles
    p50, p90 = _predict_all_positions(df, models)
    sig = _sigma_from_quantiles(p50, p90)
    sig = _injury_widen(sig, df)

    players = df.copy()
    players["MED_final"] = p50.clip(lower=0.0)
    players["FPTS_p90"]  = p90.clip(lower=players["MED_final"])
    players["__mu__"]    = players["MED_final"]
    players["__sig__"]   = sig

    # Value-style GPP score with anti-ownership
    if "SAL" in players.columns:
        val = (players["FPTS_p90"] / players["SAL"].replace(0, np.nan)).fillna(0)
    else:
        val = players["FPTS_p90"].fillna(0)
    players["SCORE_gpp"] = val * (1.15 - players["OWN"].fillna(0.0))

    # Make a stable id column for internal lookups
    players = players.reset_index(drop=False).rename(columns={"index":"_id"})

    # --------- Choose mode ----------
    if n_sims is not None and int(n_sims) >= 5000:
        # Monte Carlo mode
        rng = np.random.default_rng(12345)
        # Build per-game blocks
        block_samples: Dict[tuple, np.ndarray] = {}
        block_indices: Dict[tuple, pd.Index] = {}

        # Undirected game key blocks (combine TEAM vs OPP both directions)
        game_blocks = {}
        for i, row in players.iterrows():
            key = _game_key(row["TEAM"], row["OPP"])
            game_blocks.setdefault(key, []).append(i)

        for key, idxs in game_blocks.items():
            g = players.loc[idxs]
            # Exclude DST from correlation block (optional)
            g_block = g[g["POS"].astype(str).str.upper() != "DST"].copy()
            if g_block.empty:
                continue
            mu = g_block["__mu__"].to_numpy(dtype=float)
            sg = g_block["__sig__"].to_numpy(dtype=float)
            C  = _build_corr_block(g_block)
            S  = _simulate_block_gaussian(mu, sg, C, int(n_sims), rng)  # (n_block, n_sims)
            block_samples[key] = S
            block_indices[key] = g_block.index

        # From sampled draws: set player med/p90 to simulated quantiles (close to ML marginals, but empirical)
        # Merge back per block
        # Start with analytic as fallback
        sim_p50 = pd.Series(players["MED_final"].values, index=players.index, dtype=float)
        sim_p90 = pd.Series(players["FPTS_p90"].values, index=players.index, dtype=float)
        for key, S in block_samples.items():
            idxs = block_indices[key]
            q50 = np.percentile(S, 50, axis=1)
            q90 = np.percentile(S, 90, axis=1)
            sim_p50.loc[idxs] = q50
            sim_p90.loc[idxs] = q90
        players["MED_final"] = sim_p50.clip(lower=0.0)
        players["FPTS_p90"]  = sim_p90.clip(lower=players["MED_final"])

        # Build game table from MC samples
        game_tbl = _build_game_table(players, samples_by_block=block_samples, block_indices=block_indices, n_sims=int(n_sims))

    else:
        # Analytic mode
        def rho_lookup(i: int, j: int) -> float:
            ai = players.loc[players["_id"]==i, ["POS","TEAM","OPP"]].iloc[0]
            bi = players.loc[players["_id"]==j, ["POS","TEAM","OPP"]].iloc[0]
            same_team = (ai["TEAM"] == bi["TEAM"])
            opponents = (ai["TEAM"] == bi["OPP"]) and (ai["OPP"] == bi["TEAM"])
            return _rho_rule(str(ai["POS"]), str(bi["POS"]), same_team, opponents)

        game_tbl = _build_game_table(players, rho_lookup=rho_lookup)

    # Clean public frame
    players = players.drop(columns=["__mu__","__sig__"], errors="ignore")

    return players, game_tbl

# Aliases your app already tries
def run_simulation(df: pd.DataFrame, n_sims: int = 0):
    return run_simulation_with_best_effort(df, n_sims)

def process_uploaded_file(df: pd.DataFrame, n_sims: int = 0):
    return run_simulation_with_best_effort(df, n_sims)
