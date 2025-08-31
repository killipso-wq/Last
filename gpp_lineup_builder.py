# gpp_lineup_builder.py
# One-file GPP optimizer skeleton you can extend with your blueprint sections.
# No external solver required; uses a fast heuristic with stacking/exposure controls.

from __future__ import annotations
import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# Config & small utilities
# -------------------------------

@dataclass
class ContestConfig:
    site: str = "DK"
    salary_cap: int = 50000
    # Default DK NFL classic (no DST enforced here; add if your data includes it)
    slots: List[str] = field(default_factory=lambda: ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"])
    # FLEX eligibility
    flex_pool: Set[str] = field(default_factory=lambda: {"RB", "WR", "TE"})
    # Stacking
    qb_teammate_min: int = 1   # at least 1 WR/TE with same team as QB
    qb_bringback_min: int = 0  # at least 0 opponent WR/TE (set 1 to enforce bring-back)
    # Limits
    max_exposure: float = 0.60  # 60% default cap per player
    max_per_team: int = 4       # basic team cap (tweak later)
    # Randomness
    temperature: float = 0.10   # 0..1; higher = more random
    seed: int = 7               # reproducibility

    # Columns expected from your Final GPP table
    col_player: str = "PLAYER"
    col_team: str = "TEAM"
    col_opp: str = "OPP"
    col_pos: str = "POS"
    col_salary: str = "SAL"
    col_own: str = "OWN"            # normalized 0..1 (we’ll compute from RST% if needed)
    col_score: str = "SCORE_gpp"    # target to sort/rank; fallback to MED_final
    col_med: str = "MED_final"
    col_p90: str = "FPTS_p90"       # used for a little ceiling bias
    col_used_ml: str = "used_ml"    # optional flag
    col_team_abbrs_upper: bool = True  # standardize TEAM/OPP to uppercase abbreviations


def _safe_upper(x):
    try:
        return str(x).upper().strip()
    except Exception:
        return x


def _ensure_cols(df: pd.DataFrame, names: List[str]):
    missing = [c for c in names if c not in df.columns]
    return missing


# -------------------------------
# Loading & normalization
# -------------------------------

def load_players_table(path: str) -> pd.DataFrame:
    """
    Loads Final GPP table (CSV/Parquet). If it only contains your players.csv,
    we’ll map basic columns and fabricate minimal SCORE_gpp = PROJ.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)

    # Normalize common column names in case user feeds the original players.csv
    colmap_candidates = {
        "PLAYER": ["PLAYER", "player", "player_name", "Player"],
        "TEAM": ["TEAM", "team", "Team"],
        "OPP": ["OPP", "opponent", "Opponent", "OPPONENT"],
        "POS": ["POS", "position", "Position"],
        "SAL": ["SAL", "salary", "Salary"],
        "RST%": ["RST%", "Rst%", "Ownership", "OWN%"],
        "FPTS": ["FPTS", "proj", "projection", "PROJ", "Projected", "Fantasy Points"],
        "SCORE_gpp": ["SCORE_gpp"],
        "MED_final": ["MED_final", "median", "MED"],
        "FPTS_p90": ["FPTS_p90", "p90", "Ceiling"],
        "FPTS_p25": ["FPTS_p25", "p25", "Floor"],
        "used_ml": ["used_ml"]
    }

    def pick_col(df, alts, default=None):
        for c in alts:
            if c in df.columns:
                return c
        return default

    # Build a working frame with the columns we care about
    out = pd.DataFrame()
    for std, alts in colmap_candidates.items():
        c = pick_col(raw, alts, None)
        if c is not None:
            out[std] = raw[c]

    # Fallbacks if Final GPP fields missing
    if "SCORE_gpp" not in out.columns:
        # Use provided final median if present, else FPTS from CSV as baseline
        if "MED_final" in out.columns:
            out["SCORE_gpp"] = pd.to_numeric(out["MED_final"], errors="coerce")
        elif "FPTS" in out.columns:
            out["SCORE_gpp"] = pd.to_numeric(out["FPTS"], errors="coerce")
        else:
            out["SCORE_gpp"] = 0.0

    if "MED_final" not in out.columns:
        if "FPTS" in out.columns:
            out["MED_final"] = pd.to_numeric(out["FPTS"], errors="coerce")
        else:
            out["MED_final"] = pd.to_numeric(out["SCORE_gpp"], errors="coerce")

    # OWN from RST% if OWN missing
    if "OWN" not in out.columns:
        if "RST%" in out.columns:
            own = pd.to_numeric(out["RST%"], errors="coerce").fillna(0.0) / 100.0
        else:
            own = 0.0
        out["OWN"] = own

    # Ensure types
    for c in ["SAL", "SCORE_gpp", "MED_final", "FPTS_p90", "FPTS_p25", "OWN"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Uppercase team/opponent if present
    if "TEAM" in out.columns:
        out["TEAM"] = out["TEAM"].map(_safe_upper)
    if "OPP" in out.columns:
        out["OPP"] = out["OPP"].map(_safe_upper)
    if "POS" in out.columns:
        out["POS"] = out["POS"].map(_safe_upper)

    # Drop blatantly invalid rows
    out = out.dropna(subset=["PLAYER", "TEAM", "POS", "SAL"]).copy()
    out = out[out["SAL"] > 0].copy()

    # Small ceiling-friendly score tweak: bias by p90 if available
    if "FPTS_p90" in out.columns:
        # Add tiny fraction of ceiling to the score ranking—not to the exported median
        out["_score_rank"] = out["SCORE_gpp"].fillna(0) + 0.03 * out["FPTS_p90"].fillna(0)
    else:
        out["_score_rank"] = out["SCORE_gpp"].fillna(0)

    # Extra helpful derived fields
    out["ID"] = out["PLAYER"].astype(str) + " (" + out["TEAM"].astype(str) + ")"
    return out


# -------------------------------
# Lineup model
# -------------------------------

@dataclass
class Lineup:
    players: List[int] = field(default_factory=list)  # store DF indices
    salary: int = 0
    teams: Dict[str, int] = field(default_factory=dict)
    pos_counts: Dict[str, int] = field(default_factory=dict)

    def as_names(self, df: pd.DataFrame) -> List[str]:
        return [df.loc[i, "PLAYER"] for i in self.players]

    def as_row(self, df: pd.DataFrame) -> Dict[str, object]:
        names = [df.loc[i, "PLAYER"] for i in self.players]
        teams = [df.loc[i, "TEAM"] for i in self.players]
        poss  = [df.loc[i, "POS"] for i in self.players]
        sals  = [int(df.loc[i, "SAL"]) for i in self.players]
        score = float(df.loc[self.players, "SCORE_gpp"].sum())
        med   = float(df.loc[self.players, "MED_final"].sum())
        return {
            **{f"P{j+1}": n for j, n in enumerate(names)},
            **{f"T{j+1}": t for j, t in enumerate(teams)},
            **{f"Pos{j+1}": p for j, p in enumerate(poss)},
            **{f"Sal{j+1}": s for j, s in enumerate(sals)},
            "TotalSalary": self.salary,
            "TotalScore": round(score, 3),
            "TotalMedian": round(med, 3),
        }


def _eligible_for_slot(pos: str, slot: str, flex_pool: Set[str]) -> bool:
    if slot == "FLEX":
        return pos in flex_pool
    return pos == slot


# -------------------------------
# Stacking checks (basic)
# -------------------------------

def lineup_satisfies_stacks(lineup: Lineup, df: pd.DataFrame, cfg: ContestConfig) -> bool:
    """Enforce QB teammate stack and optional bringback (opponent WR/TE)."""
    # Find the QB in lineup
    qb_idx = None
    for i in lineup.players:
        if df.loc[i, "POS"] == "QB":
            qb_idx = i
            break
    if qb_idx is None:
        return True  # no QB found (unusual), skip stack checks

    qb_team = df.loc[qb_idx, "TEAM"]
    qb_opp  = df.loc[qb_idx, "OPP"] if "OPP" in df.columns else None

    # Count teammates WR/TE with same team
    mates = 0
    for i in lineup.players:
        if i == qb_idx:
            continue
        pos = df.loc[i, "POS"]
        if pos in {"WR", "TE"} and df.loc[i, "TEAM"] == qb_team:
            mates += 1
    if mates < cfg.qb_teammate_min:
        return False

    # Bring-back (optional)
    if cfg.qb_bringback_min > 0 and qb_opp:
        opp_recv = 0
        for i in lineup.players:
            pos = df.loc[i, "POS"]
            if pos in {"WR", "TE"} and df.loc[i, "TEAM"] == qb_opp:
                opp_recv += 1
        if opp_recv < cfg.qb_bringback_min:
            return False

    return True


# -------------------------------
# Heuristic lineup builder
# -------------------------------

def build_lineups(df: pd.DataFrame, cfg: ContestConfig, n_lineups: int = 150) -> Tuple[List[Lineup], Dict[str, int]]:
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    # Index pools by slot for speed
    pools: Dict[str, List[int]] = {}
    for slot in cfg.slots:
        if slot == "FLEX":
            eligible = df.index[df["POS"].isin(cfg.flex_pool)].tolist()
        else:
            eligible = df.index[df["POS"] == slot].tolist()
        # sort by our rank score
        eligible = sorted(eligible, key=lambda i: df.loc[i, "_score_rank"], reverse=True)
        pools[slot] = eligible

    # Exposure tracking (counts across built lineups)
    exposure_counts: Dict[int, int] = {i: 0 for i in df.index}
    built: List[Lineup] = []
    seen_lineup_hashes: Set[Tuple[int, ...]] = set()

    tries = 0
    max_tries = n_lineups * 400  # safety
    while len(built) < n_lineups and tries < max_tries:
        tries += 1
        lineup = Lineup(players=[], salary=0, teams={}, pos_counts={})

        # fill slots greedily with randomness
        used: Set[int] = set()
        ok = True
        for slot in cfg.slots:
            # soft cap to leave room for remaining slots: keep a simple average per-slot budget
            slots_left = len(cfg.slots) - len(lineup.players)
            avg_budget = max(1, (cfg.salary_cap - lineup.salary) // max(1, slots_left))

            cand = candidate_picker(
                df, pools[slot], used, exposure_counts, len(built), cfg,
                salary_ceiling=avg_budget + 1500  # allow a bit above average
            )
            if cand is None:
                ok = False
                break

            # add player
            lineup.players.append(cand)
            lineup.salary += int(df.loc[cand, "SAL"])
            t = df.loc[cand, "TEAM"]
            lineup.teams[t] = lineup.teams.get(t, 0) + 1
            p = df.loc[cand, "POS"]
            lineup.pos_counts[p] = lineup.pos_counts.get(p, 0) + 1
            used.add(cand)

        if not ok:
            continue

        # Hard checks
        if lineup.salary > cfg.salary_cap:
            continue
        if any(count > cfg.max_per_team for count in lineup.teams.values()):
            continue
        # Stacking
        if not lineup_satisfies_stacks(lineup, df, cfg):
            continue

        # Deduplicate
        key = tuple(sorted(lineup.players))
        if key in seen_lineup_hashes:
            continue
        seen_lineup_hashes.add(key)

        # record lineup
        built.append(lineup)
        for i in lineup.players:
            exposure_counts[i] += 1

    # Final exposure map by player name
    name_exposure = {df.loc[i, "PLAYER"]: exposure_counts[i] for i in df.index}
    return built, name_exposure


def candidate_picker(
    df: pd.DataFrame,
    pool: List[int],
    used: Set[int],
    exposure_counts: Dict[int, int],
    built_so_far: int,
    cfg: ContestConfig,
    salary_ceiling: int
) -> Optional[int]:
    """Pick one candidate for a slot with soft constraints and randomness."""
    # Candidate mask
    cand = []
    for i in pool:
        if i in used:
            continue
        sal = int(df.loc[i, "SAL"])
        # Soft guard: don't pick someone way above the current average-per-slot budget
        if sal > salary_ceiling and random.random() > 0.25:
            continue

        # Exposure cap
        if built_so_far > 0:
            if exposure_counts[i] / max(1, built_so_far) > cfg.max_exposure:
                continue

        cand.append(i)

    if not cand:
        return None

    # Scores with small random noise (temperature)
    base = df.loc[cand, "_score_rank"].to_numpy(dtype=float)
    noise = np.random.normal(0, cfg.temperature * np.maximum(1.0, base.std() if base.std() > 0 else 1.0), size=base.shape)
    weights = np.maximum(1e-6, base + noise)

    # Bias away from chalk a bit
    if "OWN" in df.columns:
        own = df.loc[cand, "OWN"].to_numpy(dtype=float)
        weights = weights * (1.0 - 0.25 * np.clip(own, 0.0, 1.0))

    # Convert to probabilities and sample
    probs = weights / weights.sum()
    choice = np.random.choice(cand, p=probs)
    return int(choice)


# -------------------------------
# Orchestration / CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="DFS NFL GPP lineup builder (one-file heuristic optimizer).")
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Path to Final GPP CSV/Parquet (from your app). Can be your players.csv as fallback.")
    ap.add_argument("--out", dest="out_path", default="lineups.csv",
                    help="Where to write lineups CSV (default: lineups.csv)")
    ap.add_argument("--n", dest="n_lineups", type=int, default=150, help="How many lineups to build (default 150)")
    ap.add_argument("--cap", dest="cap", type=int, default=50000, help="Salary cap (default 50000)")
    ap.add_argument("--maxexp", dest="maxexp", type=float, default=0.60, help="Max exposure per player (0..1)")
    ap.add_argument("--stack", dest="stack", default="1,0", help="QB stack settings: qb_mates,bringback (e.g., 1,1)")
    ap.add_argument("--seed", dest="seed", type=int, default=7, help="Random seed")
    ap.add_argument("--temp", dest="temp", type=float, default=0.10, help="Randomness temperature 0..1")
    args = ap.parse_args()

    cfg = ContestConfig()
    cfg.salary_cap = args.cap
    cfg.max_exposure = args.maxexp
    cfg.seed = args.seed
    cfg.temperature = args.temp

    try:
        mates, bb = args.stack.split(",")
        cfg.qb_teammate_min = int(mates)
        cfg.qb_bringback_min = int(bb)
    except Exception:
        pass

    # Load table
    df = load_players_table(args.in_path)

    # Basic slot sanity: if no QB in data, try WR-only test builds
    if not (df["POS"] == "QB").any():
        cfg.slots = ["WR", "WR", "WR", "WR", "WR", "WR", "WR", "WR"]  # emergency test
        cfg.flex_pool = {"WR"}

    # Build
    lineups, exposures = build_lineups(df, cfg, n_lineups=args.n_lineups)

    if not lineups:
        print("No lineups built. Try relaxing constraints or verify your input columns.", file=sys.stderr)
        sys.exit(2)

    # Save lineups CSV
    rows = [lu.as_row(df) for lu in lineups]
    out = pd.DataFrame(rows)
    # Sum fantasy score using SCORE_gpp as the objective shown
    out = out.sort_values("TotalScore", ascending=False).reset_index(drop=True)
    out.to_csv(args.out_path, index=False)
    print(f"✓ Wrote {len(out)} lineups -> {os.path.abspath(args.out_path)}")

    # Exposure report
    exp_tbl = (
        pd.DataFrame({"PLAYER": list(exposures.keys()), "Count": list(exposures.values())})
        .assign(ExposurePct=lambda d: 100.0 * d["Count"] / len(lineups))
        .sort_values(["Count", "PLAYER"], ascending=[False, True])
        .reset_index(drop=True)
    )
    exp_csv = os.path.splitext(args.out_path)[0] + "_exposure.csv"
    exp_tbl.to_csv(exp_csv, index=False)
    print(f"✓ Wrote exposure -> {os.path.abspath(exp_csv)}")

    # Quick print top 10 exposure
    print("\nTop 10 exposures:")
    print(exp_tbl.head(10).to_string(index=False))

    # ---------- BLUEPRINT HOOKS ----------
    # As you share each blueprint section, we'll add:
    # - Extra constraints in candidate_picker & lineup_satisfies_stacks
    # - Pre-filters on df (e.g., min p25, value thresholds)
    # - Advanced correlation/bringback/secondary stacks
    # - Global exposure bands (pos/team/QB stacks)
    # - Late swap modes, etc.


if __name__ == "__main__":
    main()
