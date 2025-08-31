# optimizer_gpp.py
# -------------------------------------------------------------
# One-file NFL DFS GPP optimizer that consumes your Final GPP
# table (from the Streamlit app) and builds 150 lineups aimed
# at first-place finishes. Heuristic search with stacking,
# exposure caps, team caps, and hooks for your blueprint rules.
# -------------------------------------------------------------
from __future__ import annotations
import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# =====================
# Config + RuleBook
# =====================
@dataclass
class RuleBook:
    # Site + roster
    site: str = "DK"
    salary_cap: int = 50000
    slots: List[str] = field(default_factory=lambda: [
        "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"
    ])
    flex_pool: Set[str] = field(default_factory=lambda: {"RB", "WR", "TE"})

    # Stacking
    qb_teammate_min: int = 1   # min WR/TE teammates with QB
    qb_bringback_min: int = 0  # min WR/TE bring-backs from opponent
    secondary_stack_size: int = 0  # e.g., 2 (WR+TE same team), 0 disables

    # Limits
    max_exposure: float = 0.60      # per-player cap, 0..1
    max_per_team: int = 4           # max players from same team
    min_salary_used: int = 49500    # avoid leaving too much salary
    min_uniques: int = 2            # force some diversity across lineups

    # Ownership / filtering
    fade_chalk_weight: float = 0.25  # how hard to downweight OWN in sampling
    min_floor_p25: Optional[float] = None  # e.g., 6.0 to avoid total dust in small fields
    min_value_med_per_1k: Optional[float] = None  # e.g., 2.0

    # Randomness
    temperature: float = 0.10  # 0..1 stochasticity in candidate sampling
    seed: int = 7

    # Column names expected in Final GPP table
    col_player: str = "PLAYER"
    col_team: str = "TEAM"
    col_opp: str = "OPP"
    col_pos: str = "POS"
    col_salary: str = "SAL"
    col_score: str = "SCORE_gpp"   # target to maximize
    col_med: str = "MED_final"
    col_p90: str = "FPTS_p90"
    col_p25: str = "FPTS_p25"
    col_own: str = "OWN"           # 0..1

    # Optional controls (locks/excludes/groups)
    lock_list: Set[str] = field(default_factory=set)     # player names to force
    exclude_list: Set[str] = field(default_factory=set)  # player names to ban

    # BLUEPRINT HOOKS: We'll add more fields here as you paste sections


# =====================
# Data loading
# =====================
def _safe_upper(x):
    try:
        return str(x).upper().strip()
    except Exception:
        return x


def _pick_col(df: pd.DataFrame, alts: List[str], default=None):
    for c in alts:
        if c in df.columns:
            return c
    return default


def load_final_table(path: str, strict: bool = False) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)

    # map columns
    cmap = {
        "PLAYER": ["PLAYER", "player", "player_name", "Player"],
        "TEAM": ["TEAM", "team", "Team"],
        "OPP": ["OPP", "opponent", "Opponent"],
        "POS": ["POS", "position", "Position"],
        "SAL": ["SAL", "salary", "Salary"],
        "SCORE_gpp": ["SCORE_gpp"],
        "MED_final": ["MED_final", "median", "MED"],
        "FPTS_p90": ["FPTS_p90", "p90", "Ceiling"],
        "FPTS_p25": ["FPTS_p25", "p25", "Floor"],
        "OWN": ["OWN", "ownership", "Ownership"],
        "RST%": ["RST%", "OWN%", "own%"],
    }

    out = pd.DataFrame()
    for k, alts in cmap.items():
        c = _pick_col(raw, alts)
        if c is not None:
            out[k] = raw[c]

    # OWN fallback from RST%
    if "OWN" not in out.columns and "RST%" in out.columns:
        out["OWN"] = pd.to_numeric(out["RST%"], errors="coerce").fillna(0.0) / 100.0

    # Type coercions
    num_cols = ["SAL", "SCORE_gpp", "MED_final", "FPTS_p90", "FPTS_p25", "OWN"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # uppercase teams/pos
    for c in ["TEAM", "OPP", "POS"]:
        if c in out.columns:
            out[c] = out[c].map(_safe_upper)

    # Basic validation
    need = ["PLAYER", "TEAM", "POS", "SAL"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in Final GPP table: {missing}")

    # Score fallback
    if "SCORE_gpp" not in out.columns:
        if "MED_final" in out.columns:
            out["SCORE_gpp"] = out["MED_final"].fillna(0)
        else:
            raise ValueError("Final GPP table missing SCORE_gpp and MED_final.")

    # Clean
    out = out.dropna(subset=["PLAYER", "TEAM", "POS", "SAL"]).copy()
    out = out[out["SAL"] > 0].copy()

    # Standardize DST label
    out.loc[out["POS"].isin(["DST", "D/ST", "DEF"]), "POS"] = "DST"

    # rank score with small ceiling bias
    ceiling_bump = 0.03 * out.get("FPTS_p90", pd.Series(0, index=out.index)).fillna(0)
    out["_score_rank"] = out["SCORE_gpp"].fillna(0) + ceiling_bump

    # ID for quick referencing
    out["ID"] = out["PLAYER"].astype(str) + " (" + out["TEAM"].astype(str) + ")"
    return out


# =====================
# Lineup data model
# =====================
@dataclass
class Lineup:
    idxs: List[int] = field(default_factory=list)  # store DF indices
    salary: int = 0

    def as_tuple(self) -> Tuple[int, ...]:
        return tuple(sorted(self.idxs))

    def names(self, df: pd.DataFrame) -> List[str]:
        return [df.loc[i, "PLAYER"] for i in self.idxs]

    def teams(self, df: pd.DataFrame) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for i in self.idxs:
            t = df.loc[i, "TEAM"]
            d[t] = d.get(t, 0) + 1
        return d

    def count_pos(self, df: pd.DataFrame, pos: str) -> int:
        return sum(1 for i in self.idxs if df.loc[i, "POS"] == pos)

    def total_score(self, df: pd.DataFrame) -> float:
        return float(df.loc[self.idxs, "SCORE_gpp"].sum())

    def total_median(self, df: pd.DataFrame) -> float:
        return float(df.loc[self.idxs, "MED_final"].sum())

    def to_row(self, df: pd.DataFrame) -> Dict[str, object]:
        names = [df.loc[i, "PLAYER"] for i in self.idxs]
        teams = [df.loc[i, "TEAM"] for i in self.idxs]
        poss  = [df.loc[i, "POS"] for i in self.idxs]
        sals  = [int(df.loc[i, "SAL"]) for i in self.idxs]
        row = {
            **{f"P{j+1}": v for j, v in enumerate(names)},
            **{f"T{j+1}": v for j, v in enumerate(teams)},
            **{f"Pos{j+1}": v for j, v in enumerate(poss)},
            **{f"Sal{j+1}": v for j, v in enumerate(sals)},
            "TotalSalary": int(self.salary),
            "TotalScore": round(self.total_score(df), 3),
            "TotalMedian": round(self.total_median(df), 3),
        }
        return row


# =====================
# Candidate pools & picker
# =====================

def build_slot_pools(df: pd.DataFrame, rules: RuleBook) -> Dict[str, List[int]]:
    pools: Dict[str, List[int]] = {}

    def eligible_for_slot(pos: str, slot: str) -> bool:
        if slot == "FLEX":
            return pos in rules.flex_pool
        return pos == slot

    for slot in rules.slots:
        pool = df.index[[eligible_for_slot(df.loc[i, "POS"], slot) for i in df.index]].tolist()
        # remove excluded players
        pool = [i for i in pool if df.loc[i, "PLAYER"] not in rules.exclude_list]
        # sort by our rank score
        pool = sorted(pool, key=lambda i: df.loc[i, "_score_rank"], reverse=True)
        pools[slot] = pool

    # Locks: ensure they exist in some slot pool
    for name in list(rules.lock_list):
        if name not in set(df["PLAYER"].tolist()):
            print(f"! Lock '{name}' not found in player pool, removing lock.")
            rules.lock_list.remove(name)

    return pools


def pick_candidate(
    df: pd.DataFrame,
    pool: List[int],
    used: Set[int],
    exposures: Dict[int, int],
    built: int,
    rules: RuleBook,
    salary_remaining: int,
    slots_left: int,
) -> Optional[int]:
    if slots_left <= 0:
        return None
    # crude per-slot budget guidance
    avg_budget = max(1, salary_remaining // slots_left)

    # mask pool
    cand: List[int] = []
    for i in pool:
        if i in used:
            continue
        # exposure cap
        if built > 0 and exposures[i] / max(1, built) > rules.max_exposure:
            continue
        # soft budget (allow a bit above)
        if df.loc[i, "SAL"] > avg_budget + 1500 and random.random() > 0.25:
            continue
        cand.append(i)

    if not cand:
        return None

    base = df.loc[cand, "_score_rank"].to_numpy(dtype=float)
    # stochastic noise
    sd = base.std() if base.std() > 0 else 1.0
    noise = np.random.normal(0.0, rules.temperature * sd, size=base.shape)
    weights = np.maximum(1e-6, base + noise)

    # fade chalk a bit
    if rules.col_own in df.columns:
        own = df.loc[cand, rules.col_own].to_numpy(dtype=float)
        weights = weights * (1.0 - rules.fade_chalk_weight * np.clip(own, 0.0, 1.0))

    probs = weights / weights.sum()
    return int(np.random.choice(cand, p=probs))


# =====================
# Constraints & stacks
# =====================

def enforce_basic_constraints(lineup: Lineup, df: pd.DataFrame, rules: RuleBook) -> bool:
    # team cap
    team_counts: Dict[str, int] = {}
    for i in lineup.idxs:
        t = df.loc[i, rules.col_team]
        team_counts[t] = team_counts.get(t, 0) + 1
    if any(v > rules.max_per_team for v in team_counts.values()):
        return False

    # salary floor
    if lineup.salary < rules.min_salary_used:
        return False

    # positional sanity
    pos_counts: Dict[str, int] = {}
    for i in lineup.idxs:
        p = df.loc[i, rules.col_pos]
        pos_counts[p] = pos_counts.get(p, 0) + 1

    # Count per slots (hard check) — we assume the builder fills legal slots already.
    # Optional: add max 1 QB, max 1 DST checks
    if pos_counts.get("QB", 0) != 1:
        return False
    if "DST" in rules.slots and pos_counts.get("DST", 0) != 1:
        return False

    return True


def satisfies_stacks(lineup: Lineup, df: pd.DataFrame, rules: RuleBook) -> bool:
    # locate QB
    qb_idx = None
    for i in lineup.idxs:
        if df.loc[i, rules.col_pos] == "QB":
            qb_idx = i
            break
    if qb_idx is None:
        return False

    qb_team = df.loc[qb_idx, rules.col_team]
    qb_opp = df.loc[qb_idx, rules.col_opp] if rules.col_opp in df.columns else None

    # teammate count (WR/TE)
    mates = 0
    for i in lineup.idxs:
        if i == qb_idx:
            continue
        if df.loc[i, rules.col_pos] in {"WR", "TE"} and df.loc[i, rules.col_team] == qb_team:
            mates += 1
    if mates < rules.qb_teammate_min:
        return False

    # bring-back (WR/TE from opponent)
    if rules.qb_bringback_min > 0 and qb_opp:
        bring = 0
        for i in lineup.idxs:
            if df.loc[i, rules.col_pos] in {"WR", "TE"} and df.loc[i, rules.col_team] == qb_opp:
                bring += 1
        if bring < rules.qb_bringback_min:
            return False

    # secondary stack (two non-QB players same team)
    if rules.secondary_stack_size >= 2:
        team_counts: Dict[str, int] = {}
        for i in lineup.idxs:
            if df.loc[i, rules.col_pos] == "QB":
                continue
            t = df.loc[i, rules.col_team]
            team_counts[t] = team_counts.get(t, 0) + 1
        if all(v < rules.secondary_stack_size for v in team_counts.values()):
            return False

    return True


# =====================
# Builder
# =====================

def build_lineups(df: pd.DataFrame, rules: RuleBook, n_lineups: int = 150) -> Tuple[List[Lineup], Dict[int, int]]:
    random.seed(rules.seed)
    np.random.seed(rules.seed)

    # Pre-filters
    pool = df.copy()

    # Locks & excludes
    if rules.lock_list:
        keep = set(rules.lock_list)
        missing = [n for n in keep if n not in set(pool["PLAYER"]) ]
        if missing:
            print(f"! Warning: locked players not in pool: {missing}")
    if rules.exclude_list:
        pool = pool[~pool["PLAYER"].isin(rules.exclude_list)].copy()

    # Optional floor/value filters
    if rules.min_floor_p25 is not None and rules.col_p25 in pool.columns:
        pool = pool[pool[rules.col_p25] >= rules.min_floor_p25].copy()
    if rules.min_value_med_per_1k is not None:
        value = pool[rules.col_med] / (pool[rules.col_salary] / 1000.0)
        pool = pool[value >= rules.min_value_med_per_1k].copy()

    # Ensure enough DSTs; if too few, drop DST slot entirely
    if "DST" in rules.slots and (pool[pool[rules.col_pos] == "DST"].shape[0] < 3):
        print("! Few DST options detected; removing DST slot for this run.")
        rules.slots = [s for s in rules.slots if s != "DST"]
        rules.flex_pool = {"RB", "WR", "TE"}

    # Build candidate pools per slot
    slot_pools = build_slot_pools(pool, rules)

    exposures: Dict[int, int] = {i: 0 for i in pool.index}
    built: List[Lineup] = []
    seen: Set[Tuple[int, ...]] = set()

    # For uniqueness control, we store each lineup set of players
    def unique_ok(new_idxs: List[int]) -> bool:
        if rules.min_uniques <= 1:
            return True
        new_set = set(new_idxs)
        for L in built:
            overlap = len(new_set.intersection(L.idxs))
            if len(new_idxs) - overlap < rules.min_uniques:  # not enough uniques
                return False
        return True

    max_tries = n_lineups * 400
    tries = 0

    while len(built) < n_lineups and tries < max_tries:
        tries += 1
        idxs: List[int] = []
        used: Set[int] = set()
        salary = 0
        ok = True

        for slot_i, slot in enumerate(rules.slots):
            salary_remaining = rules.salary_cap - salary
            slots_left = len(rules.slots) - slot_i
            cand = pick_candidate(
                pool, slot_pools[slot], used, exposures, len(built), rules,
                salary_remaining, slots_left
            )
            if cand is None:
                ok = False
                break
            idxs.append(cand)
            used.add(cand)
            salary += int(pool.loc[cand, rules.col_salary])

        if not ok:
            continue
        if salary > rules.salary_cap:
            continue
        lineup = Lineup(idxs=idxs, salary=salary)

        if not enforce_basic_constraints(lineup, pool, rules):
            continue
        if not satisfies_stacks(lineup, pool, rules):
            continue
        tup = lineup.as_tuple()
        if tup in seen:
            continue
        if not unique_ok(lineup.idxs):
            continue

        # Passed all checks
        built.append(lineup)
        seen.add(tup)
        for i in lineup.idxs:
            exposures[i] += 1

    return built, exposures


# =====================
# CLI & Output
# =====================

def main():
    ap = argparse.ArgumentParser(description="NFL DFS GPP optimizer using Final GPP table as inputs.")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to Final GPP CSV/Parquet from the app.")
    ap.add_argument("--out", dest="out_csv", default="out/lineups.csv", help="Lineups CSV output path.")
    ap.add_argument("--n", dest="n", type=int, default=150, help="Number of lineups (default 150)")
    ap.add_argument("--cap", dest="cap", type=int, default=50000, help="Salary cap (default 50000)")
    ap.add_argument("--maxexp", dest="maxexp", type=float, default=0.60, help="Max exposure per player (0..1)")
    ap.add_argument("--stack", dest="stack", default="1,0", help="QB stacks: qb_mates,bringback (e.g., 1,1)")
    ap.add_argument("--sec", dest="sec", type=int, default=0, help="Secondary stack size (0 disables)")
    ap.add_argument("--minsal", dest="minsal", type=int, default=49500, help="Min salary used (default 49500)")
    ap.add_argument("--uniques", dest="uniques", type=int, default=2, help="Min uniques between lineups (default 2)")
    ap.add_argument("--floor", dest="floor", type=float, default=None, help="Min p25 floor filter (optional)")
    ap.add_argument("--val", dest="val", type=float, default=None, help="Min MED per $1k filter (optional)")
    ap.add_argument("--temp", dest="temp", type=float, default=0.10, help="Sampling randomness 0..1")
    ap.add_argument("--seed", dest="seed", type=int, default=7, help="Random seed")
    ap.add_argument("--locks", dest="locks", default=None, help="Comma-separated player names to lock")
    ap.add_argument("--xout", dest="xout", default=None, help="Comma-separated player names to exclude")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # RuleBook from args
    rules = RuleBook()
    rules.salary_cap = args.cap
    rules.max_exposure = args.maxexp
    rules.secondary_stack_size = args.sec
    rules.min_salary_used = args.minsal
    rules.min_uniques = args.uniques
    rules.min_floor_p25 = args.floor
    rules.min_value_med_per_1k = args.val
    rules.temperature = args.temp
    rules.seed = args.seed
    try:
        mates, bb = args.stack.split(",")
        rules.qb_teammate_min = int(mates)
        rules.qb_bringback_min = int(bb)
    except Exception:
        pass

    if args.locks:
        rules.lock_list = set([s.strip() for s in args.locks.split(",") if s.strip()])
    if args.xout:
        rules.exclude_list = set([s.strip() for s in args.xout.split(",") if s.strip()])

    # Load Final GPP table
    df = load_final_table(args.in_path)

    # If locks exist, ensure they are in the pool
    if rules.lock_list:
        missing = [n for n in rules.lock_list if n not in set(df["PLAYER"]) ]
        if missing:
            print(f"! Locked players missing from Final GPP inputs: {missing}")

    # Build lineups
    lineups, exposures = build_lineups(df, rules, n_lineups=args.n)

    if not lineups:
        print("No lineups built. Try relaxing constraints or verify inputs.")
        return

    # Export lineups
    out_rows = [lu.to_row(df) for lu in lineups]
    out_df = pd.DataFrame(out_rows).sort_values("TotalScore", ascending=False).reset_index(drop=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"✓ Wrote {len(out_df)} lineups -> {os.path.abspath(args.out_csv)}")

    # Exposure report
    exp_tbl = (
        pd.DataFrame({"idx": list(exposures.keys()), "Count": list(exposures.values())})
        .assign(PLAYER=lambda d: d["idx"].map(df["PLAYER"].to_dict()))
        .assign(ExposurePct=lambda d: 100.0 * d["Count"] / len(lineups))
        .sort_values(["Count", "PLAYER"], ascending=[False, True])
        .reset_index(drop=True)
        [["PLAYER", "Count", "ExposurePct"]]
    )
    exp_path = os.path.splitext(args.out_csv)[0] + "_exposure.csv"
    exp_tbl.to_csv(exp_path, index=False)
    print(f"✓ Wrote exposure -> {os.path.abspath(exp_path)}")

    # Quick print
    print("\nTop 10 exposures:")
    print(exp_tbl.head(10).to_string(index=False))

    # BLUEPRINT HOOKS — as you post sections we will:
    #  - Add pre-filters (player archetypes, salary tiers, leverage gates)
    #  - Extend candidate sampling with group logic / mutually exclusive rules
    #  - Enforce stack variants (QB+2, bringback 1, secondary stack WR+TE, etc.)
    #  - Add team/game constraints, late-swap modes, contest-size presets


if __name__ == "__main__":
    main()
