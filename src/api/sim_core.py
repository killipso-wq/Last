"""
Minimal simulation orchestration stub.
Replace simulate batch with your real simulator later.
"""
import multiprocessing as mp
from pathlib import Path
import os, json, time
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))
_jobs = {}

def _status_path(job_id: str) -> Path:
    return ARTIFACTS_DIR / job_id / "status.json"

def _write_status(job_id: str, payload: dict):
    path = _status_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def job_status(job_id: str):
    p = _status_path(job_id)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)

def list_jobs():
    jobs = []
    if ARTIFACTS_DIR.exists():
        for d in ARTIFACTS_DIR.iterdir():
            if not d.is_dir(): continue
            sf = d / "status.json"
            if sf.exists():
                with open(sf, "r") as f:
                    jobs.append(json.load(f))
    return jobs

def start_simulation_background(job_id: str, players_csv_path: str):
    proc = mp.Process(target=_run_simulation_job, args=(job_id, players_csv_path), daemon=True)
    proc.start()
    _jobs[job_id] = {"pid": proc.pid, "status": "running"}
    return proc.pid

def _run_simulation_job(job_id: str, players_csv_path: str):
    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    status = {"job_id": job_id, "status": "running", "progress": 0.0, "message": "started", "result_file": None}
    _write_status(job_id, status)
    try:
        players_df = pd.read_csv(players_csv_path)
        attributes_file = ARTIFACTS_DIR / "attribute_table.parquet"
        if not attributes_file.exists():
            status.update({"status":"error", "message":"attribute_table.parquet missing in artifacts root"})
            _write_status(job_id, status)
            return
        attr_df = pd.read_parquet(attributes_file)
        seed = abs(hash(job_id)) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        MC_RUNS = 1000
        BATCH = 100
        player_fpts = {}
        for run_start in range(0, MC_RUNS, BATCH):
            for i in range(BATCH):
                for _, row in players_df.iterrows():
                    name = f"{row.get('PLAYER')}-{row.get('TEAM')}"
                    score = float(rng.normal(loc=10.0, scale=6.0))
                    player_fpts.setdefault(name, []).append(score)
            progress = (run_start + BATCH) / MC_RUNS
            status.update({"progress": min(progress, 1.0), "message": f"ran {run_start + BATCH} / {MC_RUNS}"})
            _write_status(job_id, status)
        rows = []
        for player, scores in player_fpts.items():
            arr = np.array(scores)
            rows.append({"PLAYER": player, "FPTS_MEAN": float(arr.mean()), "FPTS_STD": float(arr.std())})
        out_df = pd.DataFrame(rows)
        out_path = job_dir / "final_results.csv"
        out_df.to_csv(out_path, index=False)
        status.update({"status":"finished","progress":1.0,"message":"completed","result_file":str(out_path.name)})
        _write_status(job_id, status)
    except Exception as e:
        status.update({"status":"error","message":str(e)})
        _write_status(job_id, status)
        raise
