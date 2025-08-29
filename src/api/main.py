from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uuid, os, shutil
from pathlib import Path
from sim_core import start_simulation_background, job_status, list_jobs

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")
Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Football Monte Carlo Simulator API")

@app.post("/upload-players")
async def upload_players(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")
    job_id = str(uuid.uuid4())
    job_dir = Path(ARTIFACTS_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / "players.csv"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    start_simulation_background(job_id, str(file_path))
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    s = job_status(job_id)
    if s is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(content=s)

@app.get("/download/{job_id}")
def download_result(job_id: str):
    job_dir = Path(ARTIFACTS_DIR) / job_id
    result_file = job_dir / "final_results.csv"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="result not ready")
    return FileResponse(str(result_file), filename="final_results.csv", media_type="text/csv")

@app.get("/jobs")
def jobs():
    return JSONResponse(content={"jobs": list_jobs()})
