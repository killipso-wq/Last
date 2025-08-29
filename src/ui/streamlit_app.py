import streamlit as st
import requests, os, time

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
st.set_page_config(page_title="Football Monte Carlo Simulator", layout="wide")
st.title("Football Monte Carlo Simulator — Single-User")
uploaded = st.file_uploader("players.csv", type=["csv"])
if uploaded is not None:
    with st.spinner("Uploading and starting simulation..."):
        files = {"file": ("players.csv", uploaded.getvalue(), "text/csv")}
        r = requests.post(f"{API_BASE}/upload-players", files=files)
        if r.status_code != 200:
            st.error(f"Upload failed: {r.text}")
        else:
            job_id = r.json().get("job_id")
            st.success(f"Job started: {job_id}")
            status_box = st.empty()
            download_box = st.empty()
            while True:
                time.sleep(1.0)
                s = requests.get(f"{API_BASE}/status/{job_id}")
                if s.status_code != 200:
                    status_box.error(f"Failed to fetch status: {s.text}")
                    break
                data = s.json()
                status_box.markdown(f\"**Status:** {data.get('status')}  \n**Progress:** {data.get('progress'):.2%}  \n**Message:** {data.get('message')}\")
                if data.get("status") == "finished":
                    download_url = f"{API_BASE}/download/{job_id}"
                    download_box.markdown(f"[Download results]({download_url})")
                    break
                if data.get("status") == "error":
                    status_box.error(f"Job error: {data.get('message')}")
                    break
