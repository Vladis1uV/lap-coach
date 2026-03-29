import tempfile
import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from analysis_combined import get_all_recommendations

app = FastAPI(title="Lap Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store plot directories keyed by session id so we can serve them later
_plot_dirs: dict[str, str] = {}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/plots/{session_id}/{filename}")
async def get_plot(session_id: str, filename: str):
    """Serve a saved plot image."""
    plot_dir = _plot_dirs.get(session_id)
    if not plot_dir:
        return {"error": "session not found"}
    path = os.path.join(plot_dir, filename)
    if not os.path.exists(path):
        return {"error": "file not found"}
    return FileResponse(path, media_type="image/png")


@app.post("/api/process")
async def process_data(
    file_fast: UploadFile = File(...),
    file_good: UploadFile = File(...),
):
    """Receive two MCAP files and return recommendations + plot URLs."""
    tmp_dir = tempfile.mkdtemp()
    fast_path = os.path.join(tmp_dir, "fast.mcap")
    good_path = os.path.join(tmp_dir, "good.mcap")

    # Create a plots directory that persists until the next analysis
    session_id = uuid.uuid4().hex[:12]
    plots_dir = tempfile.mkdtemp(prefix="plots_")
    _plot_dirs[session_id] = plots_dir

    try:
        with open(fast_path, "wb") as f:
            f.write(await file_fast.read())
        with open(good_path, "wb") as f:
            f.write(await file_good.read())

        raw_recommendations, plot_paths = get_all_recommendations(
            fast_path, good_path, save_dir=plots_dir
        )

        recommendations = []
        for rec in raw_recommendations:
            entry = {
                "recommendation": rec.recommendation,
                "verdict": rec.verdict.name if hasattr(rec.verdict, "name") else str(rec.verdict),
            }
            if hasattr(rec, "arc_start"):
                entry["arc_start"] = round(rec.arc_start, 2)
            if hasattr(rec, "start_arc"):
                entry["arc_start"] = round(rec.start_arc, 2)
            if hasattr(rec, "arc_end"):
                entry["arc_end"] = round(rec.arc_end, 2)
            if hasattr(rec, "end_arc"):
                entry["arc_end"] = round(rec.end_arc, 2)
            if hasattr(rec, "mean_delta"):
                entry["mean_delta"] = round(rec.mean_delta, 4)
            if hasattr(rec, "offset_m") and rec.offset_m is not None:
                entry["offset_m"] = round(rec.offset_m, 2)

            type_name = type(rec).__name__
            if "Throttle" in type_name or "Gas" in type_name:
                entry["category"] = "throttle"
            elif "Brake" in type_name:
                entry["category"] = "brake"
            elif "Steering" in type_name:
                entry["category"] = "steering"
            else:
                entry["category"] = "other"

            recommendations.append(entry)

        # Build plot URLs
        plots = {}
        for name, path in plot_paths.items():
            filename = os.path.basename(path)
            plots[name] = f"/api/plots/{session_id}/{filename}"

        return {"recommendations": recommendations, "plots": plots}
    finally:
        for p in (fast_path, good_path):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp_dir)
