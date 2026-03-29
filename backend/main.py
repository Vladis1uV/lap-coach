import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from processing.analysis_combined import get_all_recommendations

app = FastAPI(title="Lap Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_data(
    file_fast: UploadFile = File(...),
    file_good: UploadFile = File(...),
):
    """Receive two MCAP files and return recommendations."""
    tmp_dir = tempfile.mkdtemp()
    fast_path = os.path.join(tmp_dir, "fast.mcap")
    good_path = os.path.join(tmp_dir, "good.mcap")

    try:
        with open(fast_path, "wb") as f:
            f.write(await file_fast.read())
        with open(good_path, "wb") as f:
            f.write(await file_good.read())

        raw_recommendations = get_all_recommendations(fast_path, good_path)

        recommendations = []
        for rec in raw_recommendations:
            entry = {
                "recommendation": rec.recommendation,
                "verdict": rec.verdict.name if hasattr(rec.verdict, "name") else str(rec.verdict),
            }
            # Arc / position info
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

            # Category
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

        return {"recommendations": recommendations}
    finally:
        # Cleanup temp files
        for p in (fast_path, good_path):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp_dir)
