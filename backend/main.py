from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from processing.parser import parse_lap_data

app = FastAPI(title="Lap Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_data(file: UploadFile = File(...)):
    """Receive a data file from the frontend and process it."""
    contents = await file.read()
    result = parse_lap_data(contents)
    return {"result": result}
