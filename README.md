# Lap Coach 🏎️

A racing lap analysis tool that compares two MCAP telemetry recordings — a **fast (reference) lap** and a **good lap (to tune)** — and provides actionable coaching recommendations on **throttle**, **braking**, and **steering** technique.

## How It Works

1. **Upload** two `.mcap` files via the web UI
2. The **Python backend** parses telemetry data (throttle, brake, steering vs. arc position), aligns the fast reference lap with the provided by position on the track, and runs three analysis modules:
   - **Throttle analysis** — detects plateau boundaries and level differences of throttle application
   - **Brake analysis** — same approach for braking zones
   - **Steering analysis** — identifies the differences of angle of steering 
3. The backend returns a list of recommendations with verdicts (e.g. "brake earlier", "less throttle in this arc range")
4. The **React frontend** displays results grouped by category

## Project Structure

```
├── backend/            # FastAPI server + Python analysis
│   ├── main.py         # API endpoints (/api/process, /api/health)
│   ├── requirements.txt
│   ├── parser.py
│   ├── gas_analysis.py
│   ├── brake_analysis.py
│   ├── steering_analysis.py
│   └── analysis_combined.py
├── src/                # React + Vite frontend
├── data/               # Local telemetry data (git-ignored)
└── README.md
```

## Running Locally

### 1. Clone

```bash
git clone https://github.com/Vladis1uV/lap-coach.git
cd lap-coach
```

### 2. Start the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Verify: `http://localhost:8000/api/health` → `{"status": "ok"}`

### 3. Start the frontend

```bash
# From project root
npm install
npm run dev
```

Opens at `http://localhost:5173`.

### 4. Use

1. Open `http://localhost:5173`
2. Upload a **fast lap** and a **good lap** (`.mcap` format)
3. Click **Analyze Lap Data**
4. View recommendations on the results page
