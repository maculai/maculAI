"""
macul.ai — FastAPI Backend
Serves AI model predictions to the clinical platform.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from models.chart_summarizer import ChartSummarizer
from models.image_analyzer import ImageAnalyzer

app = FastAPI(
    title="macul.ai API",
    description="AI backend for ophthalmology clinical platform",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chart_model = ChartSummarizer()
image_model = ImageAnalyzer()


class ChartRequest(BaseModel):
    patient_id: str
    chart_notes: str
    diagnosis: Optional[str] = None
    medications: Optional[list] = []
    visit_history: Optional[list] = []

class ChartResponse(BaseModel):
    summary: str
    flags: list
    recommendations: list
    risk_level: str

class ImageResponse(BaseModel):
    findings: list
    confidence_scores: dict
    flag_count: int
    requires_urgent_review: bool


@app.get("/")
def root():
    return {"status": "macul.ai API running", "version": "0.1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze/chart", response_model=ChartResponse)
async def analyze_chart(request: ChartRequest):
    """Analyze patient chart notes and return AI summary + flags."""
    try:
        result = chart_model.analyze(
            notes=request.chart_notes,
            diagnosis=request.diagnosis,
            medications=request.medications,
            history=request.visit_history
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/fundus", response_model=ImageResponse)
async def analyze_fundus(file: UploadFile = File(...)):
    """Analyze fundus photo for pathology detection."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="JPEG or PNG only")
    try:
        image_bytes = await file.read()
        result = image_model.analyze_fundus(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/oct", response_model=ImageResponse)
async def analyze_oct(file: UploadFile = File(...)):
    """Analyze OCT scan for fluid quantification and layer analysis."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="JPEG or PNG only")
    try:
        image_bytes = await file.read()
        result = image_model.analyze_oct(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
