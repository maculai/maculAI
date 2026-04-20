# macul.ai — System Architecture

## Overview
macul.ai is a specialized AI platform for ophthalmology practices.
Hybrid local/cloud system — local AI inference on practice hardware,
cloud backup and sync for data.

## Components

### 1. Landing Page (macul.ai)
- GitHub Pages hosted
- Static HTML/CSS/JS
- Early access via griffin@macul.ai

### 2. Clinical Platform (platform/)
- Ophthalmologist View: chart analysis, OCT comparison, fundus AI, injection log
- Technician View: drop regimen, patient Q&A, pre-visit briefing
- Connects to API backend for live AI analysis

### 3. AI Backend (api/)
- FastAPI Python server
- Runs locally on practice hardware (RTX 4070 or similar)
- Endpoints:
  - POST /analyze/chart — clinical note summarization
  - POST /analyze/fundus — fundus photo pathology detection
  - POST /analyze/oct — OCT fluid quantification

### 4. AI Models (ai/)
- Language model: LLaMA 3 8B fine-tuned on ophthalmology notes via LoRA
- Vision model: EfficientNet-B3 trained on RFMiD/IDRiD datasets
- RAG pipeline: ChromaDB + sentence-transformers + PubMed literature

## Integration Points

### NexTech IntelleChart
- REST API integration for live chart data
- Bidirectional sync — AI writes summaries back to chart
- Auth: API key per practice

### Optos Ultra-Widefield
- DICOM file ingestion pipeline
- Automated analysis on scan upload
- Findings overlaid on image in platform UI

## Data Flow
```
NexTech Chart Data
        ↓
  macul.ai API
        ↓
  LLM + Vision Analysis
        ↓
  Clinical Platform (MD + Tech Views)
        ↓
  Ophthalmologist Reviews + Acts
```

## Security
- All patient data stays on-premise
- No PHI transmitted to external services
- HIPAA-compliant architecture by design
- TLS encryption for all API calls

## Deployment
- Development: local RTX 4070 machine
- Production: on-premise server per clinic OR HIPAA-compliant cloud (AWS GovCloud)
