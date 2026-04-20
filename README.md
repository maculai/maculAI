# macul.ai

AI-powered clinical platform for ophthalmology practices.

## What this is
macul.ai integrates with existing EHR systems (NexTech IntelleChart) and imaging platforms (Optos) to provide:
- AI chart summarization and clinical flagging
- OCT scan comparison and fluid quantification
- Fundus photo analysis (retinal tear detection, CNV, drusen mapping)
- Ophthalmologist and technician views
- Drop regimen guidance and patient communication tools

## Stack
- Frontend: HTML/CSS/JS (GitHub Pages)
- Backend: Python + FastAPI
- AI: PyTorch + Hugging Face Transformers
- Vision: EfficientNet fine-tuned on ophthalmic imaging data
- LLM: LLaMA 3 fine-tuned on clinical ophthalmology notes

## Structure
- `index.html` — public landing page (macul.ai)
- `platform/` — clinical dashboard (MD + tech views)
- `api/` — Python AI backend
- `ai/` — model training and fine-tuning scripts
- `docs/` — architecture and integration documentation

## Contact
griffin@macul.ai
