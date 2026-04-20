"""
macul.ai — Chart Summarizer
Uses a fine-tuned LLM to summarize ophthalmology chart notes
and flag clinical concerns.
"""

from transformers import pipeline
from typing import Optional
import torch


SYSTEM_PROMPT = """You are an expert ophthalmology AI assistant.
Analyze the following patient chart notes and provide:
1. A concise clinical summary (2-3 sentences)
2. Any flags or concerns requiring attention
3. Recommended next steps
4. Risk level: low, medium, or high

Focus on: AMD progression, IOP trends, diabetic retinopathy,
injection response, post-op status, and medication compliance."""


class ChartSummarizer:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ChartSummarizer] Loading on {self.device}")
        # TODO: Load fine-tuned macul.ai model from Hugging Face
        self.ready = False
        print("[ChartSummarizer] Model placeholder — connect fine-tuned model")

    def analyze(
        self,
        notes: str,
        diagnosis: Optional[str] = None,
        medications: Optional[list] = None,
        history: Optional[list] = None
    ) -> dict:
        return {
            "summary": f"Patient chart analyzed. Diagnosis: {diagnosis or 'Not specified'}. "
                       f"AI model integration pending.",
            "flags": [
                {"level": "info", "message": "AI model not yet connected"},
                {"level": "info", "message": "Fine-tune LLaMA 3 on ophthalmic notes to activate"}
            ],
            "recommendations": [
                "Connect fine-tuned language model",
                "Integrate NexTech IntelleChart API for live chart data"
            ],
            "risk_level": "low"
        }
