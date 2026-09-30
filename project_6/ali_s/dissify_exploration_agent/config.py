"""
Configuration settings for Dissify Exploration Agent.
Supports local execution via Ollama and portable paths across Windows and Linux.
"""

import os
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = AGENT_DIR.parent
# REPO_DIR = SCRIPTS_DIR.parent

XML_PATH = Path(os.getenv(
    "DISSIFY_XML_PATH",
    AGENT_DIR / "input_dataset"  / "merged_dissertations.xml"
))

CV_TIMELINE_PATH = Path(os.getenv(
    "DISSIFY_CV_PATH",
    AGENT_DIR / "input_dataset"  / "cv_extracted_timeline.json"
))

TEXT_INDIVIDUAL_DIR = Path(os.getenv(
    "DISSIFY_TEXT_DIR",
    SCRIPTS_DIR / "data" / "text_results_individual"
))

# Ollama model configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama Context and Generation Limits
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))


LOGS_DIR = AGENT_DIR / "logs"

# Agent execution limits bounded autonomy
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "12"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "3000"))
