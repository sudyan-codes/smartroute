# backend/config.py
"""Central configuration module for the Smart Route Chatbot backend.
All other modules should import settings from here instead of hard‑coding values.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root (one level above backend/)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ─── Groq ────────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
CHAT_MODEL       = "llama-3.3-70b-versatile"  # Groq LLaMA 3.3 model
MAX_TOKENS       = 500                        # Max tokens in bot's reply

# ─── RAG Settings ────────────────────────────────────────────────────────────
CHUNK_SIZE           = 800    # Characters per chunk (keeps chunks within model token limits)
CHUNK_OVERLAP        = 100    # Overlap prevents answers from being cut at chunk boundaries
TOP_K_CHUNKS         = 3      # Retrieve the 3 most relevant chunks per query
CONFIDENCE_THRESHOLD = 0.35   # Cosine similarity score; below this = check casual or escalate

# ─── File Paths ───────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))   # = /path/to/smart_route_chatbot/backend

DATA_DIR             = os.path.join(_BASE, "..", "data")
FAISS_INDEX_DIR      = os.path.join(DATA_DIR, "faiss_index")
CONVERSATIONS_FILE   = os.path.join(DATA_DIR, "conversations.json")
ESCALATIONS_FILE     = os.path.join(DATA_DIR, "escalations.json")
CHUNKS_METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "chunks_metadata.json")
UPLOAD_FOLDER        = os.path.join(_BASE, "..", "uploads")

# ─── Gmail ───────────────────────────────────────────────────────────────────
GMAIL_SENDER       = os.getenv("GMAIL_SENDER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# ─── Business Identity ────────────────────────────────────────────────────────
BUSINESS_NAME  = "Smart Route Service"
BUSINESS_HOURS = "Monday to Saturday, 9:00 AM – 7:00 PM (Closed Sundays)"

# ─── Escalation Hierarchy ─────────────────────────────────────────────────────
ESCALATION_HIERARCHY = {
    1: {
        "label"      : "Tier 1 Support",
        "email"      : os.getenv("EMAIL_LEVEL_1"),
        "description": "General query the bot cannot answer. No urgency or emotion detected."
    },
    2: {
        "label"      : "Senior Support Agent",
        "email"      : os.getenv("EMAIL_LEVEL_2"),
        "description": "Mild frustration. Customer is dissatisfied but not hostile."
    },
    3: {
        "label"      : "Support Manager",
        "email"      : os.getenv("EMAIL_LEVEL_3"),
        "description": "Clear anger, repeated failed attempts, or financial/legal concern."
    },
    4: {
        "label"      : "Business Director",
        "email"      : os.getenv("EMAIL_LEVEL_4"),
        "description": "Abusive language, threats, safety concern, or business-wide issue."
    }
}
