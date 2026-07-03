# backend/app.py
"""
Smart Route Service — Flask API Server
---------------------------------------
This file defines the HTTP API only.
All AI, storage, escalation, and email logic lives in separate modules.

Endpoints:
  GET  /                → Serve the chat frontend (index.html)
  POST /api/session     → Start a new conversation session
  POST /api/chat        → Process a customer message
  POST /api/upload      → Upload and index a knowledge base document
  GET  /api/status      → Health check
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from openai import OpenAI

from config import (
    UPLOAD_FOLDER,
    BUSINESS_NAME,
    BUSINESS_HOURS,
    ESCALATION_HIERARCHY,
    CONFIDENCE_THRESHOLD,
    CHAT_MODEL,
    GROQ_API_KEY,
    GMAIL_SENDER,
    GMAIL_APP_PASSWORD,
)
from rag_engine import load_index, build_index_from_file, retrieve, is_confident, generate_answer
from escalation_engine import detect_emotional_signal, assess_seriousness
from gmail_service import send_escalation_email
from conversation_store import (
    create_session,
    append_message,
    get_transcript,
    log_escalation,
)

# ─── Flask App Setup ──────────────────────────────────────────────────────────

# Serve frontend files from ../frontend/ relative to this file
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,    # Where to find static files
    static_url_path="",            # Serve them at the root URL path
)
CORS(app)   # Allow cross-origin requests (needed if frontend is opened separately)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # Ensure uploads/ exists

# ─── Credential Integrity Check ────────────────────────────────────────────────
is_api_key_placeholder = not GROQ_API_KEY or "your_groq" in GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_placeholder")
is_gmail_placeholder = not GMAIL_APP_PASSWORD or "your_gmail" in GMAIL_APP_PASSWORD

if is_api_key_placeholder or is_gmail_placeholder:
    print("\n" + "!" * 80)
    print("  CRITICAL SYSTEM ALERT: MISSING OR PLACEHOLDER CREDENTIALS DETECTED")
    if is_api_key_placeholder:
        print("  - GROQ_API_KEY is missing, empty, or using a placeholder value.")
    if is_gmail_placeholder:
        print("  - GMAIL_APP_PASSWORD is missing, empty, or using a placeholder value.")
    print("  The application will start, but will fall back to local rule-based engines.")
    print("!" * 80 + "\n")

ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}

# Load the FAISS index at startup (if it already exists from a previous session)
load_index()

# ─── Route Helpers ────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _do_escalation(
    session_id: str,
    level: int,
    label: str,
    trigger: str,
    transcript: list,
) -> str:
    """
    Central escalation handler:
      1. Send email to the correct hierarchy level
      2. Log the escalation to escalations.json
    Returns the label of the escalation target for the bot's reply.
    """
    try:
        send_escalation_email(level, session_id, transcript, trigger, label)
    except Exception as e:
        # Email failure should not break the chat — log and continue
        print(f"[App] Email send failed: {e}")

    log_escalation(session_id, level, trigger, label)
    return ESCALATION_HIERARCHY[level]["label"]

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def serve_frontend():
    """Serve index.html from the frontend/ folder."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status", methods=["GET"])
def status():
    """Simple health check endpoint."""
    return jsonify({"status": "online", "business": BUSINESS_NAME})


@app.route("/api/session", methods=["POST"])
def new_session():
    """
    Create a new conversation session.
    Called once by the frontend when the page loads.

    Response: { "session_id": "uuid-string" }
    """
    sid = create_session()
    return jsonify({"session_id": sid})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint. Processes one customer message.

    Request body:
      { "session_id": "...", "message": "customer text here" }

    Response body:
      {
        "response"         : "bot reply text",
        "escalated"        : true/false,
        "escalation_level" : 1–4 or null,
        "escalation_label" : "Support Manager" or null
      }
    """
    try:
        body = request.get_json(force=True)
        sid = (body.get("session_id") or "").strip()
        message = (body.get("message") or "").strip()

        if not sid:
            return jsonify({"error": "session_id is required."}), 400
        if not message:
            return jsonify({"error": "message cannot be empty."}), 400

        # ── 1. Persist the user's message ──────────────────────────────────────────
        append_message(sid, "user", message)
        transcript = get_transcript(sid)
        history = transcript[:-1]   # All messages EXCEPT the one just added

        # ── 2. Emotional/urgency signal detection (ALWAYS runs) ────────────────────
        signal_found, signal_reason = detect_emotional_signal(history, message)

        if signal_found:
            trigger = f"Emotional/urgency signal: {signal_reason}"
            level, label = assess_seriousness(history, message, trigger)
            _do_escalation(sid, level, label, trigger, transcript)

            reply = (
                f"I can see you're having a really difficult experience, and I'm truly sorry. "
                f"I've immediately escalated your case to our {label}. "
                f"A member of our team will follow up with you very shortly."
            )
            append_message(sid, "assistant", reply)
            return jsonify({
                "response": reply,
                "escalated": True,
                "escalation_level": level,
                "escalation_label": label,
            })

        # ── 3. RAG retrieval ────────────────────────────────────────────────────────
        chunks, score = retrieve(message)

        if is_confident(score):
            # We found a reliable answer — respond directly
            reply = generate_answer(message, chunks, history)
            append_message(sid, "assistant", reply)
            return jsonify({
                "response": reply,
                "escalated": False,
                "escalation_level": None,
                "escalation_label": None,
            })

        # ── 3.5. Conversational fallback — handle greetings & casual chat ──────────
        #
        # If the RAG score is low, the message might be a greeting, small talk,
        # or general conversation — NOT a support question. In that case, respond
        # conversationally instead of escalating.

        if _is_casual_message(message):
            reply = _generate_conversational_reply(message, history)
            append_message(sid, "assistant", reply)
            return jsonify({
                "response": reply,
                "escalated": False,
                "escalation_level": None,
                "escalation_label": None,
            })

        # ── 4. Knowledge gap → escalate ─────────────────────────────────────────────
        trigger = (
            f"Knowledge gap — no confident answer found in knowledge base "
            f"(retrieval score: {score:.2f}, threshold: {CONFIDENCE_THRESHOLD})"
        )
        level, label = assess_seriousness(history, message, trigger)
        _do_escalation(sid, level, label, trigger, transcript)

        reply = (
            f"I don't have the specific information needed to answer your question confidently. "
            f"I've connected you with our {label}, and someone will reach out to you shortly."
        )
        append_message(sid, "assistant", reply)
        return jsonify({
            "response": reply,
            "escalated": True,
            "escalation_level": level,
            "escalation_label": label,
        })
    except Exception as e:
        print(f"[App] Exception in /api/chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "The server encountered an error processing your request. Please try again later.",
            "details": str(e)
        }), 500


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Upload and index a knowledge base document.
    Accepts: .txt, .pdf, .docx files via multipart form-data (key = "file")

    Response: { "message": "success text" } or { "error": "..." }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file found in request. Use key 'file'."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"'{file.filename}' is not a supported file type. Use .txt, .pdf, or .docx"
        }), 400

    # Save to uploads/
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # Index the document (embed chunks and add to FAISS)
    try:
        build_index_from_file(save_path)
    except Exception as e:
        return jsonify({"error": f"Failed to index document: {str(e)}"}), 500

    return jsonify({
        "message": f"'{file.filename}' uploaded and indexed successfully."
    })


# ─── Conversational Helpers ───────────────────────────────────────────────────

_llm_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def _is_casual_message(message: str) -> bool:
    """
    Use the LLM to determine if a message is casual/greeting/small-talk
    (returns True) or a genuine support question (returns False).
    """
    prompt = f"""Classify the following customer message into one of two categories:

1. CASUAL — greetings, small talk, thank you, goodbye, pleasantries, general
   conversation, asking how you are, saying okay/alright, or any message that
   is NOT asking for specific help with a product, service, or technical issue.

2. SUPPORT — a genuine question or request about internet service, router setup,
   billing, account issues, technical problems, service outages, or any specific
   support topic.

Customer message: \"{message}\"

Respond with ONLY one word: CASUAL or SUPPORT"""

    try:
        response = _llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().upper()
        return "CASUAL" in result
    except Exception as e:
        print(f"[App] Casual message check failed: {e}")
        # Default to False (treat as support question) — safer to not miss a real query
        return False


def _generate_conversational_reply(message: str, history: list) -> str:
    """
    Generate a friendly, conversational response for greetings and casual chat.
    Does NOT claim any knowledge base information.
    """
    system_prompt = f"""You are a friendly and professional customer support assistant for {BUSINESS_NAME}, a home internet and router setup service operating in Mumbai, Kalyan, and Thane.

The customer is making casual conversation (greeting, small talk, thanks, goodbye, etc.).
Respond warmly and naturally in 1–2 sentences.
- If they're greeting you, greet them back and ask how you can help.
- If they're saying thanks, acknowledge it warmly and ask if there's anything else.
- If they're saying goodbye, wish them well.
- If they're asking how you are, respond positively and offer to help.
- Keep the tone friendly, warm, and professional.
- Do NOT make up any service information or policies.
- Business hours: {BUSINESS_HOURS}"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        response = _llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[App] Conversational reply failed: {e}")
        return (
            "Hi there! 👋 Welcome to Smart Route Service support. "
            "How can I help you today?"
        )


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  {BUSINESS_NAME} Support Chatbot")
    print(f"  Open http://localhost:5000 in your browser")
    print(f"{'='*60}\n")
    app.run(debug=True, port=5000)
