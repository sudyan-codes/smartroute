# backend/escalation_engine.py
"""
Escalation Engine
-----------------
Two functions:

1. detect_emotional_signal()
   Runs on EVERY customer message, even when a RAG answer is found.
   If the customer shows anger, frustration, or demands a human,
   this overrides the normal response flow.

2. assess_seriousness()
   Runs only when escalation is triggered.
   Reads the full conversation and assigns Level 1–4.
"""

import json
from typing import Tuple
from openai import OpenAI

from config import GROQ_API_KEY, CHAT_MODEL, ESCALATION_HIERARCHY, BUSINESS_NAME

_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def detect_emotional_signal(history: list, latest_message: str) -> Tuple[bool, str]:
    """
    Determine whether the customer's latest message (or conversation history)
    contains an emotional or urgency signal that requires escalation.

    Signals include:
    - Anger, hostility, or abusive language
    - Mild or strong frustration
    - Direct request to speak with a human or manager
    - Urgency ("right now", "immediately", "this is the third time")
    - Legal threats or complaint threats
    - Safety concerns

    Returns:
        (signal_detected: bool, reason: str)
        reason is a short explanation used in the escalation email.
    """
    history_text = _format_history(history[-6:])   # Only recent history to save tokens

    prompt = f"""You are an emotional signal detector for {BUSINESS_NAME}'s customer support chatbot.

Analyze the conversation and determine if the customer's latest message contains a CLEAR, GENUINE escalation signal.

IMPORTANT — Do NOT flag any of the following as signals:
- Greetings (hi, hello, hey, good morning, good evening, etc.)
- Polite conversation or small talk (how are you, thanks, bye, okay, etc.)
- Calm questions about services, billing, or technical issues
- General curiosity or informational queries
- Expressions of mild inconvenience that are still polite in tone

Only flag these REAL escalation signals:
1. Clear anger or hostility (e.g. "this is absolutely unacceptable", "you people are useless")
2. Sustained or escalating frustration across multiple messages (NOT a single mild remark)
3. An explicit request to speak with a human, agent, or manager
4. Urgent demands with emotional intensity (e.g. "I need this fixed RIGHT NOW or else")
5. Threats: legal action, filing a complaint, posting on social media
6. Safety concerns (electrical hazard, equipment danger, etc.)

When in doubt, return signal_detected: false. It is better to continue the conversation than to escalate unnecessarily.

Conversation history (most recent first):
{history_text}

Latest customer message:
\"{latest_message}\"

Respond with ONLY a JSON object. No explanation, no markdown, no other text:
{{"signal_detected": true, "reason": "short description"}}
or
{{"signal_detected": false, "reason": ""}}"""

    try:
        raw = _call_llm(prompt, max_tokens=80)
        result = json.loads(raw)
        return bool(result.get("signal_detected", False)), result.get("reason", "")
    except Exception as e:
        print(f"[Escalation] Error checking emotional signal: {e}")
        return _fallback_detect_emotional_signal(latest_message)


def assess_seriousness(
    history: list,
    latest_message: str,
    trigger_reason: str,
) -> Tuple[int, str]:
    """
    Assign a seriousness level (1–4) to a triggered escalation.

    Level 1 — Tier 1 Support: General knowledge gap, no emotion
    Level 2 — Senior Agent: Mild frustration
    Level 3 — Support Manager: Clear anger, financial/legal concern, repeated failures
    Level 4 — Business Director: Abusive language, threats, safety, business-wide issue

    Returns:
        (level: int, label: str)
    """
    history_text = _format_history(history[-10:])  # More context for seriousness judgment

    levels_description = "\n".join(
        f"  Level {k}: {v['label']} — {v['description']}"
        for k, v in ESCALATION_HIERARCHY.items()
    )

    prompt = f"""You are a seriousness assessment engine for {BUSINESS_NAME}'s support escalation system.

Your job: assign the correct seriousness level (1 to 4) to this escalation.

Level descriptions:
{levels_description}

Factors to consider:
- Customer's tone: calm / mildly frustrated / clearly angry / abusive
- Is the customer explicitly demanding a human agent?
- Any financial, legal, or safety concern mentioned?
- How many unresolved messages have been exchanged without resolution?
- Could this issue affect multiple customers or the business as a whole?
- Direct threats (lawyer, consumer forum, social media)

Reason this escalation was triggered: {trigger_reason}

Conversation history:
{history_text}

Latest customer message:
\"{latest_message}\"

Respond with ONLY a JSON object. No other text:
{{"level": 1, "label": "Tier 1 Support"}}
(replace level and label with your assessment)"""

    try:
        raw = _call_llm(prompt, max_tokens=60)
        result = json.loads(raw)
        level = int(result.get("level", 1))
        level = max(1, min(4, level))               # Clamp to valid 1–4 range
        label = ESCALATION_HIERARCHY[level]["label"]
        return level, label
    except Exception as e:
        print(f"[Escalation] Error assessing seriousness: {e}")
        return _fallback_assess_seriousness(latest_message, trigger_reason)


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _format_history(history: list) -> str:
    """Format a list of message dicts into a readable conversation string."""
    if not history:
        return "(No prior messages in this session)"
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )


def _call_llm(prompt: str, max_tokens: int = 100) -> str:
    """
    Make a single OpenAI chat completion call.
    temperature=0.0 ensures deterministic output — critical for classification tasks.
    """
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,   # No randomness for classification
    )
    return response.choices[0].message.content.strip()


def _fallback_detect_emotional_signal(message: str) -> Tuple[bool, str]:
    msg_lower = message.lower()
    # Check for direct human agent requests
    human_keywords = ["human", "agent", "manager", "representative", "person", "support team", "supervisor", "escalate"]
    for kw in human_keywords:
        if kw in msg_lower:
            return True, f"Request for human support (keyword: '{kw}')"
            
    # Check for strong frustration/anger/abusive keywords
    anger_keywords = ["useless", "terrible", "horrible", "frustrated", "broken", "worst", "waste", "stolen", "lawyer", "legal", "court", "outage"]
    for kw in anger_keywords:
        if kw in msg_lower:
            return True, f"Urgent/emotional tone (keyword: '{kw}')"
            
    return False, ""


def _fallback_assess_seriousness(message: str, trigger_reason: str) -> Tuple[int, str]:
    msg_lower = message.lower()
    # Level 4: Abusive language, formal threats, safety concerns
    l4_keywords = ["sue", "lawyer", "legal", "court", "police", "scam", "fraud", "shitty", "fuck", "bastard"]
    for kw in l4_keywords:
        if kw in msg_lower:
            return 4, ESCALATION_HIERARCHY[4]["label"]
            
    # Level 3: Anger, repeated failures, billing/money dispute, direct demand for manager
    l3_keywords = ["manager", "supervisor", "director", "refund", "billing", "charge", "money", "failed", "unacceptable"]
    for kw in l3_keywords:
        if kw in msg_lower:
            return 3, ESCALATION_HIERARCHY[3]["label"]
            
    # Level 2: Mild frustration, minor complaints
    l2_keywords = ["slow", "bad", "waiting", "delay", "frustrated", "help"]
    for kw in l2_keywords:
        if kw in msg_lower:
            return 2, ESCALATION_HIERARCHY[2]["label"]
            
    # Level 1: Default / Knowledge gap
    return 1, ESCALATION_HIERARCHY[1]["label"]

