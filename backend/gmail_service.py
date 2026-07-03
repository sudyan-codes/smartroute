# backend/gmail_service.py
"""
Gmail Service
-------------
Sends escalation alert emails via Gmail SMTP using an App Password.
Uses SSL on port 465 (more reliable than STARTTLS on 587 in many environments).

Email contains:
  - Timestamp and session ID
  - Seriousness level and reason for escalation
  - Full conversation transcript
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

from config import (
    GMAIL_SENDER,
    GMAIL_APP_PASSWORD,
    BUSINESS_NAME,
    ESCALATION_HIERARCHY,
)


def send_escalation_email(
    level: int,
    session_id: str,
    transcript: list,
    reason: str,
    label: str,
):
    """
    Send an escalation alert to the correct support hierarchy level.

    Parameters:
        level       : Seriousness level 1–4
        session_id  : Full UUID of the conversation session
        transcript  : List of {role, content, timestamp} message dicts
        reason      : Why escalation was triggered (for the email body)
        label       : Human-readable level label (e.g. "Support Manager")
    """
    target    = ESCALATION_HIERARCHY[level]
    recipient = target["email"]
    short_id  = session_id[:8].upper()       # First 8 chars for email subject readability
    timestamp = datetime.utcnow().strftime("%d %B %Y, %H:%M:%S UTC")

    subject = (
        f"[{BUSINESS_NAME}] 🚨 Escalation Level {level} – {label} "
        f"| Session {short_id}"
    )

    # Build transcript text
    transcript_text = "\n".join(
        f"  [{m.get('timestamp', '—')}] {m['role'].upper()}: {m['content']}"
        for m in transcript
    )

    body = f"""
╔══════════════════════════════════════════════════════════════════╗
   {BUSINESS_NAME.upper()} — CUSTOMER ESCALATION ALERT
╚══════════════════════════════════════════════════════════════════╝

Escalation Timestamp : {timestamp}
Session ID           : {session_id}
Seriousness Level    : Level {level} — {label}
Reason for Escalation: {reason}
Routed To            : {target['label']} ({recipient})

═══════════════════════════════════════════════════════════════════
FULL CONVERSATION TRANSCRIPT
═══════════════════════════════════════════════════════════════════

{transcript_text}

═══════════════════════════════════════════════════════════════════

ACTION REQUIRED:
Please follow up with this customer promptly.
This email was generated automatically by the {BUSINESS_NAME} support chatbot.

— {BUSINESS_NAME} Automated Support System
"""

    # Build MIME email message
    msg = MIMEMultipart()
    msg["From"] = GMAIL_SENDER
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # Send via Gmail SMTP (SSL, port 465)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER, recipient, msg.as_string())
        print(f"[Gmail] ✓ Escalation email sent → {recipient} (Level {level})")
    except smtplib.SMTPAuthenticationError:
        print("[Gmail] ✗ Authentication failed. Check GMAIL_SENDER and GMAIL_APP_PASSWORD in .env")
        raise
    except smtplib.SMTPException as e:
        # Log the error but do NOT re-raise.
        # The customer chat should still get a response even if the email fails.
        print(f"[Gmail] ✗ SMTP error: {e}")
