// ─────────────────────────────────────────────────────────────────────────────
// chat.js — Smart Route Service Support Chatbot Frontend Logic
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:5000";   // Change this if deploying to a server
let sessionId  = null;                       // Filled on page load by /api/session
let isEscalated = false;                     // Locks input after escalation


// ─── Initialization ───────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  // 1. Start a session with the backend
  await initSession();

  // 2. Show welcome message
  addBotMessage(
    "👋 Hello! Welcome to Smart Route Service support. " +
    "I can help you with router setup, pricing, bookings, and troubleshooting. " +
    "What can I help you with today?"
  );

  // 3. File picker: update display label when a file is chosen
  document.getElementById("doc-input").addEventListener("change", (e) => {
    const name = e.target.files[0]?.name || "No file chosen";
    document.getElementById("file-name-display").textContent = name;
  });
});


// ─── Session Management ───────────────────────────────────────────────────────

async function initSession() {
  try {
    const res  = await fetch(`${API_BASE}/api/session`, { method: "POST" });
    const data = await res.json();
    sessionId  = data.session_id;
    console.log("[Chat] Session started:", sessionId);
  } catch (err) {
    console.error("[Chat] Failed to start session:", err);
    addBotMessage(
      "⚠️ Could not connect to the support server. " +
      "Please refresh the page and try again."
    );
  }
}


// ─── Message Sending ──────────────────────────────────────────────────────────

async function sendMessage() {
  if (isEscalated) return;   // Do not allow sending after escalation

  const input   = document.getElementById("user-input");
  const message = input.value.trim();

  if (!message)    return;
  if (!sessionId)  { addBotMessage("⚠️ Session not started. Please refresh."); return; }

  // Display user's message and clear input
  input.value = "";
  addUserMessage(message);
  setInputDisabled(true);
  scrollToBottom();
  showTyping(true);

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ session_id: sessionId, message }),
    });

    const data = await res.json();
    showTyping(false);

    if (data.error) {
      addBotMessage("⚠️ Server error: " + data.error);
      setInputDisabled(false);
      return;
    }

    // Show bot reply
    addBotMessage(data.response);

    if (data.escalated) {
      // Lock the chat and show the escalation banner
      isEscalated = true;
      showEscalationBanner(data.escalation_label, data.escalation_level);
      setInputDisabled(true);   // Permanently disabled for this session
    } else {
      setInputDisabled(false);
      input.focus();
    }

  } catch (err) {
    console.error("[Chat] sendMessage error:", err);
    showTyping(false);
    addBotMessage("⚠️ Connection error. Please check your internet and try again.");
    setInputDisabled(false);
  }
}


// ─── Document Upload ──────────────────────────────────────────────────────────

async function uploadDocument() {
  const fileInput = document.getElementById("doc-input");
  const file      = fileInput.files[0];

  if (!file) {
    setUploadStatus("Please choose a file first.", "error");
    return;
  }

  const btn = document.getElementById("upload-btn");
  btn.disabled = true;
  setUploadStatus("Uploading and indexing — this may take 10–30 seconds…", "");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res  = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
    const data = await res.json();

    if (res.ok) {
      setUploadStatus("✅ " + data.message, "success");
    } else {
      setUploadStatus("❌ " + (data.error || "Upload failed."), "error");
    }
  } catch (err) {
    console.error("[Upload] Error:", err);
    setUploadStatus("❌ Could not reach server. Is Flask running?", "error");
  } finally {
    btn.disabled = false;
    fileInput.value = "";
    document.getElementById("file-name-display").textContent = "No file chosen";
  }
}


// ─── DOM Helpers ──────────────────────────────────────────────────────────────

function addUserMessage(text) {
  appendMessage("user", "👤", text);
}

function addBotMessage(text) {
  appendMessage("bot", "📡", text);
}

function appendMessage(who, avatar, text) {
  const thread = document.getElementById("message-thread");
  const wrap   = document.createElement("div");
  wrap.className = `msg ${who}`;
  wrap.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-bubble">${sanitize(text)}</div>
  `;
  thread.appendChild(wrap);
  scrollToBottom();
}

function showTyping(visible) {
  const el = document.getElementById("typing-indicator");
  el.classList.toggle("hidden", !visible);
  if (visible) scrollToBottom();
}

function showEscalationBanner(label, level) {
  const thread = document.getElementById("message-thread");
  const banner = document.createElement("div");
  banner.className = "escalation-banner-inline";
  banner.innerHTML = `
    <span class="escalation-icon">🚨</span>
    <div class="escalation-content">
      <strong>Escalated to ${sanitize(label || "support team")}</strong>
      <span>Priority Level ${level || ""}. A human agent will contact you shortly.</span>
    </div>
  `;
  thread.appendChild(banner);
  scrollToBottom();
}

function setInputDisabled(disabled) {
  document.getElementById("user-input").disabled = disabled;
  document.getElementById("send-btn").disabled   = disabled;
}

function setUploadStatus(msg, type) {
  const el      = document.getElementById("upload-status");
  el.textContent = msg;
  el.className  = `upload-status ${type}`.trim();
}

function scrollToBottom() {
  const t = document.getElementById("message-thread");
  t.scrollTop = t.scrollHeight;
}

/**
 * Prevent XSS: convert user text to HTML-safe text node,
 * then extract innerHTML. This ensures < > & " are escaped.
 */
function sanitize(text) {
  const div = document.createElement("div");
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}
