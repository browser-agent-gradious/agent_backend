"""
agent_demo_server.py
────────────────────
Demo FastAPI WebSocket server for the HR Portal voice agent.

PURPOSE
  This is a hardcoded demo — it does NOT run an LLM or make real decisions.
  When audio arrives, it:
    1. Runs STT  (Groq whisper-large-v3) → gets transcript
    2. Plays back the pre-scripted "apply sick leave" action sequence
    3. Runs TTS  (Groq canopylabs/orpheus-v1-english) on the final text
    4. Sends the TTS audio back as Base64 JSON

  Use this to verify that all frontend actions work correctly before
  wiring up the real LangGraph agent.

SETUP
  pip install fastapi uvicorn groq python-dotenv

  Create a .env file (or export env vars):
    GROQ_API_KEY=gsk_...

RUN
  python agent_demo_server.py
  # Server starts on http://localhost:8000
  # WS endpoint: ws://localhost:8000/ws/agent

FRONTEND .env
  VITE_AGENT_WS_URL=ws://localhost:8000/ws/agent
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# ── Config ─────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("agent_demo")

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
STT_MODEL     = "whisper-large-v3"
TTS_MODEL     = "canopylabs/orpheus-v1-english"
TTS_VOICE     = "autumn"           # Options: tara, leah, jessica, lily, zac, austin, eric, troy
TTS_FORMAT    = "wav"              # Groq Orpheus supports: wav (default)
STEP_DELAY    = 1.0                # seconds between each scripted action (makes it visible)
GROQ_TIMEOUT  = 60                 # seconds for API calls

# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="HR Portal Agent Demo Server")

# Allow requests from the Vite dev server (localhost:5173) and any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT)

# ── Helpers ─────────────────────────────────────────────────────────────────

async def send(ws: WebSocket, payload: dict):
    """Send a JSON message over WebSocket and log it."""
    log.info("→ SEND %s", json.dumps(payload)[:120])
    await ws.send_text(json.dumps(payload))


async def step_delay(step_delay: float = STEP_DELAY):
    """Pause between scripted actions so the frontend has time to render each step."""
    await asyncio.sleep(step_delay)


def decode_base64_audio(b64_string: str, mime_type: str) -> bytes:
    """Convert the Base64 audio string from the frontend back to raw bytes."""
    return base64.b64decode(b64_string)


def encode_audio_to_base64(audio_bytes: bytes) -> str:
    """Encode raw audio bytes to a Base64 string for sending to the frontend."""
    return base64.b64encode(audio_bytes).decode("utf-8")


# ── STT — Groq Whisper ──────────────────────────────────────────────────────

def run_stt(audio_bytes: bytes, mime_type: str) -> str:
    """
    Send raw audio bytes to Groq whisper-large-v3 for transcription.

    Groq's transcription API expects a file-like object. We write the
    audio bytes to a temporary file because the groq-python SDK requires
    a named file with a valid extension to infer the format.

    Returns the transcript string, or an error message.
    """
    # Map MIME type to a file extension Groq accepts
    ext_map = {
        "audio/webm":              ".webm",
        "audio/webm;codecs=opus":  ".webm",
        "audio/ogg":               ".ogg",
        "audio/ogg;codecs=opus":   ".ogg",
        "audio/wav":               ".wav",
        "audio/mp4":               ".mp4",
        "audio/mpeg":              ".mp3",
    }
    ext = ext_map.get(mime_type.lower().split(";")[0].strip(), ".webm")

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        log.info("STT: sending %d bytes (%s) to Groq whisper-large-v3", len(audio_bytes), mime_type)

        with open(tmp_path, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                model=STT_MODEL,
                file=(f"audio{ext}", f, mime_type),
                response_format="text",
                language="en",
            )

        # result is a plain string when response_format="text"
        transcript = str(result).strip()
        log.info("STT result: %s", transcript)
        return transcript

    except Exception as e:
        log.error("STT failed: %s", e)
        return f"[STT error: {e}]"

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── TTS — Groq Orpheus ──────────────────────────────────────────────────────

def run_tts(text: str) -> bytes:
    """
    Convert text to speech using Groq canopylabs/orpheus-v1-english.

    Returns raw WAV audio bytes.
    The Groq TTS endpoint is OpenAI-compatible:
      POST https://api.groq.com/openai/v1/audio/speech
    The groq-python SDK exposes it via client.audio.speech.create().
    response.read() returns the raw audio bytes.
    """
    try:
        log.info("TTS: generating speech for: %s", text[:80])
        response = groq_client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format=TTS_FORMAT,
        )
        audio_bytes = response.read()
        log.info("TTS: received %d bytes of WAV audio", len(audio_bytes))
        return audio_bytes

    except Exception as e:
        log.error("TTS failed: %s", e)
        return b""


# ── Scripted demo action sequence for leaves page ───────────────────────────────────────────

async def run_leave_demo_sequence(ws: WebSocket, transcript: str):
    """
    Plays back the hardcoded "apply sick leave for tomorrow" demo sequence.

    This simulates what a real LangGraph agent would do after deciding
    the intent from the transcript. Each message mirrors the format that
    the real agent backend will produce, so the frontend can be validated
    without the full LLM pipeline.

    Message types:
      ack    → immediately show transcript in the UI
      step   → agent "thinking" log (purple in UI)
      action → execute a frontend action
      audio  → Base64 WAV audio to play + final text
    """

    # ── Step 0: Acknowledge transcript ──────────────────────────────────────
    await send(ws, {
        "type":       "ack",
        "text":       f'You said: "{transcript}"',
        "transcript": transcript,
    })
    await step_delay()

    # ── Step 1: Navigate to /leaves ─────────────────────────────────────────
    await send(ws, {
        "type": "step",
        "text": "Navigating to the Leaves page…",
    })
    await step_delay()

    await send(ws, {
        "type":    "action",
        "action":  "navigate",
        "payload": { "target": "/leaves" },
    })
    await step_delay()

    # ── Step 2: Open the leave application modal ─────────────────────────────
    await send(ws, {
        "type": "step",
        "text": "Opening leave application form…",
    })
    await step_delay()

    await send(ws, {
        "type":    "action",
        "action":  "click_dot",
        "payload": {
            "label":    "Apply for Leave",
            "selector": 'button[type="button"]',
        },
    })
    await step_delay()

    # ── Step 3: Fill Leave Type ──────────────────────────────────────────────
    # First show the red dot on the Leave Type dropdown, then fill it
    await send(ws, {
        "type":    "action",
        "action":  "click_dot",
        "payload": {
            "label":    "Leave Type",
            "selector": "select[name=type]",
        },
    })
    await asyncio.sleep(0.35)   # let the dot appear before filling

    await send(ws, {
        "type":    "action",
        "action":  "fill_form",
        "payload": { "field": "type", "value": "Sick Leave" },
    })
    await step_delay()

    # ── Step 4: Fill From date ───────────────────────────────────────────────
    await send(ws, {
        "type":    "action",
        "action":  "fill_form",
        "payload": { "field": "from", "value": "2026-05-28" },
    })
    await step_delay()

    # ── Step 5: Fill To date ─────────────────────────────────────────────────
    await send(ws, {
        "type":    "action",
        "action":  "fill_form",
        "payload": { "field": "to", "value": "2026-05-28" },
    })
    await step_delay()

    # ── Step 6: Fill Reason ──────────────────────────────────────────────────
    await send(ws, {
        "type":    "action",
        "action":  "fill_form",
        "payload": { "field": "reason", "value": "Not feeling well, need rest." },
    })
    await step_delay()

    # ── Step 7: Click Submit ─────────────────────────────────────────────────
    await send(ws, {
        "type":    "action",
        "action":  "click_dot",
        "payload": {
            "label":    "Submit",
            "selector": 'button[type="submit"]',
        },
    })
    await asyncio.sleep(0.35)

    # await send(ws, {
    #     "type":    "action",
    #     "action":  "click",
    #     "payload": { "selector": 'button[type="submit"]' },
    # })
    # await step_delay()

    # ── Step 8: TTS final response ───────────────────────────────────────────
    final_text = (
        "[cheerful] Done! Your sick leave for tomorrow has been applied. "
        "You will receive a confirmation once it is approved by your manager."
    )
    await send(ws, {
        "type": "step",
        "text": "Generating voice response…",
    })

    # Run TTS in a thread pool so we don't block the event loop
    audio_bytes = await asyncio.get_event_loop().run_in_executor(
        None, run_tts, final_text
    )

    if audio_bytes:
        await send(ws, {
            "type":     "audio",
            "audio":    encode_audio_to_base64(audio_bytes),
            "mimeType": "audio/wav",
            "text":     "Done! Your sick leave for tomorrow has been applied. You will receive a confirmation once it is approved by your manager.",
        })
    else:
        # TTS failed — send text-only result as fallback
        await send(ws, {
            "type": "result",
            "text": "Done! Sick leave for tomorrow submitted successfully.",
        })

# ── Scripted demo action sequence for attendance page ───────────────────────────────────────────

async def run_attendance_demo_sequence(ws: WebSocket, transcript: str):
    """
    Plays back the hardcoded "Show me my attendance for april." demo sequence.

    This simulates what a real LangGraph agent would do after deciding
    the intent from the transcript. Each message mirrors the format that
    the real agent backend will produce, so the frontend can be validated
    without the full LLM pipeline.

    Message types:
      ack    → immediately show transcript in the UI
      step   → agent "thinking" log (purple in UI)
      action → execute a frontend action
      audio  → Base64 WAV audio to play + final text
    """

    # ── Step 0: Acknowledge transcript ──────────────────────────────────────
    await send(ws, {
        "type":       "ack",
        "text":       f'You said: "{transcript}"',
        "transcript": transcript,
    })
    await step_delay()

    # ── Step 1: Navigate to /attendance ─────────────────────────────────────────
    await send(ws, {
        "type": "step",
        "text": "Navigating to the Attendance page…",
    })
    await step_delay()

    await send(ws, {
        "type":    "action",
        "action":  "navigate",
        "payload": { "target": "/attendance" },
    })
    await step_delay()

    # ── Step 2: Open the attendance month dropdown ─────────────────────────────
    # First show the red dot on the Leave Type dropdown, then fill it
    await send(ws, {
        "type": "step",
        "text": "Selecting the attendance month…",
    })
    await step_delay()
    
    await send(ws, {
        "type":    "action",
        "action":  "click_dot",
        "payload": {
            "label":    "Attendance month dropdown",
            "selector": 'button[id="attendance-month-dropdown-trigger"]',
        },
    })
    await step_delay()

    # ── Step 3: Select the attendance month  ──────────────────────────────────────────────
    await send(ws, {
        "type":    "action",
        "action":  "click_dot",
        "payload": {
            "label":    "April 2026",
            "selector": 'li[id="attendance-month-dropdown-option-april"]',
        },
    })
    await step_delay()

    # ── Step 8: TTS final response ───────────────────────────────────────────
    final_text = (
        "[cheerful] Done! Here is your attendance information for April 2026."
    )
    await send(ws, {
        "type": "step",
        "text": "Generating voice response…",
    })

    # Run TTS in a thread pool so we don't block the event loop
    audio_bytes = await asyncio.get_event_loop().run_in_executor(
        None, run_tts, final_text
    )

    if audio_bytes:
        await send(ws, {
            "type":     "audio",
            "audio":    encode_audio_to_base64(audio_bytes),
            "mimeType": "audio/wav",
            "text":     "Done! Here is your attendance information for April 2026.",
        })
    else:
        # TTS failed — send text-only result as fallback
        await send(ws, {
            "type": "result",
            "text": "Done! Here is your attendance information for April 2026.",
        })

# ── Scripted demo action sequence for payroll page ───────────────────────────────────────────

async def run_payroll_demo_sequence(ws: WebSocket, transcript: str):
    """
    Plays back the hardcoded "Show me my payroll for march." demo sequence.

    This simulates what a real LangGraph agent would do after deciding
    the intent from the transcript. Each message mirrors the format that
    the real agent backend will produce, so the frontend can be validated
    without the full LLM pipeline.

    Message types:
      ack    → immediately show transcript in the UI
      step   → agent "thinking" log (purple in UI)
      action → execute a frontend action
      audio  → Base64 WAV audio to play + final text
    """

    # ── Step 0: Acknowledge transcript ──────────────────────────────────────
    await send(ws, {
        "type":       "ack",
        "text":       f'You said: "{transcript}"',
        "transcript": transcript,
    })
    await step_delay()

    # ── Step 1: Navigate to /payroll ─────────────────────────────────────────
    await send(ws, {
        "type": "step",
        "text": "Navigating to the Payroll page…",
    })
    await step_delay()

    await send(ws, {
        "type":    "action",
        "action":  "navigate",
        "payload": { "target": "/payroll" },
    })
    await step_delay()

    # ── Step 2: Open the payroll month dropdown ─────────────────────────────
    # First show the red dot on the Leave Type dropdown, then fill it
    await send(ws, {
        "type": "step",
        "text": "Selecting the payroll month…",
    })
    await step_delay()
    
    await send(ws, {
        "type":    "action",
        "action":  "click_dot",
        "payload": {
            "label":    "Payroll month dropdown",
            "selector": 'button[id="payroll-month-dropdown-trigger"]',
        },
    })
    await step_delay()

    # ── Step 3: Select the payroll month  ──────────────────────────────────────────────
    await send(ws, {
        "type":    "action",
        "action":  "click_dot",
        "payload": {
            "label":    "March 2026",
            "selector": 'li[id="payroll-month-dropdown-option-march"]',
        },
    })
    await step_delay()

    # ── Step 8: TTS final response ───────────────────────────────────────────
    final_text = (
        "[cheerful] Done! Here is your payroll information for March 2026."
    )
    await send(ws, {
        "type": "step",
        "text": "Generating voice response…",
    })

    # Run TTS in a thread pool so we don't block the event loop
    audio_bytes = await asyncio.get_event_loop().run_in_executor(
        None, run_tts, final_text
    )

    if audio_bytes:
        await send(ws, {
            "type":     "audio",
            "audio":    encode_audio_to_base64(audio_bytes),
            "mimeType": "audio/wav",
            "text":     "Done! Here is your payroll information for March 2026.",
        })
    else:
        # TTS failed — send text-only result as fallback
        await send(ws, {
            "type": "result",
            "text": "Done! Here is your payroll information for March 2026.",
        })


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/agent")
async def agent_ws(ws: WebSocket):
    """
    Main WebSocket handler.

    Protocol:
      Client → Server:
        { type: "audio", audio: "<base64>", mimeType: "audio/webm", current_page: "/..." }

      Server → Client (in order):
        { type: "ack",    text, transcript }
        { type: "step",   text }
        { type: "action", action, payload }
        { type: "audio",  audio, mimeType, text }
        { type: "error",  text }
    """
    await ws.accept()
    client = ws.client
    log.info("Client connected: %s:%s", client.host, client.port)

    try:
        while True:
            raw = await ws.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send(ws, { "type": "error", "text": "Invalid JSON received." })
                continue

            msg_type = msg.get("type")
            log.info("← RECV type=%s page=%s", msg_type, msg.get("current_page", "?"))

            if msg_type == "audio":
                b64_audio = msg.get("audio", "")
                mime_type = msg.get("mimeType", "audio/webm")
                current_page = msg.get("current_page", "/")

                if not b64_audio:
                    await send(ws, { "type": "error", "text": "Empty audio payload." })
                    continue

                # Decode Base64 → raw bytes
                try:
                    audio_bytes = decode_base64_audio(b64_audio, mime_type)
                except Exception as e:
                    await send(ws, { "type": "error", "text": f"Base64 decode failed: {e}" })
                    continue

                log.info("Audio received: %d bytes, mime=%s, page=%s",
                         len(audio_bytes), mime_type, current_page)

                # STT — run in thread pool so async loop isn't blocked
                transcript = await asyncio.get_event_loop().run_in_executor(
                    None, run_stt, audio_bytes, mime_type
                )

                # Run the demo action sequence
                if "attendance" in transcript.lower():
                    await run_attendance_demo_sequence(ws, transcript)
                elif "payroll" in transcript.lower():
                    await run_payroll_demo_sequence(ws, transcript)
                elif "leave" in transcript.lower() or "sick" in transcript.lower():
                    await run_leave_demo_sequence(ws, transcript)
                else:
                    await run_attendance_demo_sequence(ws, transcript)

            else:
                await send(ws, {
                    "type": "error",
                    "text": f"Unknown message type: {msg_type!r}. Expected 'audio'.",
                })

    except WebSocketDisconnect:
        log.info("Client disconnected: %s:%s", client.host, client.port)
    except Exception as e:
        log.error("Unhandled error: %s", e, exc_info=True)
        try:
            await send(ws, { "type": "error", "text": f"Server error: {e}" })
        except Exception:
            pass


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "stt_model":  STT_MODEL,
        "tts_model":  TTS_MODEL,
        "tts_voice":  TTS_VOICE,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    if not GROQ_API_KEY:
        log.warning("GROQ_API_KEY is not set. STT and TTS calls will fail.")
        log.warning("Create a .env file with: GROQ_API_KEY=gsk_...")

    log.info("Starting HR Portal Agent Demo Server")
    log.info("WS endpoint: ws://localhost:8000/ws/agent")
    log.info("Health check: http://localhost:8000/health")

    uvicorn.run(
        "agent_demo_server:app",
        host="0.0.0.0",
        port=8000,
        # reload=True,
        log_level="info",
    )
