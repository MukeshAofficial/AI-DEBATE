from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import base64
import google.generativeai as genai
from gtts import gTTS  # Import gTTS
import io
from fastapi.responses import HTMLResponse


app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
AUDIO_UPLOAD_DIR = "audio_uploads"
CONVERSATION_HISTORY = {}

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')


origins = [
    "http://localhost", 
    "http://localhost:8000",
    "http://localhost:3000",
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- Data Models ---
class DebateTurnResponse(BaseModel):
    ai_response_text: str
    ai_response_audio_base64: str
    conversation_id: str


# --- Utility Functions ---
def load_conversation_history(conversation_id: str) -> list:
    if conversation_id in CONVERSATION_HISTORY:
        return CONVERSATION_HISTORY[conversation_id]
    else:
        return []


def save_conversation_history(conversation_id: str, user_audio_base64: str, ai_response_text: str) -> None:
    if conversation_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[conversation_id] = []

    CONVERSATION_HISTORY[conversation_id].append({
        "user_audio": user_audio_base64,
        "ai_response": ai_response_text,
    })


def generate_gemini_response(audio_base64: str, conversation_history: list) -> str:
    history_summary = "\n".join([f"AI: {turn['ai_response']}" for turn in conversation_history])

    prompt = f"""You are an expert AI debater skilled in logic.
1. **Delivery**: Prioritize clarity over emotional appeals.
2. **Structure**:
    - Start with a clear thesis statement.
    - Present 2-3 strongest arguments, each supported by examples or data.
    - Preemptively address counterarguments (e.g., "One might argue X, but this fails because Y").
    - Conclude with a summary reinforcing your stance.
3. **Tone**: Stay calm, respectful, and focused on logic. Avoid personal attacks.
4. **Adaptation**: If your opponent raises new points, directly address them before advancing your own arguments.
5. **Weakness Exploitation**: Identify logical fallacies or gaps in your opponent's reasoning and highlight them.
6. **Ethos/Pathos/Logos**: Use a mix of credibility (ethos), logic (logos), and occasional emotional framing (pathos) if strategically useful.

Example response format:
"Thesis: [Your stance].
Argument 1: [Core point + evidence].
Counterargument Rebuttal: [Address likely opposition].
Argument 2: [Next point + evidence]..."
{history_summary}
Now, respond to the following audio clip:
"""

    try:
        contents = [
            prompt,
            {
                "mime_type": "audio/webm",  # Important
                "data": base64.b64decode(audio_base64)
            }
        ]

        response = model.generate_content(contents=contents)

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise HTTPException(status_code=400,
                                detail=f"Gemini API blocked the request: {response.prompt_feedback.block_reason}")

        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from Gemini API: {str(e)}")


def text_to_speech(text: str) -> str:
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()  # Use BytesIO
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)  # Important: Reset the file pointer to the beginning
        audio_base64 = base64.b64encode(mp3_fp.read()).decode("utf-8")
        return audio_base64
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Text-to-Speech conversion: {str(e)}")


# --- API Endpoints ---
class AudioUploadForm(BaseModel):
    audio_data: str
    conversation_id: str


@app.post("/debate-turn/", response_model=DebateTurnResponse)
async def debate_turn(form_data: AudioUploadForm):
    conversation_id = form_data.conversation_id
    audio_base64 = form_data.audio_data

    conversation_history = load_conversation_history(conversation_id)

    ai_response_text = generate_gemini_response(audio_base64, conversation_history)

    ai_response_audio_base64 = text_to_speech(ai_response_text)

    save_conversation_history(conversation_id, audio_base64, ai_response_text)

    return DebateTurnResponse(
        ai_response_text=ai_response_text,
        ai_response_audio_base64=ai_response_audio_base64,
        conversation_id=conversation_id
    )


@app.get("/conversation_history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return history