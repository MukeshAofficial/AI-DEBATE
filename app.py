from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import base64
import google.generativeai as genai
import io
from fastapi.responses import HTMLResponse
import requests  # Import the requests library


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")  # Get Deepgram API key from environment
AUDIO_UPLOAD_DIR = "audio_uploads"
CONVERSATION_HISTORY = {}

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')


origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://ai-debate.onrender.com",

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


class AudioUploadForm(BaseModel):
    audio_data: str
    conversation_id: str
    user_prompt: str  # Add the user prompt
    selected_voice: str  # Add selected voice


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


def generate_gemini_response(audio_base64: str, conversation_history: list, user_prompt: str) -> str:
    history_summary = "\n".join([f"AI: {turn['ai_response']}" for turn in conversation_history])

    # Use the user-provided prompt
    prompt = user_prompt + "\n" + history_summary + "\nNow, respond to the following audio clip:"

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


# Deepgram Text-to-Speech function (using requests)
def text_to_speech(text: str, voice_model: str = "aura-asteria-en") -> str:
    try:
        url = f"https://api.deepgram.com/v1/speak?model={voice_model}"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            audio_bytes = response.content
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            return audio_base64
        else:
            raise HTTPException(status_code=response.status_code,
                                detail=f"Deepgram API Error: {response.status_code} - {response.text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Text-to-Speech conversion with Deepgram: {str(e)}")


# --- API Endpoints ---
@app.post("/debate-turn/", response_model=DebateTurnResponse)
async def debate_turn(form_data: AudioUploadForm):
    conversation_id = form_data.conversation_id
    audio_base64 = form_data.audio_data
    user_prompt = form_data.user_prompt
    selected_voice = form_data.selected_voice  # Get the selected voice

    conversation_history = load_conversation_history(conversation_id)

    # Pass the user prompt to generate_gemini_response
    ai_response_text = generate_gemini_response(audio_base64, conversation_history, user_prompt)

    ai_response_audio_base64 = text_to_speech(ai_response_text, selected_voice)

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