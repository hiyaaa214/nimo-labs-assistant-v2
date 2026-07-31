from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import tempfile
import edge_tts

# Load environment variables from the .env file sitting next to this script
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Set up the Groq client (uses OpenAI's library, just pointed at Groq's servers)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI()

# Allow the frontend (running on a different address/port) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for development only, restrict this later in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    language: str = ""   # optional: Whisper's detected language, passed from the frontend


SYSTEM_PROMPT = """You are a helpful multilingual AI assistant.
Detect the language or dialect of the user's message automatically.
Always respond in the SAME language or dialect the user wrote in.
If the user writes in a mix of languages (e.g. Hindi+English), respond in that same mixed style naturally.
Pay close attention to regional Indian dialects such as Haryanvi, Marwari, Mewari, Bhojpuri, or Rajasthani.
If the user's message uses dialect-specific words or phrasing (not standard Hindi), mirror that same
dialect vocabulary and tone in your reply, rather than replying in generic standard Hindi.
Keep responses clear, helpful, and culturally appropriate for the language or dialect used."""


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # If we already know the language (from Whisper), tell the model directly
    # instead of making it guess — this makes replies far more reliable.
    if request.language:
        messages.append({
            "role": "system",
            "content": f"The user's message is in {request.language}. "
                       f"You MUST reply only in {request.language}, nothing else."
        })

    messages.append({"role": "user", "content": request.message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    reply = response.choices[0].message.content
    return {"reply": reply}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Read the raw bytes of the uploaded audio file
    audio_bytes = await audio.read()

    # Groq's Whisper needs a (filename, file_bytes, content_type) tuple
    # We reuse the original filename/content-type sent by the browser
    audio_file = (audio.filename, audio_bytes, audio.content_type)

    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
        response_format="verbose_json"   # gives us the detected language too
    )

    return {
        "text": transcription.text,
        "language": transcription.language
    }


@app.get("/")
def root():
    return {"status": "Multilingual AI Assistant backend is running!"}


# Maps Whisper's detected language name to a verified edge-tts voice.
# Run: edge-tts --list-voices   to see/add more languages later.
VOICE_MAP = {
    "english": "en-US-JennyNeural",
    "hindi": "hi-IN-SwaraNeural",
    "spanish": "es-ES-ElviraNeural",
    "french": "fr-FR-DeniseNeural",
    "german": "de-DE-KatjaNeural",
    "arabic": "ar-SA-ZariyahNeural",
    "bengali": "bn-IN-TanishaaNeural",
    "gujarati": "gu-IN-DhwaniNeural",
    "marathi": "mr-IN-AarohiNeural",
    "tamil": "ta-IN-PallaviNeural",
    "telugu": "te-IN-ShrutiNeural",
    "urdu": "ur-PK-UzmaNeural",
}
DEFAULT_VOICE = "en-US-JennyNeural"


class SpeakRequest(BaseModel):
    text: str
    language: str = "english"


@app.post("/speak")
async def speak(request: SpeakRequest):
    # Pick the right voice based on detected language, fall back to English
    voice = VOICE_MAP.get(request.language.lower().strip(), DEFAULT_VOICE)

    # Create a temporary file to hold the generated audio
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp_file.name
    tmp_file.close()

    # Generate the speech audio and save it to that file
    communicate = edge_tts.Communicate(request.text, voice)
    await communicate.save(tmp_path)

    # Send the audio file back to the browser
    return FileResponse(tmp_path, media_type="audio/mpeg", filename="speech.mp3")
