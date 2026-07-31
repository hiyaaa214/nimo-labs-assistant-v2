from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
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


SYSTEM_PROMPT = """You are a helpful multilingual voice assistant for a hotel.
Detect the language or dialect of the user's message automatically.
Always respond in the SAME language or dialect the user wrote in.
If the user writes in a mix of languages (e.g. Hindi+English), respond in that same mixed style naturally.
Pay close attention to regional Indian dialects such as Haryanvi, Marwari, Mewari, Bhojpuri, or Rajasthani.
If the user's message uses dialect-specific words or phrasing (not standard Hindi), mirror that same
dialect vocabulary and tone in your reply, rather than replying in generic standard Hindi.
IMPORTANT: Keep every reply SHORT and to the point — 1 to 2 sentences maximum, like a real spoken
conversation with hotel staff. Do not add extra explanations or lists unless the user specifically
asks for more detail. Be warm and helpful, but brief."""


@app.post("/chat")
def chat(request: ChatRequest):
    try:
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

        # Safety net: if a model ever leaks its internal "thinking" text
        # (e.g. wrapped in <think>...</think> tags), strip it before replying
        import re
        reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()

        return {"reply": reply}

    except Exception as e:
        print("ERROR in /chat:", repr(e))
        return {"reply": "Sorry, I had trouble responding just now. Please try again."}


def detect_script_language(text: str, whisper_language: str) -> str:
    """
    Whisper sometimes mislabels the detected language for short phrases.
    We double-check using the actual Unicode script of the transcribed text,
    which is far more reliable than Whisper's own language guess.
    """
    import re

    # Unicode ranges for each script
    script_ranges = {
        "gujarati": r'[\u0A80-\u0AFF]',
        "bengali": r'[\u0980-\u09FF]',
        "tamil": r'[\u0B80-\u0BFF]',
        "telugu": r'[\u0C00-\u0C7F]',
        "kannada": r'[\u0C80-\u0CFF]',
        "malayalam": r'[\u0D00-\u0D7F]',
        "punjabi": r'[\u0A00-\u0A7F]',   # Gurmukhi script
    }

    for language_name, pattern in script_ranges.items():
        if re.search(pattern, text):
            return language_name

    # Devanagari script covers both Hindi and Marathi — can't tell these
    # apart from script alone, so trust Whisper's guess in that case
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    if has_devanagari:
        return whisper_language

    # No Indian script detected at all -> almost certainly English
    if text.strip():
        return "english"

    return whisper_language


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
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

        corrected_language = detect_script_language(transcription.text, transcription.language)

        return {
            "text": transcription.text,
            "language": corrected_language
        }

    except Exception as e:
        print("ERROR in /transcribe:", repr(e))
        return {"text": "", "language": ""}


@app.get("/")
def root():
    return {"status": "Multilingual AI Assistant backend is running!"}


# Maps Whisper's detected language name to a verified edge-tts voice.
# Run: edge-tts --list-voices | findstr /C:"-IN-"   to see/add more Indian languages later.
VOICE_MAP = {
    "english": "en-IN-NeerjaNeural",
    "hindi": "hi-IN-SwaraNeural",
    "bengali": "bn-IN-TanishaaNeural",
    "gujarati": "gu-IN-DhwaniNeural",
    "kannada": "kn-IN-SapnaNeural",
    "malayalam": "ml-IN-SobhanaNeural",
    "marathi": "mr-IN-AarohiNeural",
    "tamil": "ta-IN-PallaviNeural",
    "telugu": "te-IN-ShrutiNeural",
    "urdu": "ur-IN-GulNeural",
    "punjabi": "hi-IN-SwaraNeural",   # no real Punjabi voice exists in edge-tts — Hindi is the closest available
}
DEFAULT_VOICE = "en-IN-NeerjaNeural"


class SpeakRequest(BaseModel):
    text: str
    language: str = "english"


@app.post("/speak")
async def speak(request: SpeakRequest):
    # Pick the right voice based on detected language, fall back to English
    voice = VOICE_MAP.get(request.language.lower().strip(), DEFAULT_VOICE)

    # A slightly slower pace and a touch lower pitch tends to sound calmer
    # and more natural than the engine's default rushed pace
    communicate = edge_tts.Communicate(
        request.text,
        voice,
        rate="-8%",
        pitch="-2Hz"
    )

    async def audio_stream():
        # Sends audio chunks to the browser as edge-tts generates them,
        # instead of waiting for the entire file and saving it to disk first
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")
