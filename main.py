import os
import uuid
import logging
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".m4a",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
}

MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

CHUNK_WORD_LIMIT = 800


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Meeting Minutes Generator",
    description="AI-powered meeting transcription and summarization API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD WHISPER
# ============================================================

logger.info("Loading Whisper model...")

try:
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")

except Exception as error:
    logger.exception("Failed to load Whisper model.")
    raise RuntimeError(f"Whisper model loading failed: {error}")


# ============================================================
# LOAD BART SUMMARIZER
# ============================================================

from transformers import pipeline
import torch

# GPU இருந்தால் பயன்படுத்தும்
device = 0 if torch.cuda.is_available() else -1

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

def fast_bart_summary(text, chunk_size=1800):
    words = text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    summaries = []

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=150,
            min_length=40,
            do_sample=False,
            truncation=True
        )
        summaries.append(result[0]["summary_text"])

    # Final summary
    combined = " ".join(summaries)

    if len(combined.split()) > 1800:
        result = summarizer(
            combined,
            max_length=200,
            min_length=60,
            do_sample=False,
            truncation=True
        )
        return result[0]["summary_text"]

    return combined


# Example
transcript = """
Paste your meeting transcript here...
"""

summary = fast_bart_summary(transcript)
print(summary)

# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file(filename: str, file_size: int):

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{extension}'. "
                f"Allowed formats: {allowed}"
            ),
        )

    if file_size > MAX_FILE_SIZE_BYTES:

        raise HTTPException(
            status_code=413,
            detail=f"File size cannot exceed {MAX_FILE_SIZE_MB} MB.",
        )


# ============================================================
# EXTRACT AUDIO USING FFMPEG
# ============================================================

def extract_audio(
    input_file: Path,
    output_file: Path
):

    logger.info("Extracting audio using FFmpeg...")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        str(output_file),
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=500,
            detail=(
                "FFmpeg is not installed or not available "
                "in the system PATH."
            ),
        )

    if result.returncode != 0:

        logger.error(result.stderr)

        raise HTTPException(
            status_code=500,
            detail="Failed to extract audio from the uploaded video.",
        )

    logger.info("Audio extraction completed.")


# ============================================================
# SPLIT TRANSCRIPT
# ============================================================

def split_text(
    text: str,
    word_limit: int = CHUNK_WORD_LIMIT
):

    words = text.split()

    chunks = []

    for start in range(0, len(words), word_limit):

        chunk = " ".join(
            words[start:start + word_limit]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary(text: str):

    if not text.strip():
        return "No transcript available."

    chunks = split_text(text)

    summaries = []

    for index, chunk in enumerate(chunks):

        logger.info(
            "Summarizing chunk %d/%d...",
            index + 1,
            len(chunks),
        )

        try:

            result = summarizer(
                chunk,
                max_length=200,
                min_length=40,
                do_sample=False,
            )

            summaries.append(
                result[0]["summary_text"]
            )

        except Exception as error:

            logger.exception(
                "Summary generation failed."
            )

            raise HTTPException(
                status_code=500,
                detail=f"Summary generation failed: {error}",
            )

    # If there is only one chunk
    if len(summaries) == 1:
        return summaries[0]

    # Second-level summary
    combined_summary = " ".join(summaries)

    logger.info(
        "Generating final summary from %d summaries...",
        len(summaries),
    )

    try:

        final_result = summarizer(
            combined_summary,
            max_length=300,
            min_length=60,
            do_sample=False,
        )

        return final_result[0]["summary_text"]

    except Exception:

        logger.exception(
            "Final summary generation failed."
        )

        return combined_summary


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", include_in_schema=False)
async def home():

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="frontend/index.html not found.",
        )

    return FileResponse(
        str(index_file)
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "message": "AI Meeting Minutes Generator is running",
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api")
async def api_info():

    return {
        "project": "AI Meeting Minutes Generator",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/",
            "/health",
            "/api/summarize",
            "/docs",
        ],
    }


# ============================================================
# SUMMARIZE MEETING
# ============================================================

@app.post(
    "/api/summarize",
    response_model=SummaryResponse,
)
async def summarize_meeting(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected.",
        )

    logger.info(
        "Received file: %s",
        file.filename,
    )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:

        file_content = await file.read()

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read uploaded file: {error}",
        )

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    validate_file(
        file.filename,
        len(file_content),
    )

    # --------------------------------------------------------
    # Generate safe filename
    # --------------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    unique_name = (
        f"{uuid.uuid4().hex}{extension}"
    )

    saved_file = UPLOAD_DIR / unique_name

    audio_file = None

    try:

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        saved_file.write_bytes(
            file_content
        )

        logger.info(
            "File saved: %s",
            saved_file,
        )

        # ----------------------------------------------------
        # Decide audio source
        # ----------------------------------------------------

        if extension in AUDIO_EXTENSIONS:

            audio_file = saved_file

        else:

            audio_file = (
                UPLOAD_DIR
                / f"{uuid.uuid4().hex}.wav"
            )

            extract_audio(
                saved_file,
                audio_file,
            )

        # ----------------------------------------------------
        # Whisper transcription
        # ----------------------------------------------------

        logger.info(
            "Starting Whisper transcription..."
        )

        try:

            result = whisper_model.transcribe(
                str(audio_file),
                fp16=False,
            )

        except Exception as error:

            logger.exception(
                "Whisper transcription failed."
            )

            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {error}",
            )

        transcript = result.get(
            "text",
            ""
        ).strip()

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration = None

        segments = result.get(
            "segments",
            []
        )

        if segments:

            last_segment = segments[-1]

            duration = last_segment.get(
                "end"
            )

        # ----------------------------------------------------
        # Empty transcript
        # ----------------------------------------------------

        if not transcript:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No speech was detected "
                    "in the uploaded file."
                ),
            )

        logger.info(
            "Transcription completed. Characters: %d",
            len(transcript),
        )

        # ----------------------------------------------------
        # Generate summary
        # ----------------------------------------------------

        logger.info(
            "Starting AI summary generation..."
        )

        summary = generate_summary(
            transcript
        )

        logger.info(
            "Meeting summary generated successfully."
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return SummaryResponse(

            filename=file.filename,

            transcript=transcript,

            summary=summary,

            duration_seconds=duration,
        )

    finally:

        # ----------------------------------------------------
        # Cleanup uploaded files
        # ----------------------------------------------------

        try:

            if saved_file.exists():
                saved_file.unlink()

            if (
                audio_file
                and audio_file != saved_file
                and audio_file.exists()
            ):
                audio_file.unlink()

            logger.info(
                "Temporary files cleaned."
            )

        except Exception as cleanup_error:

            logger.warning(
                "Cleanup failed: %s",
                cleanup_error,
            )


# ============================================================
# STATIC FRONTEND
# ============================================================

if FRONTEND_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(FRONTEND_DIR)
        ),
        name="static",
    )


# ============================================================
# STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
async def startup_event():

    logger.info("=" * 60)
    logger.info(
        "AI Meeting Minutes Generator started successfully."
    )
    logger.info(
        "Frontend: http://127.0.0.1:8000/"
    )
    logger.info(
        "API Docs: http://127.0.0.1:8000/docs"
    )
    logger.info("=" * 60)