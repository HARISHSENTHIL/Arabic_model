import os
import gc
import torch
import tempfile
import logging
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from torch.torch_version import TorchVersion
import torch.serialization
import nemo.collections.asr as nemo_asr
from pyannote.audio.core.task import Specifications
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stt_service.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Initializing models on device: {device}")

if device == "cuda":
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

logger.info("Loading Arabic ASR model...")
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained("nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0")
asr_model = asr_model.cuda() if device == "cuda" else asr_model
asr_model.eval()
logger.info("ASR model loaded successfully")

diarization_pipeline = None
try:
    if not HF_TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN is missing from environment variables")
    
    logger.info("Loading speaker diarization pipeline...")
    torch.serialization.add_safe_globals([TorchVersion])
    torch.serialization.add_safe_globals([Specifications])
    
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    
    if device == "cuda":
        diarization_pipeline.to(torch.device("cuda"))
    
    logger.info("Diarization pipeline loaded successfully")

except Exception as e:
    logger.error(f"Failed to load diarization pipeline: {str(e)}")
    logger.warning("Service will continue without diarization capabilities")

app = FastAPI(title="Arabic STT + Diarization API")


def post_process_diarization(diarization, min_segment_duration=1.0, min_gap_duration=0.5):
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "duration": turn.end - turn.start
        })
    
    if not segments:
        return segments
    
    segments.sort(key=lambda x: x["start"])
    
    merged_segments = []
    current_segment = segments[0].copy()
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        gap = next_segment["start"] - current_segment["end"]
        
        if (current_segment["speaker"] == next_segment["speaker"] and 
            gap < min_gap_duration):
            current_segment["end"] = next_segment["end"]
            current_segment["duration"] = current_segment["end"] - current_segment["start"]
        else:
            if current_segment["duration"] >= min_segment_duration:
                merged_segments.append(current_segment)
            current_segment = next_segment.copy()
    
    if current_segment["duration"] >= min_segment_duration:
        merged_segments.append(current_segment)
    
    return merged_segments

def get_processing_parameters(speaker_count):
    if speaker_count <= 2:
        return 1.0, 0.5
    elif speaker_count <= 4:
        return 0.8, 0.3
    else:
        return 0.6, 0.2

def validate_audio(tmp_path):
    logger.info("Validating audio file")
    audio_original, sr_original = librosa.load(tmp_path, sr=None)
    duration = len(audio_original) / sr_original
    
    logger.info(f"Audio properties: duration={duration:.2f}s, sample_rate={sr_original}Hz, samples={len(audio_original)}")
    
    if duration < 1.0:
        logger.error(f"Audio validation failed: duration {duration:.2f}s is below minimum 1.0s")
        raise ValueError(f"Audio too short for diarization: {duration:.2f}s. Minimum required: 1.0s")
    
    logger.info("Audio validation passed")
    return duration

def process_diarization(tmp_path):
    if diarization_pipeline is None:
        logger.error("Diarization pipeline not available")
        raise RuntimeError("PyAnnote diarization not available.")
    
    logger.info("Processing audio for diarization")
    audio_16k, sr_16k = librosa.load(tmp_path, sr=16000)
    resampled_path = tmp_path.replace('.wav', '_16k.wav')
    sf.write(resampled_path, audio_16k, sr_16k)
    
    logger.info("Running speaker diarization analysis")
    diarization = diarization_pipeline(resampled_path)
    
    detected_speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        detected_speakers.add(speaker)
    
    speaker_count = len(detected_speakers)
    logger.info(f"Detected {speaker_count} speakers: {sorted(detected_speakers)}")
    
    min_segment_duration, min_gap_duration = get_processing_parameters(speaker_count)
    logger.info(f"Processing parameters: min_segment={min_segment_duration}s, min_gap={min_gap_duration}s")
    
    processed_segments = post_process_diarization(diarization, min_segment_duration, min_gap_duration)
    
    segments_out = []
    for segment in processed_segments:
        segments_out.append({
            "speaker": segment["speaker"],
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": ""
        })
    
    logger.info(f"Generated {len(segments_out)} segments for transcription")
    return segments_out, audio_16k, resampled_path

def transcribe_segment(segment, audio_16k, sr_16k=16000):
    MIN_ASR_DURATION = 0.8
    duration = segment["end"] - segment["start"]
    
    logger.debug(f"Processing segment {segment['speaker']}: {duration:.2f}s ({segment['start']:.2f}-{segment['end']:.2f})")
    
    if duration < MIN_ASR_DURATION:
        logger.warning(f"Skipping short segment: {duration:.2f}s < {MIN_ASR_DURATION}s")
        segment["text"] = "[SEGMENT_TOO_SHORT]"
        return
    
    try:
        start_sample = int(segment["start"] * sr_16k)
        end_sample = int(segment["end"] * sr_16k)
        segment_audio = audio_16k[start_sample:end_sample]
        
        if len(segment_audio) < 12800:
            logger.warning(f"Segment too short in samples: {len(segment_audio)} < 12800")
            segment["text"] = "[SEGMENT_TOO_SHORT]"
            return
        
        segment_path = f"temp_segment_{segment['start']:.2f}.wav"
        sf.write(segment_path, segment_audio, sr_16k)
        transcription = asr_model.transcribe([segment_path])[0]
        
        if hasattr(transcription, 'text'):
            text_result = transcription.text
        elif isinstance(transcription, str):
            text_result = transcription
        else:
            text_result = str(transcription)
        
        segment["text"] = text_result if text_result and text_result.strip() else "[NO_SPEECH_DETECTED]"
        os.remove(segment_path)
        logger.debug(f"Transcribed segment {segment['speaker']}: {len(text_result) if text_result else 0} characters")
        
    except Exception as e:
        logger.error(f"Transcription failed for segment {segment['speaker']}: {str(e)}")
        segment["text"] = f"[TRANSCRIPTION_ERROR: {str(e)[:50]}]"

def cleanup_memory():
    logger.debug("Cleaning up GPU memory")
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

@app.post("/transcribe-diarize")
async def transcribe_diarize(file: UploadFile = File(...)):
    logger.info(f"Received transcription request for file: {file.filename}")
    
    if not file.content_type or not file.content_type.startswith("audio/"):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    logger.info(f"Processing audio file: {file.filename}")
    
    try:
        validate_audio(tmp_path)
        segments_out, audio_16k, resampled_path = process_diarization(tmp_path)
        
        logger.info(f"Starting transcription of {len(segments_out)} segments")
        for i, segment in enumerate(segments_out, 1):
            logger.debug(f"Transcribing segment {i}/{len(segments_out)}")
            transcribe_segment(segment, audio_16k)
        
        logger.info("Transcription completed successfully")
        cleanup_memory()
        return JSONResponse({"segments": segments_out})
        
    except Exception as e:
        logger.error(f"Pipeline failed for file {file.filename}: {str(e)}")
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        resampled_path = tmp_path.replace('.wav', '_16k.wav')
        if os.path.exists(resampled_path):
            os.remove(resampled_path)
        logger.debug("Cleanup completed")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1, log_level="info")