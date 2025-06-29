from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import librosa
import onnxruntime as ort
import json
import os
import tempfile
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import io
from pydantic import BaseModel
from services.classifier import InstrumentClassifier
from utils.config import Config
from utils.audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response models
class PredictionResult(BaseModel):
    instrument: str
    confidence: float
    probabilities: Dict[str, float]
    confidence_category: str
    is_uncertain: bool
    entropy: float
    prediction_std: float
    segments_used: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Instrument information
INSTRUMENT_INFO = {
    'khean': 'Khaen (ແຄນ) - Traditional Lao mouth organ',
    'khong_vong': 'Khong Wong (ຄ້ອງວົງ) - Circular gong arrangement',
    'pin': 'Pin (ພິນ) - Plucked string instrument',
    'ranad': 'Ranad (ລະນາດ) - Wooden xylophone',
    'saw': 'So U (ຊໍອູ້) - Two-stringed bowed instrument',
    'ranad': 'Sing (ຊິ່ງ) - Small cymbals',
    'unknown': 'Unknown sound - Not a recognized instrument'
}

# Initialize FastAPI app
app = FastAPI(
    title="Lao Instrument Classifier API",
    description="AI-powered recognition of traditional Lao musical instruments",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier
classifier = InstrumentClassifier()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success, message = classifier.load_model()
    if success:
        logger.info("✅ Model loaded successfully on startup")
    else:
        logger.error(f"❌ Failed to load model on startup: {message}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if classifier.model_loaded else "unhealthy",
        model_loaded=classifier.model_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResult)
async def predict_instrument(file: UploadFile = File(...)):
    """
    Predict instrument from uploaded audio file
    
    Supports: WAV, MP3, OGG, M4A, FLAC
    """
    if not classifier.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check file type
    allowed_types = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/ogg', 'audio/vorbis',
        'audio/mp4', 'audio/m4a',
        'audio/flac'
    }
    
    if file.content_type not in allowed_types and not any(
        file.filename.lower().endswith(ext) 
        for ext in ['.wav', '.mp3', '.ogg', '.m4a', '.flac']
    ):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: WAV, MP3, OGG, M4A, FLAC"
        )
    
    try:
        # Read file
        contents = await file.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Load audio
            audio_data, sr = librosa.load(tmp_path, sr=Config.SAMPLE_RATE)
            
            # Check duration
            duration = len(audio_data) / sr
            if duration < 1.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio too short ({duration:.1f}s). Minimum 1 second required."
                )
            
            # Make prediction
            result = classifier.ensemble_predict(audio_data, sr)
            
            if result is None:
                raise HTTPException(status_code=500, detail="Prediction failed")
            
            return PredictionResult(**result)
        
        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except:
                pass
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict instruments from multiple audio files
    """
    if not classifier.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each file
            contents = await file.read()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            try:
                audio_data, sr = librosa.load(tmp_path, sr=Config.SAMPLE_RATE)
                result = classifier.ensemble_predict(audio_data, sr)
                
                if result:
                    results.append({
                        "filename": file.filename,
                        "index": i,
                        "success": True,
                        "result": result
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "index": i,
                        "success": False,
                        "error": "Prediction failed"
                    })
            
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "successful_predictions": sum(1 for r in results if r["success"]),
        "results": results
    }

@app.get("/model-info")
async def get_model_info():
    """Get model information and configuration"""
    if not classifier.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": Config.MODEL_PATH,
        "classes": list(classifier.idx_to_label.values()) if classifier.idx_to_label else [],
        "total_classes": len(classifier.idx_to_label) if classifier.idx_to_label else 0,
        "audio_config": {
            "sample_rate": Config.SAMPLE_RATE,
            "segment_duration": Config.SEGMENT_DURATION,
            "n_mels": Config.N_MELS,
            "n_fft": Config.N_FFT,
            "hop_length": Config.HOP_LENGTH,
            "fmax": Config.FMAX
        },
        "prediction_config": {
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
            "ensemble_segments": Config.N_ENSEMBLE_SEGMENTS
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Lao Instrument Classifier API",
        "version": "2.0.0",
        "description": "AI-powered recognition of traditional Lao musical instruments",
        "model_loaded": classifier.model_loaded,
        "supported_instruments": list(INSTRUMENT_INFO.keys()),
        "endpoints": {
            "predict": "/predict - Upload audio file for prediction",
            "predict_batch": "/predict-batch - Upload multiple files",
            "health": "/health - Health check",
            "instruments": "/instruments - List supported instruments",
            "model_info": "/model-info - Model configuration",
            "docs": "/docs - API documentation"
        }
    }

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )