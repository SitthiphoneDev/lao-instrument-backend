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

# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    is_difficult: Optional[bool] = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    environment: str = "render"

# Initialize FastAPI app with Render-optimized settings
app = FastAPI(
    title="Lao Instrument Classifier API",
    description="AI-powered recognition of traditional Lao musical instruments (Enhanced HPSS Model)",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - updated for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local Next.js dev
        "https://*.vercel.app",   # Vercel deployments
        "https://*.netlify.app",  # Netlify deployments
        "https://*.render.com",   # Other Render services
        "*"  # Remove this in production, add your specific domains
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize classifier
classifier = InstrumentClassifier()

@app.on_event("startup")
async def startup_event():
    """Load model on startup with Render-specific logging"""
    logger.info("🚀 Starting Lao Instrument Classifier API on Render...")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Available files: {os.listdir('.')}")
    
    # Check if models directory exists
    if os.path.exists("models"):
        logger.info(f"Models directory contents: {os.listdir('models')}")
    else:
        logger.warning("Models directory not found!")
    
    success, message = classifier.load_model()
    if success:
        logger.info(f"✅ {message}")
    else:
        logger.error(f"❌ {message}")
        logger.info("Available files for debugging:")
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(('.onnx', '.json')):
                    logger.info(f"Found: {os.path.join(root, file)}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with Render deployment info"""
    return {
        "name": "Lao Instrument Classifier API",
        "version": "2.1.0",
        "description": "Enhanced AI model with HPSS features for Lao instruments",
        "model_loaded": classifier.model_loaded,
        "environment": "Render",
        "model_type": "Enhanced Multi-Channel ONNX",
        "supported_instruments": [
            "khean", "khong_vong", "pin", "ranad", "saw", "sing", "unknown"
        ],
        "endpoints": {
            "predict": "POST /predict - Upload audio file",
            "predict_batch": "POST /predict-batch - Multiple files",
            "health": "GET /health - Health check",
            "model_info": "GET /model-info - Model details",
            "docs": "GET /docs - API documentation"
        },
        "features": [
            "Harmonic-Percussive Source Separation (HPSS)",
            "6-channel multi-spectral analysis", 
            "Ensemble prediction with uncertainty estimation",
            "Special handling for difficult instruments (khean/pin/saw)"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Render monitoring"""
    return HealthResponse(
        status="healthy" if classifier.model_loaded else "unhealthy",
        model_loaded=classifier.model_loaded,
        timestamp=datetime.now().isoformat(),
        environment="render"
    )

@app.post("/predict", response_model=PredictionResult)
async def predict_instrument(file: UploadFile = File(...)):
    """
    Predict instrument from uploaded audio file
    
    **Enhanced Model Features:**
    - HPSS (Harmonic-Percussive Source Separation)
    - 6-channel spectral analysis
    - Improved accuracy for khean/pin/saw discrimination
    
    **Supported formats:** WAV, MP3, OGG, M4A, FLAC
    """
    if not classifier.model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Enhanced model not loaded. Please check server logs."
        )
    
    # Enhanced file type checking
    allowed_types = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/ogg', 'audio/vorbis',
        'audio/mp4', 'audio/m4a', 'audio/x-m4a',
        'audio/flac'
    }
    
    allowed_extensions = ['.wav', '.mp3', '.ogg', '.m4a', '.flac']
    
    file_valid = (
        file.content_type in allowed_types or 
        any(file.filename.lower().endswith(ext) for ext in allowed_extensions)
    )
    
    if not file_valid:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Supported: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Process the uploaded file
        contents = await file.read()
        
        # Enhanced file size check
        file_size_mb = len(contents) / (1024 * 1024)
        if file_size_mb > 50:  # 50MB limit for Render
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum 50MB allowed."
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Load audio with enhanced error handling
            try:
                audio_data, sr = librosa.load(tmp_path, sr=Config.SAMPLE_RATE)
            except Exception as load_error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load audio file: {str(load_error)}"
                )
            
            # Enhanced duration check
            duration = len(audio_data) / sr
            if duration < 0.5:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio too short ({duration:.1f}s). Minimum 0.5 seconds required."
                )
            
            if duration > 300:  # 5 minutes max
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio too long ({duration:.1f}s). Maximum 5 minutes allowed."
                )
            
            # Make enhanced prediction
            result = classifier.ensemble_predict(audio_data, sr)
            
            if result is None:
                raise HTTPException(
                    status_code=500, 
                    detail="Enhanced prediction failed. Please try again."
                )
            
            logger.info(f"Prediction: {result['instrument']} ({result['confidence']:.3f})")
            
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
        logger.error(f"Unexpected error processing file: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Get enhanced model information and configuration"""
    if not classifier.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Enhanced Multi-Channel ONNX",
        "model_path": Config.MODEL_PATH,
        "classes": list(classifier.idx_to_label.values()) if classifier.idx_to_label else [],
        "total_classes": len(classifier.idx_to_label) if classifier.idx_to_label else 0,
        "enhanced_features": {
            "hpss_enabled": Config.USE_HPSS,
            "multi_channel": Config.USE_MULTI_CHANNEL,
            "feature_channels": Config.FEATURE_CHANNELS,
            "hpss_margin": Config.HPSS_MARGIN
        },
        "audio_config": {
            "sample_rate": Config.SAMPLE_RATE,
            "segment_duration": Config.SEGMENT_DURATION,
            "n_mels": Config.N_MELS,
            "n_mfcc": Config.N_MFCC,
            "n_fft": Config.N_FFT,
            "hop_length": Config.HOP_LENGTH,
            "fmax": Config.FMAX
        },
        "prediction_config": {
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
            "ensemble_segments": Config.N_ENSEMBLE_SEGMENTS
        },
        "deployment": {
            "platform": "Render",
            "environment": "production"
        }
    }

# Error handlers for better debugging on Render
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,  # Disable reload in production
        log_level="info"
    )