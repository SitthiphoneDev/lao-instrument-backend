import os
from pathlib import Path

class Config:
    """Enhanced configuration matching the new training pipeline exactly"""
    
    # Audio parameters (must match training)
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Enhanced feature extraction parameters (match EnhancedAudioConfig)
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    N_MFCC = 20
    FMAX = 8000
    
    # HPSS parameters
    HPSS_MARGIN = (1.0, 5.0)  # (harmonic_margin, percussive_margin)
    USE_HPSS = True
    
    # Feature channels configuration
    USE_MULTI_CHANNEL = True
    FEATURE_CHANNELS = ['mel', 'harmonic', 'percussive', 'mfcc', 'delta', 'delta2']
    
    # Model paths - update these based on your new model
    BASE_DIR = Path(__file__).parent.parent  # Points to project root
    MODEL_DIR = BASE_DIR / "models"
    
    # NEW: Use the enhanced model files
    # Update these paths to match your actual enhanced model files
    MODEL_PATH = str(MODEL_DIR / "enhanced_model.onnx")
    LABEL_MAPPING_PATH = str(MODEL_DIR / "label_mapping.json")
    
    # Alternative paths if you have a timestamped model directory
    # MODEL_PATH = str(MODEL_DIR / "enhanced_model_gemini_20241229_123456" / "enhanced_model.onnx")
    # LABEL_MAPPING_PATH = str(MODEL_DIR / "enhanced_model_gemini_20241229_123456" / "label_mapping.json")
    
    # Prediction parameters
    CONFIDENCE_THRESHOLD = 0.4
    N_ENSEMBLE_SEGMENTS = 3
    
    @classmethod
    def validate_paths(cls):
        """Validate that required files exist"""
        issues = []
        
        if not os.path.exists(cls.MODEL_PATH):
            issues.append(f"Enhanced model file not found: {cls.MODEL_PATH}")
            
        if not os.path.exists(cls.LABEL_MAPPING_PATH):
            issues.append(f"Label mapping file not found: {cls.LABEL_MAPPING_PATH}")
            
        return issues
    
    @classmethod
    def list_available_files(cls):
        """List available files in models directory for debugging"""
        model_dir = cls.MODEL_DIR
        files = []
        
        if model_dir.exists():
            # List all files and subdirectories
            for item in model_dir.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(model_dir)
                    files.append(str(relative_path))
        
        return files
    
    @classmethod
    def auto_detect_enhanced_model(cls):
        """Auto-detect the latest enhanced model"""
        model_dir = cls.MODEL_DIR
        
        if not model_dir.exists():
            return False, "Models directory not found"
        
        # Look for enhanced model directories
        enhanced_dirs = []
        for item in model_dir.iterdir():
            if item.is_dir() and "enhanced_model" in item.name.lower():
                enhanced_dirs.append(item)
        
        if not enhanced_dirs:
            return False, "No enhanced model directories found"
        
        # Get the most recent one
        latest_dir = max(enhanced_dirs, key=lambda x: x.stat().st_mtime)
        
        # Check for required files
        onnx_file = latest_dir / "enhanced_model.onnx"
        label_file = latest_dir / "label_mapping.json"
        
        if onnx_file.exists() and label_file.exists():
            cls.MODEL_PATH = str(onnx_file)
            cls.LABEL_MAPPING_PATH = str(label_file)
            return True, f"Auto-detected enhanced model: {latest_dir.name}"
        
        return False, f"Enhanced model files not found in {latest_dir.name}"