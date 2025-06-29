import requests
import json
from pathlib import Path
import time
import sys
import os

# Add the parent directory to the path so we can import from utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config

class LaoInstrumentClassifierClient:
    """Client for the Lao Instrument Classifier API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def get_instruments(self):
        """Get list of supported instruments"""
        try:
            response = requests.get(f"{self.base_url}/instruments")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get instruments: {e}")
            return None
    
    def predict_single(self, audio_file_path: str):
        """Predict instrument from single audio file"""
        try:
            file_path = Path(audio_file_path)
            if not file_path.exists():
                print(f"File not found: {audio_file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'audio/wav')}
                
                start_time = time.time()
                response = requests.post(f"{self.base_url}/predict", files=files)
                end_time = time.time()
                
                response.raise_for_status()
                result = response.json()
                result['client_processing_time'] = (end_time - start_time) * 1000
                
                return result
                
        except requests.exceptions.RequestException as e:
            print(f"Prediction failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Error details: {e.response.text}")
            return None
    
    def predict_batch(self, audio_file_paths: list):
        """Predict instruments from multiple audio files"""
        try:
            files = []
            for file_path in audio_file_paths:
                path = Path(file_path)
                if path.exists():
                    files.append(('files', (path.name, open(path, 'rb'), 'audio/wav')))
                else:
                    print(f"Warning: File not found: {file_path}")
            
            if not files:
                print("No valid files to process")
                return None
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/predict-batch", files=files)
            end_time = time.time()
            
            # Close file handles
            for _, (_, file_handle, _) in files:
                file_handle.close()
            
            response.raise_for_status()
            result = response.json()
            result['client_processing_time'] = (end_time - start_time) * 1000
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Batch prediction failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Error details: {e.response.text}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model-info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get model info: {e}")
            return None

def format_prediction_result(result):
    """Format prediction result for display"""
    if not result:
        return "No result"
    
    output = []
    output.append(f"🎵 Predicted Instrument: {result['instrument'].upper()}")
    output.append(f"🎯 Confidence: {result['confidence']:.1%} ({result['confidence_category']})")
    output.append(f"⏱️ Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
    
    if result.get('is_uncertain'):
        output.append("⚠️ Uncertain prediction - consider recording quality")
    
    output.append("\n📊 All Probabilities:")
    for instrument, prob in sorted(result['probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True):
        output.append(f"  {instrument}: {prob:.1%}")
    
    output.append(f"\n🔍 Technical Details:")
    output.append(f"  Entropy: {result.get('entropy', 0):.3f}")
    output.append(f"  Prediction Std: {result.get('prediction_std', 0):.3f}")
    output.append(f"  Segments Used: {result.get('segments_used', 1)}")
    
    return "\n".join(output)

def check_model_files():
    """Check if model files exist and provide helpful information"""
    print("🔍 Checking model files...")
    
    issues = Config.validate_paths()
    available_files = Config.list_available_files()
    
    print(f"📁 Models directory: {Config.MODEL_DIR}")
    print(f"📂 Available files: {available_files}")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\n💡 Suggestions:")
        print("  1. Make sure you have run the training script successfully")
        print("  2. Check that the ONNX model was generated")
        print("  3. Update the file paths in utils/config.py to match your actual files")
        print("  4. Common file names to look for:")
        print("     - model.onnx, mel_cnn_model_6sec.onnx")
        print("     - label_mapping.json, model_metadata.json")
        
        return False
    else:
        print("✅ All model files found!")
        return True

def main():
    """Example usage of the API client"""
    print("🎵 Lao Instrument Classifier API Client Test")
    print("=" * 50)
    
    # Check model files first
    if not check_model_files():
        print("\n❌ Cannot proceed without model files. Please fix the issues above.")
        return
    
    # Initialize client
    client = LaoInstrumentClassifierClient()
    
    # Health check
    print("\n1. Health Check...")
    health = client.health_check()
    if health:
        print(f"   Status: {health['status']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        if not health['model_loaded']:
            print("   ❌ Model not loaded - check server logs for details")
            return
    else:
        print("   ❌ API not available - make sure the server is running")
        print("   Start server with: python -m uvicorn app.main:app --reload")
        return
    
    # Get supported instruments
    print("\n2. Supported Instruments...")
    instruments = client.get_instruments()
    if instruments:
        print(f"   Total Classes: {instruments['total_classes']}")
        for key, info in instruments['instruments'].items():
            print(f"   - {key}: {info}")
    
    # Get model info
    print("\n3. Model Information...")
    model_info = client.get_model_info()
    if model_info:
        print(f"   Classes: {', '.join(model_info['classes'])}")
        print(f"   Sample Rate: {model_info['audio_config']['sample_rate']} Hz")
        print(f"   Segment Duration: {model_info['audio_config']['segment_duration']}s")
        print(f"   Ensemble Segments: {model_info['prediction_config']['ensemble_segments']}")
    
    # Example prediction (replace with your audio file)
    print("\n4. Example Prediction...")
    
    # Look for test audio files in common locations
    test_files = [
        "test_audio.wav",
        "../test_audio.wav", 
        "sample.wav",
        "../sample.wav"
    ]
    
    audio_file = None
    for test_file in test_files:
        if Path(test_file).exists():
            audio_file = test_file
            break
    
    if audio_file:
        print(f"   Analyzing: {audio_file}")
        result = client.predict_single(audio_file)
        
        if result:
            print("   " + "="*30)
            formatted_result = format_prediction_result(result)
            for line in formatted_result.split('\n'):
                print(f"   {line}")
            print("   " + "="*30)
        else:
            print("   ❌ Prediction failed")
    else:
        print("   ℹ️ No test audio file found. Place a .wav file in the current directory to test.")
        print(f"   Expected files: {', '.join(test_files)}")
    
    # Example batch prediction
    print("\n5. Batch Prediction Example...")
    batch_files = [f for f in test_files if Path(f).exists()]
    
    if len(batch_files) > 1:
        print(f"   Analyzing batch: {batch_files}")
        batch_result = client.predict_batch(batch_files)
        
        if batch_result:
            print(f"   Processed {batch_result['total_files']} files")
            print(f"   Successful: {batch_result['successful_predictions']}")
            
            for result in batch_result['results']:
                if result['success']:
                    print(f"   ✅ {result['filename']}: {result['result']['instrument']} ({result['result']['confidence']:.1%})")
                else:
                    print(f"   ❌ {result['filename']}: {result['error']}")
    else:
        print("   ℹ️ Need multiple audio files for batch testing")
    
    print("\n✅ API test completed!")
    print("\nNext steps:")
    print("1. Place your audio files in the current directory")
    print("2. Use the client code above to integrate with your application")
    print("3. Check the API documentation at http://localhost:8000/docs")

if __name__ == "__main__":
    main()