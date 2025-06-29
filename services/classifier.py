import onnxruntime as ort
import json
import os
import numpy as np
import librosa
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from utils.config import Config
from utils.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class InstrumentClassifier:
    """Enhanced ONNX-based instrument classifier for multi-channel model"""
    
    def __init__(self):
        self.model_path = Config.MODEL_PATH
        self.label_mapping_path = Config.LABEL_MAPPING_PATH
        self.session = None
        self.idx_to_label = None
        self.model_loaded = False
        
    def load_model(self) -> Tuple[bool, str]:
        """Load enhanced ONNX model and label mapping"""
        try:
            # Try auto-detection first
            auto_success, auto_message = Config.auto_detect_enhanced_model()
            if auto_success:
                logger.info(auto_message)
                self.model_path = Config.MODEL_PATH
                self.label_mapping_path = Config.LABEL_MAPPING_PATH
            
            if not os.path.exists(self.model_path):
                available_files = Config.list_available_files()
                return False, f"Enhanced model file not found: {self.model_path}\nAvailable files: {available_files}"
                
            if not os.path.exists(self.label_mapping_path):
                return False, f"Label mapping file not found: {self.label_mapping_path}"
            
            # Load label mapping
            with open(self.label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            self.idx_to_label = {int(idx): label for label, idx in label_mapping.items()}
            
            # Create ONNX session for enhanced model
            self.session = ort.InferenceSession(self.model_path)
            
            # Check model input shape
            input_shape = self.session.get_inputs()[0].shape
            expected_channels = len(Config.FEATURE_CHANNELS)
            
            if len(input_shape) >= 4 and input_shape[-1] != expected_channels:
                logger.warning(f"Model expects {input_shape[-1]} channels, config has {expected_channels}")
            
            self.model_loaded = True
            
            logger.info(f"Enhanced model loaded successfully from {self.model_path}")
            logger.info(f"Model input shape: {input_shape}")
            logger.info(f"Available classes: {list(self.idx_to_label.values())}")
            
            return True, f"Enhanced model loaded successfully (input shape: {input_shape})"
            
        except Exception as e:
            logger.error(f"Error loading enhanced model: {str(e)}")
            return False, f"Error loading enhanced model: {str(e)}"
    
    def ensemble_predict(self, audio: np.ndarray, sr: int, 
                        confidence_threshold: float = 0.4) -> Optional[Dict]:
        """
        Make ensemble prediction with enhanced multi-channel features
        """
        if not self.model_loaded:
            return None
        
        try:
            start_time = datetime.now()
            
            # Resample if needed
            if sr != Config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
                sr = Config.SAMPLE_RATE
            
            # Get multiple segments for ensemble prediction
            segments = AudioProcessor.advanced_segment_selection(
                audio, sr, 
                Config.SEGMENT_DURATION,
                n_segments=Config.N_ENSEMBLE_SEGMENTS
            )
            
            predictions = []
            confidences = []
            all_features = []
            
            # Predict on each segment using enhanced features
            for segment in segments:
                # Extract enhanced multi-channel features
                multi_channel_features, feature_dict = AudioProcessor.extract_features(segment, sr)
                
                # Prepare input for enhanced model (expects multi-channel input)
                features_batch = np.expand_dims(multi_channel_features, axis=0).astype(np.float32)
                
                # Check input shape
                expected_shape = self.session.get_inputs()[0].shape
                if features_batch.shape[1:] != tuple(expected_shape[1:]):
                    logger.warning(f"Input shape mismatch: got {features_batch.shape}, expected {expected_shape}")
                
                # Run inference on enhanced model
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: features_batch})
                probabilities = outputs[0][0]
                
                predictions.append(probabilities)
                confidences.append(np.max(probabilities))
                all_features.append({
                    'multi_channel': multi_channel_features,
                    'feature_dict': feature_dict,
                    'segment': segment
                })
            
            # Ensemble prediction with confidence weighting
            if len(predictions) > 1:
                weights = np.array(confidences) / (np.sum(confidences) + 1e-8)
                ensemble_probs = np.average(predictions, axis=0, weights=weights)
            else:
                ensemble_probs = predictions[0]
            
            # Final prediction
            max_prob_idx = np.argmax(ensemble_probs)
            max_prob = ensemble_probs[max_prob_idx]
            instrument = self.idx_to_label[max_prob_idx]
            
            # Calculate enhanced uncertainty metrics
            entropy = -np.sum(ensemble_probs * np.log2(ensemble_probs + 1e-10)) / np.log2(len(ensemble_probs))
            prediction_std = np.std([np.max(p) for p in predictions])
            
            # Enhanced uncertainty detection
            is_uncertain = (entropy > 0.6 or 
                          max_prob < confidence_threshold or 
                          prediction_std > 0.15)
            
            # Use the best segment's features for visualization
            best_segment_idx = np.argmax(confidences)
            best_features = all_features[best_segment_idx]
            
            # Processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check if this is a difficult instrument (khean, pin, saw)
            difficult_instruments = ['khean', 'pin', 'saw']
            is_difficult = instrument in difficult_instruments
            
            # Enhanced result with multi-channel analysis
            result = {
                'instrument': instrument,
                'confidence': float(max_prob),
                'entropy': float(entropy),
                'prediction_std': float(prediction_std),
                'is_uncertain': is_uncertain,
                'is_difficult': is_difficult,
                'segments_used': len(predictions),
                'individual_confidences': [float(c) for c in confidences],
                'probabilities': {self.idx_to_label[i]: float(prob) 
                                for i, prob in enumerate(ensemble_probs)},
                'enhanced_features': {
                    'multi_channel_shape': best_features['multi_channel'].shape,
                    'channels_used': Config.FEATURE_CHANNELS,
                    'hpss_enabled': Config.USE_HPSS
                },
                'confidence_category': self._get_confidence_category(max_prob, entropy, prediction_std),
                'processing_time_ms': processing_time
            }
            
            # Add analysis for difficult instruments
            if is_difficult:
                result['difficulty_analysis'] = self._analyze_difficult_instrument(
                    instrument, ensemble_probs, best_features['feature_dict']
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced prediction error: {str(e)}")
            return None
    
    def _get_confidence_category(self, confidence: float, entropy: float, std: float) -> str:
        """Enhanced confidence categorization"""
        if confidence > 0.85 and entropy < 0.3 and std < 0.08:
            return "Very High"
        elif confidence > 0.7 and entropy < 0.5 and std < 0.12:
            return "High"
        elif confidence > 0.5 and entropy < 0.7 and std < 0.18:
            return "Medium"
        elif confidence > 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _analyze_difficult_instrument(self, instrument: str, probabilities: np.ndarray, 
                                    features: dict) -> dict:
        """Analyze predictions for difficult instruments (khean, pin, saw)"""
        difficult_mapping = {
            'khean': ['pin', 'saw'],
            'pin': ['khean', 'saw'], 
            'saw': ['khean', 'pin']
        }
        
        confusion_candidates = difficult_mapping.get(instrument, [])
        
        analysis = {
            'instrument': instrument,
            'difficulty_reason': 'Similar harmonic characteristics',
            'confusion_probabilities': {},
            'feature_analysis': {}
        }
        
        # Get probabilities for confusing instruments
        for candidate in confusion_candidates:
            for idx, label in self.idx_to_label.items():
                if label == candidate:
                    analysis['confusion_probabilities'][candidate] = float(probabilities[idx])
        
        # Analyze HPSS features if available
        if 'harmonic' in features and 'percussive' in features:
            harmonic_energy = np.mean(features['harmonic']**2)
            percussive_energy = np.mean(features['percussive']**2)
            hp_ratio = harmonic_energy / (percussive_energy + 1e-8)
            
            analysis['feature_analysis'] = {
                'harmonic_energy': float(harmonic_energy),
                'percussive_energy': float(percussive_energy),
                'harmonic_percussive_ratio': float(hp_ratio),
                'interpretation': self._interpret_hp_ratio(instrument, hp_ratio)
            }
        
        return analysis
    
    def _interpret_hp_ratio(self, instrument: str, hp_ratio: float) -> str:
        """Interpret harmonic-percussive ratio for difficult instruments"""
        if instrument == 'khean':
            if hp_ratio > 2.0:
                return "Strong harmonic content consistent with khaen"
            else:
                return "Lower harmonic ratio - might be confused with plucked instruments"
        elif instrument == 'pin':
            if hp_ratio < 1.5:
                return "Good percussive attack pattern for pin"
            else:
                return "High harmonic content - might be confused with khaen"
        elif instrument == 'saw':
            if 1.0 < hp_ratio < 3.0:
                return "Balanced harmonic-percussive ratio typical for bowed instruments"
            else:
                return "Unusual ratio for saw - check bow technique"
        return "Analysis not available for this instrument"