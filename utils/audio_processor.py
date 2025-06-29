import numpy as np
import librosa
import logging
from typing import List, Optional, Tuple
from utils.config import Config

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Enhanced audio processing class matching the new training pipeline"""
    
    @staticmethod
    def extract_enhanced_features(audio: np.ndarray, sr: int) -> dict:
        """
        Extract multi-channel features including HPSS - MATCHES TRAINING EXACTLY
        """
        features = {}
        
        # 1. Basic Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            fmax=Config.FMAX
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel'] = mel_spec_db
        
        # 2. Harmonic-Percussive Source Separation
        try:
            harmonic, percussive = librosa.effects.hpss(
                audio, 
                margin=(1.0, 5.0)  # Match training config
            )
            
            # Harmonic mel spectrogram
            harmonic_mel = librosa.feature.melspectrogram(
                y=harmonic, sr=sr,
                n_fft=Config.N_FFT,
                hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS,
                fmax=Config.FMAX
            )
            features['harmonic'] = librosa.power_to_db(harmonic_mel, ref=np.max)
            
            # Percussive mel spectrogram
            percussive_mel = librosa.feature.melspectrogram(
                y=percussive, sr=sr,
                n_fft=Config.N_FFT,
                hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS,
                fmax=Config.FMAX
            )
            features['percussive'] = librosa.power_to_db(percussive_mel, ref=np.max)
        except Exception as e:
            logger.warning(f"HPSS failed, using original audio: {e}")
            # Fallback to duplicating mel spectrogram
            features['harmonic'] = features['mel'].copy()
            features['percussive'] = features['mel'].copy()
        
        # 3. MFCCs and derivatives
        try:
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr,
                n_mfcc=20,  # Match training config
                n_fft=Config.N_FFT,
                hop_length=Config.HOP_LENGTH
            )
            
            # Pad MFCCs to match mel spectrogram dimensions
            if mfcc.shape[0] < Config.N_MELS:
                pad_width = Config.N_MELS - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            
            features['mfcc'] = mfcc
            
            # Delta and delta-delta features
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            features['delta'] = delta
            features['delta2'] = delta2
            
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            # Fallback to zeros
            features['mfcc'] = np.zeros((Config.N_MELS, features['mel'].shape[1]))
            features['delta'] = np.zeros((Config.N_MELS, features['mel'].shape[1]))
            features['delta2'] = np.zeros((Config.N_MELS, features['mel'].shape[1]))
        
        return features

    @staticmethod
    def create_multi_channel_input(features: dict) -> np.ndarray:
        """
        Create multi-channel input from extracted features - MATCHES TRAINING
        """
        channels = ['mel', 'harmonic', 'percussive', 'mfcc', 'delta', 'delta2']
        channel_data = []
        
        for channel in channels:
            if channel in features:
                data = features[channel]
                # Normalize each channel
                data_norm = (data - data.mean()) / (data.std() + 1e-8)
                channel_data.append(data_norm)
            else:
                # Fallback to zeros if channel missing
                logger.warning(f"Missing channel: {channel}")
                data_shape = features['mel'].shape
                channel_data.append(np.zeros(data_shape))
        
        # Stack along the channel dimension: (height, width, channels)
        multi_channel = np.stack(channel_data, axis=-1)
        return multi_channel

    @staticmethod
    def advanced_segment_selection(audio: np.ndarray, sr: int, 
                                 segment_duration: float = 6.0, 
                                 n_segments: int = 3) -> List[np.ndarray]:
        """
        Advanced segment selection focusing on information-rich portions - MATCHES TRAINING
        """
        segment_len = int(segment_duration * sr)
        
        if len(audio) <= segment_len:
            return [np.pad(audio, (0, segment_len - len(audio)), mode='constant')]
        
        # Extract onset strength for finding musically relevant segments
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        except:
            onset_env = np.ones(len(audio) // 512)  # Fallback
        
        # Calculate energy and spectral features for each possible segment
        hop_len = segment_len // 4
        segments = []
        scores = []
        
        for start in range(0, len(audio) - segment_len + 1, hop_len):
            segment = audio[start:start + segment_len]
            
            # Multi-criteria scoring
            try:
                # Energy-based score
                rms = np.sqrt(np.mean(segment**2))
                
                # Spectral complexity score
                spectral_cent = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
                spectral_bw = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
                
                # Onset density score
                segment_onset_start = start // 512
                segment_onset_end = min((start + segment_len) // 512, len(onset_env))
                if segment_onset_end > segment_onset_start:
                    onset_density = np.mean(onset_env[segment_onset_start:segment_onset_end])
                else:
                    onset_density = 0
                
                # Combined score (same as training)
                score = (rms * 0.3 + 
                        (spectral_cent / 4000) * 0.3 + 
                        (spectral_bw / 4000) * 0.2 + 
                        onset_density * 0.2)
                
            except Exception as e:
                logger.warning(f"Scoring failed for segment at {start}: {e}")
                score = np.sqrt(np.mean(segment**2))  # Fallback to RMS
            
            segments.append(segment)
            scores.append(score)
        
        # Select diverse high-scoring segments
        if len(segments) <= n_segments:
            return segments
        
        # Get top scoring segments with diversity
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = []
        
        for idx in sorted_indices:
            # Check if this segment is far enough from already selected ones
            if not selected_indices:
                selected_indices.append(idx)
            else:
                min_distance = min([abs(idx - sel_idx) for sel_idx in selected_indices])
                if min_distance >= 2:  # At least 2 hops apart
                    selected_indices.append(idx)
            
            if len(selected_indices) >= n_segments:
                break
        
        return [segments[i] for i in selected_indices]
    
    @staticmethod
    def extract_features(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, dict]:
        """
        Main feature extraction method for the enhanced model
        """
        # Get best segments
        segments = AudioProcessor.advanced_segment_selection(
            audio, sr, 
            Config.SEGMENT_DURATION,
            n_segments=1  # Use best segment for feature extraction
        )
        
        best_segment = segments[0]
        
        # Extract enhanced features
        features = AudioProcessor.extract_enhanced_features(best_segment, sr)
        
        # Create multi-channel input
        multi_channel = AudioProcessor.create_multi_channel_input(features)
        
        return multi_channel, features