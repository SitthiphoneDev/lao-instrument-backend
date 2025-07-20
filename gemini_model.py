import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import tf2onnx
import onnx
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Enable GPU growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU devices available: {len(physical_devices)}")
    except:
        pass

print(f"TensorFlow version: {tf.__version__}")

class EnhancedAudioConfig:
    """Enhanced configuration with advanced feature engineering"""
    
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Enhanced feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    N_MFCC = 20
    FMAX = 8000
    
    # HPSS parameters for harmonic-percussive separation
    HPSS_MARGIN = (1.0, 5.0)  # (harmonic_margin, percussive_margin)
    USE_HPSS = True
    
    # Feature channels configuration
    USE_MULTI_CHANNEL = True
    FEATURE_CHANNELS = ['mel', 'harmonic', 'percussive', 'mfcc', 'delta', 'delta2']
    
    # Advanced augmentation parameters
    USE_SPEC_AUGMENT = True
    FREQ_MASK_PARAM = 27  # Maximum frequency channels to mask
    TIME_MASK_PARAM = 100  # Maximum time steps to mask
    NUM_FREQ_MASKS = 2
    NUM_TIME_MASKS = 2
    
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    
    # Training parameters
    BATCH_SIZE = 16  # Smaller for multi-channel features
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    EARLY_STOPPING_PATIENCE = 20
    
    # Advanced loss configuration
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25
    
    # Model architecture
    USE_ATTENTION = True
    DROPOUT_RATE = 0.5
    L2_REGULARIZATION = 0.01
    
    # Enhanced class weights for difficult instruments
    USE_ENHANCED_WEIGHTS = True
    DIFFICULT_INSTRUMENTS = ['khean', 'pin', 'saw']
    DIFFICULT_WEIGHT_BOOST = 1.5
    
    # Data parameters
    K_FOLDS = 2
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/enhanced_model_gemini"
    
    # Instrument mapping
    INSTRUMENT_MAPPING = {
        'khean': ['khean', 'khaen', 'แคน', 'ແຄນ'],
        'khong_vong': ['khong', 'kong', 'ຄ້ອງວົງ', 'khong_vong'],
        'pin': ['pin', 'ພິນ'],
        'ranad': ['ranad', 'nad', 'ລະນາດ'],
        'saw': ['saw', 'so', 'ຊໍ', 'ຊໍອູ້'],
        'sing': ['sing', 'ຊິ່ງ'],
        'unknown': ['unknown', 'other', 'misc']
    }

# Create model directory
model_path = f"{EnhancedAudioConfig.MODEL_SAVE_PATH}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_path, exist_ok=True)

def extract_enhanced_features(audio, sr):
    """
    Extract multi-channel features including HPSS, MFCCs, and derivatives
    """
    features = {}
    
    # 1. Basic Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=EnhancedAudioConfig.N_FFT,
        hop_length=EnhancedAudioConfig.HOP_LENGTH,
        n_mels=EnhancedAudioConfig.N_MELS,
        fmax=EnhancedAudioConfig.FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel'] = mel_spec_db
    
    if EnhancedAudioConfig.USE_HPSS:
        # 2. Harmonic-Percussive Source Separation
        harmonic, percussive = librosa.effects.hpss(
            audio, 
            margin=EnhancedAudioConfig.HPSS_MARGIN
        )
        
        # Harmonic mel spectrogram
        harmonic_mel = librosa.feature.melspectrogram(
            y=harmonic, sr=sr,
            n_fft=EnhancedAudioConfig.N_FFT,
            hop_length=EnhancedAudioConfig.HOP_LENGTH,
            n_mels=EnhancedAudioConfig.N_MELS,
            fmax=EnhancedAudioConfig.FMAX
        )
        features['harmonic'] = librosa.power_to_db(harmonic_mel, ref=np.max)
        
        # Percussive mel spectrogram
        percussive_mel = librosa.feature.melspectrogram(
            y=percussive, sr=sr,
            n_fft=EnhancedAudioConfig.N_FFT,
            hop_length=EnhancedAudioConfig.HOP_LENGTH,
            n_mels=EnhancedAudioConfig.N_MELS,
            fmax=EnhancedAudioConfig.FMAX
        )
        features['percussive'] = librosa.power_to_db(percussive_mel, ref=np.max)
    
    # 3. MFCCs and derivatives
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=EnhancedAudioConfig.N_MFCC,
        n_fft=EnhancedAudioConfig.N_FFT,
        hop_length=EnhancedAudioConfig.HOP_LENGTH
    )
    
    # Pad MFCCs to match mel spectrogram dimensions
    if mfcc.shape[0] < EnhancedAudioConfig.N_MELS:
        pad_width = EnhancedAudioConfig.N_MELS - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    
    features['mfcc'] = mfcc
    
    # Delta and delta-delta features
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    features['delta'] = delta
    features['delta2'] = delta2
    
    return features

def create_multi_channel_input(features, channels=None):
    """
    Create multi-channel input from extracted features
    """
    if channels is None:
        channels = EnhancedAudioConfig.FEATURE_CHANNELS
    
    # Stack selected channels
    channel_data = []
    for channel in channels:
        if channel in features:
            # Normalize each channel
            data = features[channel]
            data_norm = (data - data.mean()) / (data.std() + 1e-8)
            channel_data.append(data_norm)
    
    # Stack along the channel dimension
    if len(channel_data) > 0:
        multi_channel = np.stack(channel_data, axis=-1)
    else:
        # Fallback to single channel
        multi_channel = np.expand_dims(features['mel'], axis=-1)
    
    return multi_channel

def spec_augment(mel_spectrogram, num_freq_masks=2, num_time_masks=2):
    """
    Apply SpecAugment: frequency and time masking
    """
    augmented = mel_spectrogram.copy()
    freq_size, time_size = augmented.shape[:2]
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, EnhancedAudioConfig.FREQ_MASK_PARAM)
        f0 = np.random.randint(0, max(1, freq_size - f))
        augmented[f0:f0+f, :] = 0
    
    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, EnhancedAudioConfig.TIME_MASK_PARAM)
        t0 = np.random.randint(0, max(1, time_size - t))
        augmented[:, t0:t0+t] = 0
    
    return augmented

def mixup_batch(X_batch, y_batch, alpha=0.4):
    """
    Apply Mixup augmentation to a batch
    """
    batch_size = X_batch.shape[0]
    indices = np.random.permutation(batch_size)
    
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Mix inputs and labels
    X_mixed = lam * X_batch + (1 - lam) * X_batch[indices]
    y_mixed = lam * y_batch + (1 - lam) * y_batch[indices]
    
    return X_mixed, y_mixed

class AttentionLayer(tf.keras.layers.Layer):
    """
    Custom attention layer for focusing on important frequency bands
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Calculate attention scores
        attention_scores = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention
        weighted_input = x * attention_weights
        return weighted_input

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for addressing class imbalance and hard examples
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip prediction to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = tf.where(tf.equal(y_true, 1), 
                               tf.pow(1 - p_t, gamma), 
                               tf.pow(p_t, gamma))
        
        # Compute cross entropy
        cross_entropy = -tf.math.log(p_t)
        
        # Apply focal loss
        loss = alpha_factor * focal_weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return focal_loss_fixed

def build_enhanced_model(input_shape, num_classes):
    """
    Build enhanced CNN model with attention and advanced features
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Initial convolution block
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Second block with increased filters
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Third block
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    
    # Fourth block with more filters
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Attention mechanism if enabled
    if EnhancedAudioConfig.USE_ATTENTION:
        x = tf.keras.layers.Reshape((-1, 1))(x)
        x = AttentionLayer()(x)
        x = tf.keras.layers.Flatten()(x)
    
    # Dense layers with regularization
    x = tf.keras.layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(EnhancedAudioConfig.L2_REGULARIZATION)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(EnhancedAudioConfig.DROPOUT_RATE)(x)
    
    x = tf.keras.layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(EnhancedAudioConfig.L2_REGULARIZATION)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(EnhancedAudioConfig.DROPOUT_RATE)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def advanced_segment_selection(audio, sr, segment_duration=6.0, n_segments=3):
    """
    Advanced segment selection focusing on information-rich portions
    """
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return [np.pad(audio, (0, segment_len - len(audio)), mode='constant')]
    
    # Extract onset strength for finding musically relevant segments
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    
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
            segment_onset_end = (start + segment_len) // 512
            onset_density = np.mean(onset_env[segment_onset_start:segment_onset_end])
            
            # Combined score
            score = (rms * 0.3 + 
                    (spectral_cent / 4000) * 0.3 + 
                    (spectral_bw / 4000) * 0.2 + 
                    onset_density * 0.2)
            
        except:
            score = np.sqrt(np.mean(segment**2))
        
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

def create_enhanced_class_weights(y_train, class_names):
    """
    Create enhanced class weights with special focus on difficult instruments
    """
    # Compute base weights
    unique_classes = np.unique(y_train)
    base_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    
    # Apply boost to difficult instruments
    if EnhancedAudioConfig.USE_ENHANCED_WEIGHTS:
        label_to_name = {i: name for i, name in enumerate(class_names)}
        
        enhanced_weights = {}
        for i, weight in enumerate(base_weights):
            class_name = label_to_name.get(i, '')
            if class_name in EnhancedAudioConfig.DIFFICULT_INSTRUMENTS:
                enhanced_weights[i] = weight * EnhancedAudioConfig.DIFFICULT_WEIGHT_BOOST
            else:
                enhanced_weights[i] = weight
        
        return enhanced_weights
    
    return dict(enumerate(base_weights))

class EnhancedDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator with on-the-fly augmentation
    """
    def __init__(self, X, y, batch_size=32, augment=True, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X_batch = self.X[indices].copy()
        y_batch = self.y[indices].copy()
        
        if self.augment:
            # Apply SpecAugment
            if EnhancedAudioConfig.USE_SPEC_AUGMENT:
                for i in range(len(X_batch)):
                    if np.random.random() < 0.5:
                        X_batch[i] = spec_augment(
                            X_batch[i],
                            num_freq_masks=EnhancedAudioConfig.NUM_FREQ_MASKS,
                            num_time_masks=EnhancedAudioConfig.NUM_TIME_MASKS
                        )
            
            # Apply Mixup
            if EnhancedAudioConfig.USE_MIXUP and np.random.random() < 0.5:
                X_batch, y_batch = mixup_batch(X_batch, y_batch, EnhancedAudioConfig.MIXUP_ALPHA)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def process_enhanced_dataset():
    """
    Process dataset with enhanced feature extraction
    """
    print("Processing dataset with enhanced features...")
    
    # Collect files
    all_files = []
    all_labels = []
    
    instrument_folders = [d for d in os.listdir(EnhancedAudioConfig.DATA_PATH) 
                         if os.path.isdir(os.path.join(EnhancedAudioConfig.DATA_PATH, d))]
    
    # Map folders to instruments
    for folder in instrument_folders:
        instrument = None
        folder_lower = folder.lower()
        
        for standard_name, variants in EnhancedAudioConfig.INSTRUMENT_MAPPING.items():
            for variant in variants:
                if variant.lower() in folder_lower:
                    instrument = standard_name
                    break
            if instrument:
                break
        
        if not instrument:
            continue
        
        folder_path = os.path.join(EnhancedAudioConfig.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            all_files.append(file_path)
            all_labels.append(instrument)
    
    print(f"Total files found: {len(all_files)}")
    
    # Get unique class names
    class_names = sorted(list(set(all_labels)))
    print(f"Classes: {class_names}")
    
    # Train-test split
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=EnhancedAudioConfig.TEST_SIZE,
        random_state=EnhancedAudioConfig.RANDOM_SEED,
        stratify=all_labels
    )
    
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
    
    # Process files with enhanced features
    X_train = []
    y_train = []
    
    print("Extracting enhanced features for training set...")
    for file_path, label in tqdm(zip(train_files, train_labels), total=len(train_files)):
        try:
            audio, sr = librosa.load(file_path, sr=EnhancedAudioConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 2:  # Skip very short files
                continue
            
            # Get multiple segments
            segments = advanced_segment_selection(
                audio, sr, 
                EnhancedAudioConfig.SEGMENT_DURATION,
                n_segments=2  # 2 segments per file for training
            )
            
            for segment in segments:
                # Extract enhanced features
                features = extract_enhanced_features(segment, sr)
                
                # Create multi-channel input
                if EnhancedAudioConfig.USE_MULTI_CHANNEL:
                    multi_channel = create_multi_channel_input(features)
                else:
                    multi_channel = np.expand_dims(features['mel'], axis=-1)
                
                X_train.append(multi_channel)
                y_train.append(label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Process test files
    X_test = []
    y_test = []
    
    print("Extracting enhanced features for test set...")
    for file_path, label in tqdm(zip(test_files, test_labels), total=len(test_files)):
        try:
            audio, sr = librosa.load(file_path, sr=EnhancedAudioConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 2:
                continue
            
            # Single best segment for testing
            segments = advanced_segment_selection(
                audio, sr, 
                EnhancedAudioConfig.SEGMENT_DURATION,
                n_segments=1
            )
            
            segment = segments[0]
            
            # Extract enhanced features
            features = extract_enhanced_features(segment, sr)
            
            # Create multi-channel input
            if EnhancedAudioConfig.USE_MULTI_CHANNEL:
                multi_channel = create_multi_channel_input(features)
            else:
                multi_channel = np.expand_dims(features['mel'], axis=-1)
            
            X_test.append(multi_channel)
            y_test.append(label)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Convert to arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nFinal dataset shape:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Number of channels: {X_train.shape[-1]}")
    
    return X_train, X_test, y_train, y_test, class_names

def train_enhanced_model(X_train, y_train, X_test, y_test, class_names):
    """
    Train the enhanced model with advanced techniques
    """
    print("\nTraining enhanced model...")
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_train_encoded = np.array([label_to_int[label] for label in y_train])
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    # Convert to categorical for focal loss
    y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, len(class_names))
    y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, len(class_names))
    
    # Save label mapping
    with open(os.path.join(model_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_to_int, f, indent=4)
    
    # Create enhanced class weights
    class_weights = create_enhanced_class_weights(y_train_encoded, class_names)
    
    print("\nClass weights:")
    for i, weight in class_weights.items():
        print(f"  {class_names[i]}: {weight:.3f}")
    
    # K-fold cross validation
    kf = StratifiedKFold(
        n_splits=EnhancedAudioConfig.K_FOLDS,
        shuffle=True,
        random_state=EnhancedAudioConfig.RANDOM_SEED
    )
    
    fold_results = []
    best_model = None
    best_val_acc = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_encoded)):
        print(f"\n--- Fold {fold+1}/{EnhancedAudioConfig.K_FOLDS} ---")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train_categorical[train_idx], y_train_categorical[val_idx]
        
        # Create data generators
        train_generator = EnhancedDataGenerator(
            X_fold_train, y_fold_train,
            batch_size=EnhancedAudioConfig.BATCH_SIZE,
            augment=True,
            shuffle=True
        )
        
        val_generator = EnhancedDataGenerator(
            X_fold_val, y_fold_val,
            batch_size=EnhancedAudioConfig.BATCH_SIZE,
            augment=False,
            shuffle=False
        )
        
        # Build model
        model = build_enhanced_model(X_train.shape[1:], len(class_names))
        
        # Compile with appropriate loss
        if EnhancedAudioConfig.USE_FOCAL_LOSS:
            loss_fn = focal_loss(
                gamma=EnhancedAudioConfig.FOCAL_GAMMA,
                alpha=EnhancedAudioConfig.FOCAL_ALPHA
                )
        else:
            loss_fn = 'categorical_crossentropy'
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=EnhancedAudioConfig.LEARNING_RATE),
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=EnhancedAudioConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_path, f'fold_{fold+1}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        print(f"Training fold {fold+1}...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EnhancedAudioConfig.EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate fold
        val_loss, val_acc = model.evaluate(val_generator, verbose=0)
        print(f"Fold {fold+1} validation accuracy: {val_acc:.4f}")
        
        # Check performance on difficult instruments
        val_pred = np.argmax(model.predict(X_fold_val, verbose=0), axis=1)
        y_fold_val_labels = np.argmax(y_fold_val, axis=1)
        
        print(f"\nFold {fold+1} - Difficult instruments performance:")
        for instrument in EnhancedAudioConfig.DIFFICULT_INSTRUMENTS:
            if instrument in label_to_int:
                inst_idx = label_to_int[instrument]
                inst_mask = y_fold_val_labels == inst_idx
                if np.any(inst_mask):
                    inst_recall = np.mean(val_pred[inst_mask] == inst_idx)
                    print(f"  {instrument}: {inst_recall:.4f}")
        
        history_serializable = {}
        for key, values in history.history.items():
            history_serializable[key] = [float(v) for v in values]

        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': float(val_acc),
            'val_loss': float(val_loss),
            'history': history_serializable
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            print(f"✓ New best model from fold {fold+1}")
    
    # Save fold results
    with open(os.path.join(model_path, 'cv_results.json'), 'w') as f:
        json.dump(make_json_serializable(fold_results), f, indent=4)
    
    # Final evaluation on test set
    if best_model is not None:
        print("\n" + "="*60)
        print("ENHANCED MODEL - FINAL TEST EVALUATION")
        print("="*60)
        
        test_loss, test_acc = best_model.evaluate(X_test, y_test_categorical, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Detailed predictions
        y_pred_probs = best_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Enhanced Model - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(model_path, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, target_names=class_names))
        
        # Focus on difficult instruments
        print("\n" + "="*60)
        print("DIFFICULT INSTRUMENTS ANALYSIS")
        print("="*60)
        
        for instrument in EnhancedAudioConfig.DIFFICULT_INSTRUMENTS:
            if instrument in label_to_int:
                inst_idx = label_to_int[instrument]
                inst_mask = y_test_encoded == inst_idx
                
                if np.any(inst_mask):
                    # Get predictions for this instrument
                    inst_pred = y_pred[inst_mask]
                    inst_true = y_test_encoded[inst_mask]
                    
                    # Calculate metrics
                    inst_accuracy = np.mean(inst_pred == inst_true)
                    
                    # Confusion analysis
                    print(f"\n{instrument.upper()} Analysis:")
                    print(f"  Accuracy: {inst_accuracy:.4f}")
                    print(f"  Total samples: {np.sum(inst_mask)}")
                    
                    # What it's confused with
                    confused_with = {}
                    for i, pred_idx in enumerate(inst_pred):
                        if pred_idx != inst_idx:
                            pred_name = class_names[pred_idx]
                            confused_with[pred_name] = confused_with.get(pred_name, 0) + 1
                    
                    if confused_with:
                        print(f"  Confused with:")
                        for conf_inst, count in sorted(confused_with.items(), key=lambda x: x[1], reverse=True):
                            print(f"    - {conf_inst}: {count} times ({count/len(inst_pred)*100:.1f}%)")
        
        # Save results
        test_results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.astype(int).tolist(),
            'class_names': class_names,
            'per_class_metrics': {}
        }
        
        with open(os.path.join(model_path, 'test_results.json'), 'w') as f:
            json.dump(make_json_serializable(test_results), f, indent=4)
        
        # Save model
        try:
            best_model.save(os.path.join(model_path, 'enhanced_model.h5'))
            print("\n✅ Enhanced model saved")
        except Exception as e:
            print(f"Model save error: {e}")
        
        # Convert to ONNX
        try:
            input_signature = [tf.TensorSpec(best_model.inputs[0].shape, tf.float32)]
            onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=input_signature)
            onnx.save_model(onnx_model, os.path.join(model_path, 'enhanced_model.onnx'))
            print("✅ ONNX model saved")
        except Exception as e:
            print(f"ONNX conversion error: {e}")
        
        return best_model, test_results
    
    return None, None


def make_json_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def plot_training_history(fold_results, model_path):
    """
    Create comprehensive plots for training history across all folds
    """
    print("\nGenerating training history plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss curves for all folds
    ax1 = plt.subplot(2, 3, 1)
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        history = fold_result['history']
        epochs = range(1, len(history['loss']) + 1)
        
        ax1.plot(epochs, history['loss'], label=f'Fold {fold_num} Train', alpha=0.7)
        ax1.plot(epochs, history['val_loss'], label=f'Fold {fold_num} Val', 
                linestyle='--', alpha=0.7)
    
    ax1.set_title('Loss Across All Folds', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves for all folds
    ax2 = plt.subplot(2, 3, 2)
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        history = fold_result['history']
        epochs = range(1, len(history['accuracy']) + 1)
        
        ax2.plot(epochs, history['accuracy'], label=f'Fold {fold_num} Train', alpha=0.7)
        ax2.plot(epochs, history['val_accuracy'], label=f'Fold {fold_num} Val', 
                linestyle='--', alpha=0.7)
    
    ax2.set_title('Accuracy Across All Folds', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Average performance across folds
    ax3 = plt.subplot(2, 3, 3)
    
    # Calculate average metrics
    max_epochs = max(len(fr['history']['loss']) for fr in fold_results)
    avg_train_loss = np.zeros(max_epochs)
    avg_val_loss = np.zeros(max_epochs)
    avg_train_acc = np.zeros(max_epochs)
    avg_val_acc = np.zeros(max_epochs)
    count = np.zeros(max_epochs)
    
    for fold_result in fold_results:
        history = fold_result['history']
        n_epochs = len(history['loss'])
        avg_train_loss[:n_epochs] += history['loss']
        avg_val_loss[:n_epochs] += history['val_loss']
        avg_train_acc[:n_epochs] += history['accuracy']
        avg_val_acc[:n_epochs] += history['val_accuracy']
        count[:n_epochs] += 1
    
    # Avoid division by zero
    count[count == 0] = 1
    avg_train_loss /= count
    avg_val_loss /= count
    avg_train_acc /= count
    avg_val_acc /= count
    
    epochs = range(1, max_epochs + 1)
    
    # Plot average accuracy
    ax3.plot(epochs, avg_train_acc, 'b-', label='Avg Train Acc', linewidth=2)
    ax3.plot(epochs, avg_val_acc, 'r-', label='Avg Val Acc', linewidth=2)
    ax3.fill_between(epochs, avg_train_acc, avg_val_acc, alpha=0.3)
    
    ax3.set_title('Average Performance Across Folds', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning rate schedule (if available)
    ax4 = plt.subplot(2, 3, 4)
    
    # Try to plot learning rate if it's in the history
    lr_found = False
    for fold_result in fold_results:
        history = fold_result['history']
        if 'lr' in history:
            lr_found = True
            fold_num = fold_result['fold']
            epochs = range(1, len(history['lr']) + 1)
            ax4.plot(epochs, history['lr'], label=f'Fold {fold_num}', alpha=0.7)
    
    if lr_found:
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Learning rate data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    # 5. Final validation accuracies comparison
    ax5 = plt.subplot(2, 3, 5)
    
    fold_nums = [fr['fold'] for fr in fold_results]
    val_accs = [fr['val_accuracy'] for fr in fold_results]
    
    bars = ax5.bar(fold_nums, val_accs, color='skyblue', edgecolor='navy', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, val_accs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Add average line
    avg_val_acc_final = np.mean(val_accs)
    ax5.axhline(y=avg_val_acc_final, color='red', linestyle='--', 
               label=f'Average: {avg_val_acc_final:.3f}')
    
    ax5.set_title('Final Validation Accuracy by Fold', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('Validation Accuracy')
    ax5.set_ylim(0, 1.05)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Training time analysis (if we tracked it)
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate epochs per fold
    epochs_per_fold = [len(fr['history']['loss']) for fr in fold_results]
    
    ax6.bar(fold_nums, epochs_per_fold, color='lightgreen', edgecolor='darkgreen', linewidth=2)
    
    for i, (fold, epochs) in enumerate(zip(fold_nums, epochs_per_fold)):
        ax6.text(fold, epochs, str(epochs), ha='center', va='bottom')
    
    ax6.set_title('Epochs Trained per Fold (Early Stopping)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Fold')
    ax6.set_ylabel('Number of Epochs')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate figure for convergence analysis
    fig2 = plt.figure(figsize=(15, 5))
    
    # Convergence speed analysis
    ax7 = plt.subplot(1, 3, 1)
    
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        history = fold_result['history']
        val_acc = history['val_accuracy']
        
        # Find epoch where accuracy reaches 90% of final value
        final_acc = val_acc[-1]
        target_acc = 0.9 * final_acc
        
        convergence_epoch = next((i for i, acc in enumerate(val_acc) if acc >= target_acc), len(val_acc))
        
        ax7.scatter(fold_num, convergence_epoch + 1, s=100, alpha=0.7)
    
    ax7.set_title('Convergence Speed (90% of Final Accuracy)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Fold')
    ax7.set_ylabel('Epoch')
    ax7.grid(True, alpha=0.3)
    
    # Overfitting analysis
    ax8 = plt.subplot(1, 3, 2)
    
    overfitting_gaps = []
    for fold_result in fold_results:
        history = fold_result['history']
        # Calculate average gap in last 5 epochs
        if len(history['accuracy']) >= 5:
            train_acc_end = np.mean(history['accuracy'][-5:])
            val_acc_end = np.mean(history['val_accuracy'][-5:])
            gap = train_acc_end - val_acc_end
        else:
            gap = history['accuracy'][-1] - history['val_accuracy'][-1]
        overfitting_gaps.append(gap)
    
    bars = ax8.bar(fold_nums, overfitting_gaps, color='salmon', edgecolor='darkred', linewidth=2)
    
    for bar, gap in zip(bars, overfitting_gaps):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.3f}', ha='center', va='bottom' if gap > 0 else 'top')
    
    ax8.set_title('Overfitting Analysis (Train-Val Gap)', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Fold')
    ax8.set_ylabel('Accuracy Gap')
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax8.grid(True, alpha=0.3)
    
    # Best epoch analysis
    ax9 = plt.subplot(1, 3, 3)
    
    best_epochs = []
    for fold_result in fold_results:
        history = fold_result['history']
        best_epoch = np.argmax(history['val_accuracy']) + 1
        best_epochs.append(best_epoch)
    
    ax9.scatter(fold_nums, best_epochs, s=150, c='purple', alpha=0.7, edgecolors='black', linewidth=2)
    ax9.plot(fold_nums, best_epochs, 'purple', alpha=0.3, linewidth=2)
    
    avg_best_epoch = np.mean(best_epochs)
    ax9.axhline(y=avg_best_epoch, color='red', linestyle='--', 
               label=f'Average: {avg_best_epoch:.1f}')
    
    ax9.set_title('Best Epoch per Fold', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Fold')
    ax9.set_ylabel('Best Epoch')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Training history plots saved")
    
    # Generate summary statistics
    summary_stats = {
        'average_final_val_accuracy': float(np.mean(val_accs)),
        'std_val_accuracy': float(np.std(val_accs)),
        'best_fold': int(fold_nums[np.argmax(val_accs)]),
        'best_accuracy': float(max(val_accs)),
        'average_epochs': float(np.mean(epochs_per_fold)),
        'average_overfitting_gap': float(np.mean(overfitting_gaps)),
        'average_best_epoch': float(avg_best_epoch)
    }
    
    with open(os.path.join(model_path, 'training_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=4)

        training_summary = plot_training_history(fold_results, model_path)
        print(f"\nTraining Summary:")
        print(f"  Average Validation Accuracy: {training_summary['average_final_val_accuracy']:.4f} ± {training_summary['std_val_accuracy']:.4f}")
        print(f"  Best Fold: {training_summary['best_fold']} ({training_summary['best_accuracy']:.4f})")
        print(f"  Average Overfitting Gap: {training_summary['average_overfitting_gap']:.4f}")
    
    return summary_stats


def create_enhanced_inference_script():
    """
    Create inference script for the enhanced model
    """
    inference_script = f'''
import numpy as np
import librosa
import onnxruntime as ort
import json

class EnhancedAudioClassifier:
    """
    Enhanced classifier with multi-channel features and HPSS
    """
    
    def __init__(self, model_path='enhanced_model.onnx', label_mapping_path='label_mapping.json'):
        self.session = ort.InferenceSession(model_path)
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        self.idx_to_label = {{int(idx): label for label, idx in self.label_mapping.items()}}
        
        # Audio parameters
        self.SAMPLE_RATE = {EnhancedAudioConfig.SAMPLE_RATE}
        self.SEGMENT_DURATION = {EnhancedAudioConfig.SEGMENT_DURATION}
        self.N_FFT = {EnhancedAudioConfig.N_FFT}
        self.HOP_LENGTH = {EnhancedAudioConfig.HOP_LENGTH}
        self.N_MELS = {EnhancedAudioConfig.N_MELS}
        self.N_MFCC = {EnhancedAudioConfig.N_MFCC}
        self.FMAX = {EnhancedAudioConfig.FMAX}
        self.USE_MULTI_CHANNEL = {EnhancedAudioConfig.USE_MULTI_CHANNEL}
    
    def extract_features(self, audio, sr):
        """Extract enhanced features including HPSS"""
        features = {{}}
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            fmax=self.FMAX
        )
        features['mel'] = librosa.power_to_db(mel_spec, ref=np.max)
        
        if self.USE_MULTI_CHANNEL:
            # HPSS
            harmonic, percussive = librosa.effects.hpss(audio, margin=(1.0, 5.0))
            
            harmonic_mel = librosa.feature.melspectrogram(
                y=harmonic, sr=sr,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                n_mels=self.N_MELS,
                fmax=self.FMAX
            )
            features['harmonic'] = librosa.power_to_db(harmonic_mel, ref=np.max)
            
            percussive_mel = librosa.feature.melspectrogram(
                y=percussive, sr=sr,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                n_mels=self.N_MELS,
                fmax=self.FMAX
            )
            features['percussive'] = librosa.power_to_db(percussive_mel, ref=np.max)
            
            # MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr,
                n_mfcc=self.N_MFCC,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH
            )
            
            if mfcc.shape[0] < self.N_MELS:
                pad_width = self.N_MELS - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            
            features['mfcc'] = mfcc
            features['delta'] = librosa.feature.delta(mfcc)
            features['delta2'] = librosa.feature.delta(mfcc, order=2)
        
        return features
    
    def create_multi_channel_input(self, features):
        """Create multi-channel input"""
        if self.USE_MULTI_CHANNEL:
            channels = ['mel', 'harmonic', 'percussive', 'mfcc', 'delta', 'delta2']
            channel_data = []
            
            for channel in channels:
                if channel in features:
                    data = features[channel]
                    data_norm = (data - data.mean()) / (data.std() + 1e-8)
                    channel_data.append(data_norm)
            
            return np.stack(channel_data, axis=-1)
        else:
            data = features['mel']
            data_norm = (data - data.mean()) / (data.std() + 1e-8)
            return np.expand_dims(data_norm, axis=-1)
    
    def predict(self, audio_path, confidence_threshold=0.5):
        """Predict instrument with enhanced features"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        
        # Extract segment
        segment_len = int(self.SEGMENT_DURATION * sr)
        if len(audio) > segment_len:
            # Find best segment based on energy
            best_start = 0
            best_energy = 0
            
            for start in range(0, len(audio) - segment_len, segment_len // 4):
                segment = audio[start:start + segment_len]
                energy = np.sqrt(np.mean(segment**2))
                if energy > best_energy:
                    best_energy = energy
                    best_start = start
            
            audio = audio[best_start:best_start + segment_len]
        else:
            audio = np.pad(audio, (0, segment_len - len(audio)), mode='constant')
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Create input
        multi_channel = self.create_multi_channel_input(features)
        input_data = np.expand_dims(multi_channel, axis=0).astype(np.float32)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {{input_name: input_data}})
        probabilities = outputs[0][0]
        
        # Get prediction
        max_idx = np.argmax(probabilities)
        max_prob = probabilities[max_idx]
        instrument = self.idx_to_label[max_idx]
        
        return {{
            'instrument': instrument,
            'confidence': float(max_prob),
            'probabilities': {{self.idx_to_label[i]: float(p) for i, p in enumerate(probabilities)}},
            'is_uncertain': max_prob < confidence_threshold
        }}

# Example usage
if __name__ == "__main__":
    classifier = EnhancedAudioClassifier('enhanced_model.onnx', 'label_mapping.json')
    result = classifier.predict('test_audio.wav')
    print(f"Predicted: {{result['instrument']}} ({{result['confidence']:.3f}})")
'''
    
    with open(os.path.join(model_path, 'enhanced_inference.py'), 'w') as f:
        f.write(inference_script)
    
    print("✅ Enhanced inference script saved")

def save_enhanced_config():
    """
    Save configuration for reproducibility
    """
    config_dict = {}
    for key in dir(EnhancedAudioConfig):
        if not key.startswith('__'):
            value = getattr(EnhancedAudioConfig, key)
            if not callable(value):
                if isinstance(value, (list, tuple)):
                    config_dict[key] = list(value)
                else:
                    config_dict[key] = value
    
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print("✅ Configuration saved")

def main():
    """
    Main function to run the enhanced training pipeline
    """
    print("="*80)
    print("ENHANCED LAO INSTRUMENT CLASSIFIER")
    print("="*80)
    print("🎯 Mission: Improve khean/pin/saw classification with advanced features")
    print("🔧 Key enhancements:")
    print("   • Harmonic-Percussive Source Separation (HPSS)")
    print("   • Multi-channel features (Mel + Harmonic + Percussive + MFCC)")
    print("   • SpecAugment for robustness")
    print("   • Mixup augmentation")
    print("   • Focal loss for hard examples")
    print("   • Attention mechanism")
    print("   • Enhanced weights for difficult instruments")
    print("="*80)
    
    # Check GPU availability
    if len(physical_devices) > 0:
        print(f"✅ Using GPU: {physical_devices[0].name}")
    else:
        print("⚠️ No GPU found, using CPU (training will be slower)")
    
    # Save configuration
    save_enhanced_config()
    
    # Process dataset
    print("\n1. PROCESSING DATASET WITH ENHANCED FEATURES...")
    start_time = datetime.now()
    
    try:
        X_train, X_test, y_train, y_test, class_names = process_enhanced_dataset()
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return
    
    if len(X_train) == 0:
        print("❌ No training data found!")
        return
    
    processing_time = datetime.now() - start_time
    print(f"✓ Dataset processed in {processing_time}")
    
    # Display class distribution
    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {cls}: {count} samples")
    
    # Train model
    print("\n2. TRAINING ENHANCED MODEL...")
    train_start = datetime.now()
    
    try:
        best_model, results = train_enhanced_model(X_train, y_train, X_test, y_test, class_names)
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    training_time = datetime.now() - train_start
    total_time = datetime.now() - start_time
    
    print(f"\n✓ Training completed in {training_time}")
    print(f"✓ Total time: {total_time}")
    
    if results:
        print(f"\n📊 ENHANCED MODEL RESULTS:")
        print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"   Model saved to: {model_path}")
        
        # Create inference script
        create_enhanced_inference_script()
        
        print("\n🎯 Focus on difficult instruments:")
        for instrument in EnhancedAudioConfig.DIFFICULT_INSTRUMENTS:
            print(f"   • Check {instrument} performance in classification report above")
        
        print("\n💡 Next steps:")
        print("   • Test with problematic khean/pin/saw recordings")
        print("   • Use enhanced_inference.py for deployment")
        print("   • The multi-channel features should better separate similar instruments")
        print("   • HPSS helps distinguish harmonic (khean/saw) from percussive sounds")
        print("   • SpecAugment improves robustness to background music")
    else:
        print("\n❌ Training failed - check error messages above")

if __name__ == "__main__":
    main()