# src/train_cnn_encoder.py

import os
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_encoder import AudioEncoder
import joblib

class SpectrogramPreprocessor:
    def __init__(self, sr=22050, n_mels=128, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
    
    def audio_to_melspec(self, audio_path):
        """Convert audio file to mel spectrogram."""
        try:
            if isinstance(audio_path, str):
                # Load from file
                y, _ = librosa.load(audio_path, sr=self.sr)
            else:
                # Use provided numpy array
                y = audio_path
            
            # Pad or trim to 3 seconds
            target_length = 3 * self.sr
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
            
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=self.sr,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
            mel_spec = librosa.util.fix_length(mel_spec, size=128, axis=1)
            
            return mel_spec.reshape(128, 128, 1)
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return None

def prepare_dataset(raw_dir, aug_dir, preprocessor):
    """Prepare dataset including augmented samples."""
    spectrograms = []
    filenames = []
    
    # Process original files
    print("Processing original files...")
    audio_files = [f for f in os.listdir(raw_dir) if f.endswith(('.wav', '.mp3'))]
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(raw_dir, audio_file)
        mel_spec = preprocessor.audio_to_melspec(audio_path)
        if mel_spec is not None:
            spectrograms.append(mel_spec)
            filenames.append(audio_file)
    
    # Process augmented files if they exist
    if os.path.exists(aug_dir):
        print("Processing augmented files...")
        aug_files = [f for f in os.listdir(aug_dir) if f.endswith(('.wav', '.mp3'))]
        for audio_file in tqdm(aug_files):
            audio_path = os.path.join(aug_dir, audio_file)
            mel_spec = preprocessor.audio_to_melspec(audio_path)
            if mel_spec is not None:
                spectrograms.append(mel_spec)
                filenames.append(f"aug_{audio_file}")
    
    return np.array(spectrograms), filenames

def main():
    # Initialize preprocessor and model
    preprocessor = SpectrogramPreprocessor()
    model = AudioEncoder(latent_dim=128)
    
    # Prepare dataset
    print("Preparing dataset...")
    raw_dir = "data/raw_samples"
    aug_dir = "data/augmented_samples"
    spectrograms, filenames = prepare_dataset(raw_dir, aug_dir, preprocessor)
    
    print(f"Total dataset size: {len(spectrograms)} samples")
    print(f"Original samples: {len([f for f in filenames if not f.startswith('aug_')])}")
    print(f"Augmented samples: {len([f for f in filenames if f.startswith('aug_')])}")
    
    # Split into train and validation
    indices = np.random.permutation(len(spectrograms))
    split_idx = int(0.85 * len(spectrograms))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    x_train = spectrograms[train_indices]
    x_val = spectrograms[val_indices]
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Train
    print("Training model...")
    history = model.fit(
        x_train, x_train,  # Autoencoder reconstructs input
        epochs=100,  # Increased epochs for larger dataset
        batch_size=32,
        validation_data=(x_val, x_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/cnn_encoder_checkpoint_v2.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )
    
    # Save encoder weights with new version
    os.makedirs("models", exist_ok=True)
    model.encoder.save_weights("models/cnn_encoder.weights.h5")
    
    # Save training history
    joblib.dump(history.history, "models/training_history_v2.joblib")
    
    print("Training complete! Saved encoder weights v2.")

if __name__ == "__main__":
    main()