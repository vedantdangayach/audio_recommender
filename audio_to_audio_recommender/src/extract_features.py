import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from typing import Dict, List, Union, Tuple
import warnings
import pandas as pd

def load_features_with_version(features_path):
    """Load features with version checking."""
    try:
        return joblib.load(features_path)
    except Exception as e:
        print(f"Error loading features: {e}")
        print("This might be due to version mismatch in the saved file.")
        print("Consider re-running feature extraction with current package versions.")
        raise

class AudioFeatureExtractor:
    def __init__(self, 
                 sr: int = 22050, 
                 n_mfcc: int = 20,
                 hop_length: int = 512,
                 window_size: int = 2048,
                 mel_bins: int = 80):
        """Initialize the feature extractor with spectrogram parameters."""
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.window_size = window_size
        self.mel_bins = mel_bins

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Load audio file and return signal and length."""
        try:
            y, _ = librosa.load(audio_path, sr=self.sr)
            duration = librosa.get_duration(y=y, sr=self.sr)
            return y, duration
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            return None, None

    def compute_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram using TensorFlow."""
        # Convert to tensorflow format
        waveform = tf.convert_to_tensor(y, dtype=tf.float32)
        
        # Compute STFT
        stft = tf.signal.stft(
            waveform,
            frame_length=self.window_size,
            frame_step=self.hop_length,
            pad_end=True
        )
        
        # Convert to power spectrum
        spectrogram = tf.abs(stft)
        power_spectrogram = tf.math.square(spectrogram)
        
        # Create mel filterbank matrix
        num_spectrogram_bins = stft.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self.sr,
            lower_edge_hertz=20,
            upper_edge_hertz=self.sr/2
        )
        
        # Convert to mel spectrograms
        mel_spectrograms = tf.tensordot(
            power_spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        
        return mel_spectrograms.numpy()

    def extract_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features from an audio signal."""
        features = {}
        
        # TensorFlow-based mel spectrogram features
        mel_spec = self.compute_mel_spectrogram(y)
        features['mel_mean'] = np.mean(mel_spec, axis=0)
        features['mel_std'] = np.std(mel_spec, axis=0)
        
        # Compute temporal features from mel spectrogram
        features['temporal_envelope'] = np.mean(mel_spec, axis=1)
        
        # Extract frequency band energies
        band_energies = np.sum(mel_spec, axis=0)
        features['band_energy'] = band_energies
        
        # Traditional features
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        features['mfcc_mean'] = np.mean(mfccs.T, axis=0)
        features['mfcc_var'] = np.var(mfccs.T, axis=0)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        features['spectral_centroid'] = np.mean(spectral_centroids)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
        features['tempo'] = tempo
        
        # Zero crossing rate for noisiness
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate'] = np.mean(zcr)
        
        return features

    def process_audio_file(self, audio_path: str) -> Dict[str, Union[Dict[str, np.ndarray], float]]:
        """Process a single audio file and return its features and length."""
        y, duration = self.load_audio(audio_path)
        if y is None:
            return None
        
        features = self.extract_features(y)
        return {
            'features': features,
            'length': duration
        }

    def batch_process_directory(self, 
                              input_dir: str, 
                              output_dir: str,
                              file_extension: str = '.wav') -> None:
        """Process all audio files in a directory and save their features."""
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = [f for f in os.listdir(input_dir) 
                      if f.endswith(file_extension)]
        
        features_dict = {}
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            audio_path = os.path.join(input_dir, audio_file)
            result = self.process_audio_file(audio_path)
            if result is not None:
                features_dict[audio_file] = result
        
        output_path = os.path.join(output_dir, 'audio_features.joblib')
        joblib.dump(features_dict, output_path)
        print(f"Features saved to {output_path}")
    def save_features_with_version(self, features_dict, output_path):
        """Save features with version information."""
        metadata = {
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'features': features_dict
        }
        joblib.dump(metadata, output_path)

    def load_features_with_version(self, features_path):
        """Load features with version checking."""
        try:
            metadata = joblib.load(features_path)
            if isinstance(metadata, dict) and 'numpy_version' in metadata:
                print(f"Features were created with:")
                print(f"- NumPy version: {metadata['numpy_version']}")
                print(f"- Pandas version: {metadata['pandas_version']}")
                print(f"Current versions:")
                print(f"- NumPy version: {np.__version__}")
                print(f"- Pandas version: {pd.__version__}")
                return metadata['features']
            return metadata  # Old format without version info
        except Exception as e:
            print(f"Error loading features: {e}")
            print("This might be due to version mismatch in the saved file.")
            print("Consider re-running feature extraction with current package versions.")
            raise

if __name__ == "__main__":
    # Example usage
    extractor = AudioFeatureExtractor()
    
    # Set your input and output directories
    input_dir = "/Users/vedantdangayach/audio_recommender/data/raw_samples"
    output_dir = "/Users/vedantdangayach/audio_recommender/data/embeddings"
    
    # Process all files
    extractor.batch_process_directory(input_dir, output_dir)

