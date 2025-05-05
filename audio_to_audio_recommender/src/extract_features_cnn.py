import os
import numpy as np
import tensorflow as tf
import librosa
import joblib
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_encoder import AudioEncoder
from src.extract_features import AudioFeatureExtractor

class CombinedFeatureExtractor:
    def __init__(self):
        # Load CNN encoder with new weights
        self.cnn_model = AudioEncoder(latent_dim=128)
        self.cnn_model.encoder.load_weights("/Users/vedantdangayach/audio_recommender/audio_to_audio_recommender/models/cnn_encoder.weights.h5")
        
        # Initialize traditional feature extractor
        self.traditional_extractor = AudioFeatureExtractor()
        
        # Initialize mel spectrogram parameters
        self.sr = 22050
        self.n_mels = 128
        self.hop_length = 512

    def audio_to_melspec(self, audio_path):
        """Convert audio to mel spectrogram for CNN."""
        try:
            y, _ = librosa.load(audio_path, sr=self.sr)
            
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
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def extract_combined_features(self, audio_path):
        """Extract both CNN and traditional features."""
        try:
            # Get CNN embeddings
            mel_spec = self.audio_to_melspec(audio_path)
            if mel_spec is None:
                return None
            cnn_embedding = self.cnn_model.encoder.predict(mel_spec[np.newaxis, ...], verbose=0)[0]
            
            # Get traditional features
            trad_features = self.traditional_extractor.process_audio_file(audio_path)
            if trad_features is None:
                return None
            
            # Combine features
            return {
                'cnn_embedding': cnn_embedding,
                'traditional_features': trad_features['features'],
                'length': trad_features['length']
            }
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None

    def batch_process_directory(self, input_dir, output_dir):
        """Process all audio files in directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = [f for f in os.listdir(input_dir) 
                      if f.endswith(('.wav', '.mp3'))]
        
        features_dict = {}
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            audio_path = os.path.join(input_dir, audio_file)
            features = self.extract_combined_features(audio_path)
            if features is not None:
                features_dict[audio_file] = features
        
        # Save combined features
        output_path = os.path.join(output_dir, 'combined_features.joblib')
        joblib.dump(features_dict, output_path)
        print(f"Features saved to {output_path}")

def main():
    extractor = CombinedFeatureExtractor()
    
    # Process audio files
    input_dir = "data/raw_samples"
    output_dir = "data/embeddings_combined_v2"  # New output directory for v2
    
    print("Extracting combined features...")
    extractor.batch_process_directory(input_dir, output_dir)
    print("Feature extraction complete!")

if __name__ == "__main__":
    main()
