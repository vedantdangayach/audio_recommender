import torch
from laion_clap import CLAP_Module
import numpy as np
from typing import List, Tuple
import joblib
from tqdm import tqdm
import os
from torch.serialization import add_safe_globals

# Add the required numpy globals to the safe list
add_safe_globals(['numpy.core.multiarray.scalar'])

class TextAudioMatcher:
    def __init__(self, features_path: str = "data/embeddings/audio_features.joblib"):
        """Initialize the CLAP-based text-to-audio matcher."""
        print("Loading CLAP model...")
        try:
            # Initialize CLAP model with debug info
            print("Initializing CLAP model...")
            self.model = CLAP_Module(enable_fusion=False)
            
            # Load the model with modified settings
            print("Loading CLAP weights...")
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **kwargs)
            try:
                self.model.load_ckpt()  # Try default loading first
            except Exception as e1:
                print(f"Default loading failed: {e1}")
                print("Trying alternative loading method...")
                # If default fails, try direct loading
                self.model = CLAP_Module(
                    enable_fusion=False,
                    amodel='HTSAT-base',  # Specify the audio model
                    tmodel='roberta'  # Specify the text model
                )
            finally:
                torch.load = original_load
            
            print("CLAP model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading CLAP model: {str(e)}")
            raise

        print("Loading feature dictionary...")
        self.features_dict = joblib.load(features_path)
        if isinstance(self.features_dict, dict) and 'features' in self.features_dict:
            self.features_dict = self.features_dict['features']
        
        self.filenames = [f for f in list(self.features_dict.keys()) 
                         if isinstance(self.features_dict[f], dict)]
        
        print(f"Found {len(self.filenames)} audio files")
        print("Computing audio embeddings...")
        self.audio_embeddings = self._compute_audio_embeddings()

    def _compute_audio_embeddings(self) -> np.ndarray:
        """Pre-compute audio embeddings for all files in the dataset."""
        embeddings = []
        
        for filename in tqdm(self.filenames):
            audio_path = os.path.join("data/raw_samples", filename)
            try:
                # Add debug info
                print(f"\nProcessing: {filename}")
                audio_data = self.model.get_audio_embedding_from_filePath(audio_path)
                print(f"Embedding shape: {audio_data.shape}")
                print(f"Embedding range: {audio_data.min():.3f} to {audio_data.max():.3f}")
                embeddings.append(audio_data)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                embeddings.append(np.zeros(512))
        
        embeddings_array = np.vstack(embeddings)
        print(f"\nFinal embeddings shape: {embeddings_array.shape}")
        return embeddings_array

    def search_by_text(self, text_query: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """Search for audio files matching the text description."""
        print(f"\nProcessing query: '{text_query}'")
        
        # Get text embedding with debug info
        print("Computing text embedding...")
        text_embedding = self.model.get_text_embedding([text_query])
        print(f"Text embedding shape: {text_embedding.shape}")
        print(f"Text embedding sample (first 5 values): {text_embedding[0][:5]}")
        
        # Calculate similarities with debug info
        print("Calculating similarities...")
        similarities = self._calculate_similarities(text_embedding, self.audio_embeddings)
        print(f"Similarity scores range: {similarities.min():.3f} to {similarities.max():.3f}")
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        # Return results with debug info
        results = []
        print("\nTop matches:")
        for idx in top_indices:
            filename = self.filenames[idx]
            similarity = similarities[idx]
            results.append((filename, similarity))
            print(f"- {filename}: {similarity:.3f}")
        
        return results

    def _calculate_similarities(self, text_embedding: np.ndarray, audio_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between text and audio embeddings."""
        # Normalize embeddings
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
        audio_embeddings = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(text_embedding, audio_embeddings.T)[0]
        
        return similarities