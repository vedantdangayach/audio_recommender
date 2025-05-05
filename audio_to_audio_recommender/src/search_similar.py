import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import List, Tuple, Dict
import os
import numpy.typing as npt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoundSimilaritySearch:
    def __init__(self, features_path: str):
        """Initialize the similarity search system."""
        logger.info(f"Loading features from {features_path}")
        self.features_dict = joblib.load(features_path)
        self.feature_vectors = None
        self.filenames = list(self.features_dict.keys())
        self.scaler = StandardScaler()
        logger.info(f"Loaded {len(self.filenames)} audio files")
        self._prepare_feature_vectors()

    def _safely_flatten_array(self, arr: np.ndarray, expected_size: int) -> np.ndarray:
        """Safely flatten and resize an array to a specified size."""
        try:
            # Convert to numpy array if not already
            arr = np.array(arr, dtype=np.float32)
            
            # Flatten the array
            flat_arr = arr.ravel()
            
            # Resize to expected size
            if len(flat_arr) > expected_size:
                return flat_arr[:expected_size]
            elif len(flat_arr) < expected_size:
                return np.pad(flat_arr, (0, expected_size - len(flat_arr)), 
                            mode='constant', constant_values=0)
            return flat_arr
        except Exception as e:
            logger.error(f"Error flattening array: {e}")
            # Return zero array if processing fails
            return np.zeros(expected_size, dtype=np.float32)

    def _flatten_features(self, features: Dict[str, npt.ArrayLike]) -> np.ndarray:
        """Flatten feature dictionary into a 1D array with consistent shape."""
        flattened = []
        
        # Handle scalar features
        scalar_features = ['spectral_centroid', 'spectral_bandwidth', 
                         'spectral_rolloff', 'tempo', 'zero_crossing_rate']
        for feature_name in scalar_features:
            if feature_name in features:
                try:
                    value = float(features[feature_name])
                    flattened.append(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing scalar feature {feature_name}: {e}")
                    flattened.append(0.0)
        
        # Handle array features with consistent shapes
        array_features = {
            'mfcc_mean': 20,
            'mfcc_var': 20,
            'mel_mean': 80,
            'mel_std': 80,
            'temporal_envelope': 100,
            'band_energy': 80
        }
        
        for feature_name, expected_size in array_features.items():
            if feature_name in features:
                try:
                    feature_array = self._safely_flatten_array(features[feature_name], expected_size)
                    flattened.extend(feature_array.tolist())
                except Exception as e:
                    logger.error(f"Error processing array feature {feature_name}: {e}")
                    flattened.extend([0.0] * expected_size)
        
        # Convert to numpy array
        flattened_array = np.array(flattened, dtype=np.float32)
        logger.debug(f"Flattened feature vector shape: {flattened_array.shape}")
        return flattened_array

    def _prepare_feature_vectors(self):
        """Convert the feature dictionary into a matrix for efficient similarity computation."""
        all_features = []
        
        for filename in self.filenames:
            try:
                features = self.features_dict[filename]['features']
                flattened_features = self._flatten_features(features)
                all_features.append(flattened_features)
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                # Create a zero vector with the expected size
                expected_size = 385  # 5 scalar + 20*2 MFCC + 80*3 mel + 100 temporal
                all_features.append(np.zeros(expected_size, dtype=np.float32))
        
        # Convert to numpy array and normalize
        try:
            feature_matrix = np.array(all_features, dtype=np.float32)
            logger.info(f"Feature matrix shape: {feature_matrix.shape}")
            self.feature_vectors = self.scaler.fit_transform(feature_matrix)
        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            raise

    def find_similar(self, 
                    query_file: str, 
                    n_results: int = 5, 
                    metric: str = 'cosine') -> List[Tuple[str, float]]:
        """Find similar sounds to a query file."""
        if query_file not in self.features_dict:
            raise ValueError(f"Query file {query_file} not found in database")
        
        query_idx = self.filenames.index(query_file)
        query_vector = self.feature_vectors[query_idx]
        
        if metric == 'cosine':
            similarities = cosine_similarity([query_vector], self.feature_vectors)[0]
        else:
            similarities = -np.linalg.norm(self.feature_vectors - query_vector, axis=1)
        
        similar_indices = np.argsort(similarities)[::-1][1:n_results+1]
        return [(self.filenames[idx], similarities[idx]) for idx in similar_indices]

    def find_similar_to_features(self, 
                               query_features: Dict, 
                               n_results: int = 5, 
                               metric: str = 'cosine') -> List[Tuple[str, float]]:
        """Find similar sounds to a set of features."""
        query_vector = self._flatten_features(query_features)
        query_vector = self.scaler.transform([query_vector])[0]
        
        if metric == 'cosine':
            similarities = cosine_similarity([query_vector], self.feature_vectors)[0]
        else:
            similarities = -np.linalg.norm(self.feature_vectors - query_vector, axis=1)
        
        similar_indices = np.argsort(similarities)[::-1][:n_results]
        return [(self.filenames[idx], similarities[idx]) for idx in similar_indices]

if __name__ == "__main__":
    # Example usage
    features_path = "../data/embeddings/audio_features.joblib"
    searcher = SoundSimilaritySearch(features_path)
    
    # Example: Find similar sounds to the first file in the database
    query_file = searcher.filenames[0]
    similar_sounds = searcher.find_similar(query_file, n_results=5)
    
    print(f"\nSimilar sounds to {query_file}:")
    for filename, score in similar_sounds:
        print(f"{filename}: similarity = {score:.3f}")
