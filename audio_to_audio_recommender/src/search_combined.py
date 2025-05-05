import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import List, Tuple, Dict
import logging
from src.type_classifier import AudioTypeClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedSimilaritySearch:
    def __init__(self, 
                 features_path: str = "data/embeddings_combined_v2/combined_features.joblib",
                 type_classifier_path: str = "models/type_classifier.joblib",
                 cnn_weight: float = 0.6,
                 type_confidence_threshold: float = 0.6):
        """
        Initialize the combined similarity search system.
        
        Args:
            features_path: Path to combined features file
            type_classifier_path: Path to trained type classifier
            cnn_weight: Weight for CNN features (0-1)
            type_confidence_threshold: Minimum confidence for type filtering
        """
        self.features_dict = joblib.load(features_path)
        self.filenames = list(self.features_dict.keys())
        self.cnn_weight = cnn_weight
        self.trad_weight = 1 - cnn_weight
        self.type_confidence_threshold = type_confidence_threshold
        
        # Load type classifier
        self.type_classifier = AudioTypeClassifier.load(type_classifier_path)
        
        # Initialize scalers
        self.cnn_scaler = StandardScaler()
        self.trad_scaler = StandardScaler()
        
        # Prepare feature vectors
        self._prepare_feature_vectors()
        
        # Pre-compute types for all samples
        self._compute_sample_types()
        
        logger.info(f"Loaded {len(self.filenames)} samples with combined features")
        logger.info(f"Using weights: CNN={self.cnn_weight:.2f}, Traditional={self.trad_weight:.2f}")

    def _flatten_traditional_features(self, features: Dict) -> np.ndarray:
        """Flatten traditional features into a 1D array."""
        flattened = []
        
        # Handle scalar features
        scalar_features = ['spectral_centroid', 'spectral_bandwidth', 
                         'spectral_rolloff', 'tempo', 'zero_crossing_rate']
        for feature_name in scalar_features:
            if feature_name in features:
                flattened.append(float(features[feature_name]))
        
        # Handle array features
        array_features = ['mfcc_mean', 'mfcc_var']
        for feature_name in array_features:
            if feature_name in features:
                flattened.extend(features[feature_name].flatten())
        
        return np.array(flattened)

    def _prepare_feature_vectors(self):
        """Prepare and normalize both CNN and traditional feature vectors."""
        cnn_features = []
        trad_features = []
        
        for filename in self.filenames:
            # Get CNN embeddings
            cnn_embedding = self.features_dict[filename]['cnn_embedding']
            cnn_features.append(cnn_embedding)
            
            # Get traditional features
            trad_feature = self._flatten_traditional_features(
                self.features_dict[filename]['traditional_features']
            )
            trad_features.append(trad_feature)
        
        # Convert to arrays and normalize
        self.cnn_vectors = self.cnn_scaler.fit_transform(np.array(cnn_features))
        self.trad_vectors = self.trad_scaler.fit_transform(np.array(trad_features))
        
        logger.info(f"CNN feature shape: {self.cnn_vectors.shape}")
        logger.info(f"Traditional feature shape: {self.trad_vectors.shape}")

    def _compute_sample_types(self):
        """Pre-compute types for all samples in the database."""
        self.sample_types = {}
        for filename, features in self.features_dict.items():
            pred_type, _ = self.type_classifier.predict_type(features)
            self.sample_types[filename] = pred_type

    def compute_combined_similarity(self, 
                                 query_cnn: np.ndarray, 
                                 query_trad: np.ndarray) -> np.ndarray:
        """Compute weighted combination of CNN and traditional similarities."""
        # Compute CNN similarity
        cnn_similarities = cosine_similarity([query_cnn], self.cnn_vectors)[0]
        
        # Compute traditional feature similarity
        trad_similarities = cosine_similarity([query_trad], self.trad_vectors)[0]
        
        # Weighted combination
        combined_similarities = (self.cnn_weight * cnn_similarities + 
                               self.trad_weight * trad_similarities)
        
        return combined_similarities

    def find_similar(self, 
                    query_file: str, 
                    n_results: int = 5,
                    enforce_type_matching: bool = True) -> List[Tuple[str, float, str]]:
        """
        Find similar sounds to a query file.
        
        Args:
            query_file: Path to query audio file
            n_results: Number of results to return
            enforce_type_matching: Whether to enforce type matching
        
        Returns:
            List of (filename, similarity, type) tuples
        """
        if query_file not in self.features_dict:
            raise ValueError(f"Query file {query_file} not found in database")
        
        # Get query features
        query_data = self.features_dict[query_file]
        query_cnn = self.cnn_scaler.transform([query_data['cnn_embedding']])[0]
        query_trad = self.trad_scaler.transform(
            [self._flatten_traditional_features(query_data['traditional_features'])]
        )[0]
        
        # Predict query type
        query_type, type_confidence = self.type_classifier.predict_type(query_data)
        logger.info(f"Query predicted as type '{query_type}' with confidence {type_confidence:.2f}")
        
        # Compute similarities
        similarities = self.compute_combined_similarity(query_cnn, query_trad)
        
        # Filter by type if confidence is high enough
        if enforce_type_matching and type_confidence >= self.type_confidence_threshold:
            type_mask = np.array([
                self.sample_types[fname] == query_type 
                for fname in self.filenames
            ])
            similarities = similarities * type_mask
        
        # Get top N similar sounds (excluding the query itself)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Remove query file and zero similarity matches
        similar_indices = [
            idx for idx in similar_indices 
            if self.filenames[idx] != query_file and similarities[idx] > 0
        ][:n_results]
        
        return [
            (self.filenames[idx], 
             similarities[idx],
             self.sample_types[self.filenames[idx]]) 
            for idx in similar_indices
        ]

    def find_similar_to_features(self, 
                               query_features: Dict, 
                               n_results: int = 5,
                               enforce_type_matching: bool = True) -> List[Tuple[str, float, str]]:
        """
        Find similar sounds to a set of features.
        
        Args:
            query_features: Dictionary containing extracted features
            n_results: Number of results to return
            enforce_type_matching: Whether to enforce type matching
            
        Returns:
            List of (filename, similarity, audio_type) tuples
        """
        # Normalize query features
        query_cnn = self.cnn_scaler.transform([query_features['cnn_embedding']])[0]
        query_trad = self.trad_scaler.transform(
            [self._flatten_traditional_features(query_features['traditional_features'])]
        )[0]
        
        # Predict query type
        query_type, type_confidence = self.type_classifier.predict_type(query_features)
        logger.info(f"Query predicted as type '{query_type}' with confidence {type_confidence:.2f}")
        
        # Compute similarities
        similarities = self.compute_combined_similarity(query_cnn, query_trad)
        
        # Filter by type if confidence is high enough
        if enforce_type_matching and type_confidence >= self.type_confidence_threshold:
            type_mask = np.array([
                self.sample_types[fname] == query_type 
                for fname in self.filenames
            ])
            similarities = similarities * type_mask
        
        # Get top N similar sounds
        similar_indices = np.argsort(similarities)[::-1]
        
        # Remove zero similarity matches
        similar_indices = [
            idx for idx in similar_indices 
            if similarities[idx] > 0
        ][:n_results]
        
        return [
            (self.filenames[idx], 
             similarities[idx],
             self.sample_types[self.filenames[idx]]) 
            for idx in similar_indices
        ]

    def adjust_weights(self, cnn_weight: float):
        """Adjust the weights between CNN and traditional features."""
        self.cnn_weight = max(0.0, min(1.0, cnn_weight))
        self.trad_weight = 1 - self.cnn_weight
        logger.info(f"Updated weights: CNN={self.cnn_weight:.2f}, "
                   f"Traditional={self.trad_weight:.2f}")

def main():
    # Example usage
    searcher = CombinedSimilaritySearch()
    
    # Try a sample search
    try:
        sample_file = searcher.filenames[0]
        print(f"\nFinding similar sounds to: {sample_file}")
        
        similar_sounds = searcher.find_similar(sample_file, n_results=5)
        
        print("\nTop 5 similar sounds:")
        for filename, similarity, audio_type in similar_sounds:
            print(f"{filename} ({audio_type}): {similarity:.3f}")
            
    except Exception as e:
        logger.error(f"Error during sample search: {str(e)}")

if __name__ == "__main__":
    main()
