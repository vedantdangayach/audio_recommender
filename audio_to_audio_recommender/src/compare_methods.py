import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import List, Tuple, Dict
import logging
from tabulate import tabulate
import random

class MethodComparison:
    def __init__(self):
        # Load combined features
        self.combined_features = joblib.load("data/embeddings_combined/combined_features.joblib")
        # Load original features
        self.traditional_features = joblib.load("data/embeddings/audio_features.joblib")
        self.filenames = list(self.combined_features.keys())
        
        # Initialize scalers
        self.cnn_scaler = StandardScaler()
        self.trad_scaler = StandardScaler()
        
        # Prepare feature vectors
        self._prepare_features()
        
    def _prepare_features(self):
        """Prepare both CNN and traditional feature vectors."""
        # Prepare CNN features
        cnn_features = []
        for filename in self.filenames:
            cnn_features.append(self.combined_features[filename]['cnn_embedding'])
        self.cnn_vectors = self.cnn_scaler.fit_transform(np.array(cnn_features))
        
        # Prepare traditional features
        trad_features = []
        for filename in self.filenames:
            features = self.traditional_features[filename]['features']
            trad_features.append(self._flatten_traditional_features(features))
        self.trad_vectors = self.trad_scaler.fit_transform(np.array(trad_features))

    def _flatten_traditional_features(self, features: Dict) -> np.ndarray:
        """Flatten traditional features into a 1D array."""
        flattened = []
        
        scalar_features = ['spectral_centroid', 'spectral_bandwidth', 
                         'spectral_rolloff', 'tempo', 'zero_crossing_rate']
        for feature_name in scalar_features:
            if feature_name in features:
                flattened.append(float(features[feature_name]))
        
        array_features = ['mfcc_mean', 'mfcc_var']
        for feature_name in array_features:
            if feature_name in features:
                flattened.extend(features[feature_name].flatten())
        
        return np.array(flattened)

    def find_similar(self, 
                    query_file: str, 
                    n_results: int = 5, 
                    method: str = 'combined',
                    cnn_weight: float = 0.6) -> List[Tuple[str, float]]:
        """Find similar sounds using specified method."""
        if query_file not in self.filenames:
            raise ValueError(f"Query file {query_file} not found")
            
        idx = self.filenames.index(query_file)
        
        if method == 'traditional':
            query_vector = self.trad_vectors[idx]
            similarities = cosine_similarity([query_vector], self.trad_vectors)[0]
        elif method == 'cnn':
            query_vector = self.cnn_vectors[idx]
            similarities = cosine_similarity([query_vector], self.cnn_vectors)[0]
        else:  # combined
            trad_similarities = cosine_similarity([self.trad_vectors[idx]], self.trad_vectors)[0]
            cnn_similarities = cosine_similarity([self.cnn_vectors[idx]], self.cnn_vectors)[0]
            similarities = cnn_weight * cnn_similarities + (1 - cnn_weight) * trad_similarities
        
        # Get top N similar sounds (excluding the query itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_results+1]
        return [(self.filenames[idx], similarities[idx]) for idx in similar_indices]

    def compare_methods(self, query_file: str = None, n_results: int = 5):
        """Compare results from all three methods."""
        if query_file is None:
            query_file = random.choice(self.filenames)
            
        print(f"\nQuery file: {query_file}")
        print("=" * 50)
        
        methods = {
            'Traditional Features': ('traditional', 0),
            'CNN Features': ('cnn', 0),
            'Combined (60% CNN)': ('combined', 0.6),
            'Combined (40% CNN)': ('combined', 0.4)
        }
        
        results = {}
        for method_name, (method, weight) in methods.items():
            similar = self.find_similar(query_file, n_results, method, weight)
            results[method_name] = similar
        
        # Print comparison table
        headers = ["Rank"] + list(methods.keys())
        table = []
        
        for i in range(n_results):
            row = [f"#{i+1}"]
            for method_name in methods.keys():
                filename, score = results[method_name][i]
                row.append(f"{filename}\n({score:.3f})")
            table.append(row)
        
        print(tabulate(table, headers=headers, tablefmt="grid"))
        return results

    def analyze_differences(self, results_dict: Dict):
        """Analyze how different the results are between methods."""
        methods = list(results_dict.keys())
        n_methods = len(methods)
        
        print("\nResult Overlap Analysis:")
        print("=" * 50)
        
        for i in range(n_methods):
            for j in range(i+1, n_methods):
                method1, method2 = methods[i], methods[j]
                files1 = set(file for file, _ in results_dict[method1])
                files2 = set(file for file, _ in results_dict[method2])
                
                overlap = len(files1.intersection(files2))
                print(f"\n{method1} vs {method2}:")
                print(f"Common results: {overlap} out of {len(files1)}")
                print(f"Unique to {method1}: {len(files1 - files2)}")
                print(f"Unique to {method2}: {len(files2 - files1)}")

def main():
    comparator = MethodComparison()
    
    # Test with 3 random samples
    for _ in range(3):
        results = comparator.compare_methods(n_results=5)
        comparator.analyze_differences(results)
        input("\nPress Enter to see next comparison...")

if __name__ == "__main__":
    main() 