import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_combined import CombinedSimilaritySearch
from src.extract_features_cnn import CombinedFeatureExtractor
from typing import List, Tuple
import shutil
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioRecommender:
    def __init__(self, 
                 features_path: str = "/Users/vedantdangayach/audio_recommender/audio_to_audio_recommender/data/embeddings_combined_v2/combined_features.joblib",
                 type_classifier_path: str = "/Users/vedantdangayach/audio_recommender/audio_to_audio_recommender/models/type_classifier.joblib",
                 cnn_weight: float = 0.4,
                 type_confidence_threshold: float = 0.6):
        """Initialize the recommender system with combined features and type filtering."""
        self.searcher = CombinedSimilaritySearch(
            features_path=features_path,
            type_classifier_path=type_classifier_path,
            cnn_weight=cnn_weight,
            type_confidence_threshold=type_confidence_threshold
        )
        self.feature_extractor = CombinedFeatureExtractor()
        logger.info(f"Loaded {len(self.searcher.filenames)} samples")
        logger.info(f"Using CNN weight: {cnn_weight:.2f}, Traditional weight: {1-cnn_weight:.2f}")

    def get_recommendations_from_file(self, 
                                    file_path: str, 
                                    n_results: int = 5,
                                    enforce_type_matching: bool = True) -> List[Tuple[str, float, str]]:
        """Get recommendations for an uploaded file."""
        try:
            features = self.feature_extractor.extract_combined_features(file_path)
            if features is None:
                raise Exception("Could not extract features from the file")
            return self.searcher.find_similar_to_features(
                features, 
                n_results,
                enforce_type_matching=enforce_type_matching
            )
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            return []

    def get_recommendations_from_dataset(self, 
                                       query_file: str, 
                                       n_results: int = 5,
                                       enforce_type_matching: bool = True) -> List[Tuple[str, float, str]]:
        """Get recommendations for a file from the existing dataset."""
        try:
            return self.searcher.find_similar(
                query_file, 
                n_results,
                enforce_type_matching=enforce_type_matching
            )
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

    def print_recommendations(self, query_file: str, recommendations: List[Tuple[str, float, str]]):
        """Print recommendations in a nice format."""
        print("\n" + "="*50)
        print(f"Query sound: {query_file}")
        print("="*50)
        print("\nMost Similar Sounds:")
        print("-"*50)
        
        for i, (filename, score, audio_type) in enumerate(recommendations, 1):
            similarity_percentage = score * 100
            print(f"{i}. {filename}")
            print(f"   Type: {audio_type}")
            print(f"   Similarity: {similarity_percentage:.1f}%")
            print()

    def list_available_samples_by_type(self) -> dict:
        """Get a sorted list of all available samples grouped by type."""
        samples_by_type = {}
        for filename in self.searcher.filenames:
            audio_type = self.searcher.sample_types[filename]
            if audio_type not in samples_by_type:
                samples_by_type[audio_type] = []
            samples_by_type[audio_type].append(filename)
        
        # Sort within each type
        for audio_type in samples_by_type:
            samples_by_type[audio_type].sort()
        
        return samples_by_type

    def adjust_weights(self, cnn_weight: float):
        """Adjust the balance between CNN and traditional features."""
        self.searcher.adjust_weights(cnn_weight)
        logger.info(f"Updated weights - CNN: {cnn_weight:.2f}, Traditional: {1-cnn_weight:.2f}")

def handle_file_upload() -> str:
    """Handle file upload process."""
    while True:
        file_path = input("\nEnter the path to your audio file (or 'b' to go back): ").strip()
        
        if file_path.lower() == 'b':
            return None
            
        if not os.path.exists(file_path):
            print("File not found. Please enter a valid file path.")
            continue
            
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = os.path.basename(file_path)
        destination = os.path.join(upload_dir, filename)
        try:
            shutil.copy2(file_path, destination)
            return destination
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            return None

def main():
    print("\nInitializing Audio Recommendation System...")
    recommender = AudioRecommender()
    
    while True:
        print("\n" + "="*50)
        print("Audio Sample Finder (Type-Aware CNN + Traditional Features)")
        print("="*50)
        print("\nChoose an option:")
        print("1. Find similar samples from the database")
        print("2. Upload your own audio file")
        print("3. Adjust CNN/Traditional feature weights")
        print("4. Toggle type filtering")
        print("5. Quit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\nAvailable samples by type:")
            print("-"*50)
            samples_by_type = recommender.list_available_samples_by_type()
            
            # Print samples grouped by type
            total_count = 0
            for audio_type, samples in samples_by_type.items():
                print(f"\n{audio_type.upper()}:")
                for filename in samples:
                    total_count += 1
                    print(f"{total_count}. {filename}")
            
            try:
                file_choice = input("\nEnter the number of the sample (or 'b' to go back): ")
                
                if file_choice.lower() == 'b':
                    continue
                
                file_index = int(file_choice) - 1
                all_samples = [s for samples in samples_by_type.values() for s in samples]
                
                if 0 <= file_index < len(all_samples):
                    query_file = all_samples[file_index]
                    recommendations = recommender.get_recommendations_from_dataset(query_file)
                    recommender.print_recommendations(query_file, recommendations)
                else:
                    print("\nPlease enter a valid number")
            
            except ValueError:
                print("\nPlease enter a valid number")

        elif choice == "2":
            uploaded_file = handle_file_upload()
            if uploaded_file:
                print("\nAnalyzing your audio file...")
                recommendations = recommender.get_recommendations_from_file(uploaded_file)
                if recommendations:
                    recommender.print_recommendations(os.path.basename(uploaded_file), recommendations)
                else:
                    print("\nCould not process the audio file. Please try another file.")
                
                try:
                    os.remove(uploaded_file)
                except:
                    pass

        elif choice == "3":
            try:
                weight = float(input("\nEnter CNN weight (0.0 to 1.0, default is 0.4): ").strip())
                if 0 <= weight <= 1:
                    recommender.adjust_weights(weight)
                    print(f"\nWeights updated - CNN: {weight:.2f}, Traditional: {1-weight:.2f}")
                else:
                    print("\nPlease enter a value between 0 and 1")
            except ValueError:
                print("\nPlease enter a valid number")

        elif choice == "4":
            current = recommender.searcher.type_confidence_threshold
            print(f"\nCurrent type matching threshold: {current:.2f}")
            try:
                new_threshold = float(input("Enter new threshold (0.0 to 1.0, or -1 to disable): ").strip())
                if -1 <= new_threshold <= 1:
                    recommender.searcher.type_confidence_threshold = new_threshold
                    if new_threshold == -1:
                        print("\nType filtering disabled")
                    else:
                        print(f"\nType matching threshold updated to {new_threshold:.2f}")
                else:
                    print("\nPlease enter a value between -1 and 1")
            except ValueError:
                print("\nPlease enter a valid number")

        elif choice == "5":
            print("\nThank you for using Audio Sample Finder!")
            break
            
        else:
            print("\nPlease enter a valid choice (1-5)")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()