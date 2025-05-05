from search_similar import SoundSimilaritySearch
import os

def main():
    # Initialize the similarity search
    features_path = "data/embeddings/audio_features.joblib"
    searcher = SoundSimilaritySearch(features_path)

    # Print database stats
    print(f"\nTotal number of sounds in database: {len(searcher.filenames)}")
    
    # Show all available files
    print("\nAvailable files:")
    sorted_files = sorted(searcher.filenames)
    for i, filename in enumerate(sorted_files):
        print(f"{i+1}. {filename}")

    # Let user pick a file
    while True:
        try:
            choice = input("\nEnter the number of the file you want to test (1-{}): ".format(len(sorted_files)))
            file_index = int(choice) - 1
            if 0 <= file_index < len(sorted_files):
                query_file = sorted_files[file_index]
                break
            else:
                print("Please enter a valid number")
        except ValueError:
            print("Please enter a valid number")

    print(f"\nFinding similar sounds to: {query_file}")
    
    # Get similar sounds using cosine similarity
    similar_sounds = searcher.find_similar(query_file, n_results=10, metric='cosine')
    
    print("\nMost similar sounds (by cosine similarity):")
    print("----------------------------------------")
    for i, (filename, score) in enumerate(similar_sounds, 1):
        # Convert similarity score to percentage
        similarity_percentage = score * 100
        print(f"{i}. {filename}")
        print(f"   Similarity: {similarity_percentage:.1f}%")
        print()

if __name__ == "__main__":
    main()
