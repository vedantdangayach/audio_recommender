import os
import sys
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import joblib
from models.clap_wrapper import ClapWrapper

def get_audio_embeddings(clap, audio_paths, cache_path):
    # If cache exists, load it
    if os.path.exists(cache_path):
        print(f"Loading cached audio embeddings from {cache_path}")
        cache = joblib.load(cache_path)
        # Check if all files are present in cache
        if set(cache['audio_paths']) == set(audio_paths):
            return cache['embeddings']
        else:
            print("Audio files changed, recomputing embeddings...")

    # Otherwise, compute and cache
    print("Encoding audio files (this may take a moment)...")
    embeddings = clap.encode_audio(audio_paths)
    joblib.dump({'audio_paths': audio_paths, 'embeddings': embeddings}, cache_path)
    print(f"Cached audio embeddings to {cache_path}")
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Text-to-Audio Similarity Search (CLAP, with caching)")
    parser.add_argument("--text", type=str, required=True, help="Text prompt to search for")
    parser.add_argument("--topk", type=int, default=5, help="Number of top matches to show")
    parser.add_argument("--samples_dir", type=str, default=None, 
                       help="Directory with audio samples (default: data/raw_samples)")
    parser.add_argument("--cache", type=str, default="audio_embeddings.joblib", 
                       help="Path to cache file")
    parser.add_argument("--device", type=str, choices=['mps', 'cpu'], default=None, 
                       help="Force device type (default: auto-detect)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for softmax (lower = more concentrated)")
    parser.add_argument("--raw_scores", action="store_true",
                       help="Use raw similarity scores instead of softmax")
    args = parser.parse_args()

    # Set up paths
    samples_dir = args.samples_dir or os.path.join(project_root, "data", "raw_samples")
    audio_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(".wav")]
    audio_paths = [os.path.join(samples_dir, f) for f in audio_files]

    if not audio_files:
        print(f"No .wav files found in {samples_dir}")
        return

    print(f"Loaded {len(audio_files)} audio files from {samples_dir}")

    # Initialize CLAP
    clap = ClapWrapper(device=args.device)

    # Get (or compute) audio embeddings
    cache_path = os.path.join(samples_dir, args.cache)
    audio_emb = get_audio_embeddings(clap, audio_paths, cache_path)

    # Encode text and compute similarity
    print(f"Encoding text: \"{args.text}\"")
    text_emb = clap.encode_text([args.text])
    similarity = clap.compute_similarity(
        text_emb, 
        audio_emb,
        temperature=args.temperature,
        use_softmax=not args.raw_scores
    )
    
    # Get top matches
    top_indices = similarity[0].argsort()[::-1][:args.topk]

    print("\nTop matches:")
    for rank, idx in enumerate(top_indices, 1):
        fname = audio_files[idx]
        score = similarity[0][idx]
        if not args.raw_scores:
            print(f"{rank}. {fname}  (probability: {score:.1%})")
        else:
            print(f"{rank}. {fname}  (similarity: {score:.3f})")

if __name__ == "__main__":
    main()
