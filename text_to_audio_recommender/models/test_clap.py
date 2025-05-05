import os
import numpy as np
from clap_wrapper import ClapWrapper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_clap")

def test_clap_wrapper():
    """Comprehensive validation test for the CLAP wrapper"""
    
    # Initialize wrapper
    clap = ClapWrapper()
    
    # Test 1: Text Encoding
    logger.info("\nTest 1: Text Encoding")
    test_prompts = [
        "punchy kick drum",
        "warm vocal sample",
        "bright cymbal crash",
        "deep bass synth",
        "atmospheric pad"
    ]
    
    text_embeddings = clap.encode_text(test_prompts)
    logger.info(f"✓ Text encoding shape: {text_embeddings.shape}")
    logger.info(f"✓ Text embedding norm: {np.linalg.norm(text_embeddings[0]):.3f}")
    
    # Test 2: Audio Encoding
    logger.info("\nTest 2: Audio Encoding")
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_samples")
    
    # Test with different types of audio files
    test_files = [
        "KM_WbR_TOM_02.wav",      # Percussion
        "KM_WbR_SHAK_21.wav",     # Shaker
        "KM_WbR_HATS_FLUKE_02.wav" # Hi-hats
    ]
    
    audio_files = [os.path.join(samples_dir, f) for f in test_files]
    
    if all(os.path.exists(f) for f in audio_files):
        audio_embeddings = clap.encode_audio(audio_files)
        logger.info(f"✓ Audio encoding shape: {audio_embeddings.shape}")
        logger.info(f"✓ Audio embedding norm: {np.linalg.norm(audio_embeddings[0]):.3f}")
        
        # Test 3: Similarity Computation
        logger.info("\nTest 3: Similarity Computation")
        # Test with different temperature values
        for temp in [0.1, 0.05, 0.01]:
            similarity = clap.compute_similarity(text_embeddings, audio_embeddings, temperature=temp)
            logger.info(f"\nSimilarity scores (temperature={temp}):")
            
            for i, prompt in enumerate(test_prompts):
                logger.info(f"\n{prompt}:")
                scores = similarity[i]
                # Get top match
                top_idx = np.argmax(scores)
                logger.info(f"  Best match: {test_files[top_idx]} (score: {scores[top_idx]:.3f})")
                
                # Print all scores for analysis
                for j, audio in enumerate(test_files):
                    logger.info(f"  {audio}: {scores[j]:.3f}")
    
    else:
        logger.error(f"Some test audio files not found in {samples_dir}")
        available_files = os.listdir(samples_dir)
        logger.info(f"Available files: {available_files[:5]}...")

def validate_embeddings(embeddings):
    """Validate embedding properties"""
    # Check for NaN values
    if np.any(np.isnan(embeddings)):
        return False, "Contains NaN values"
    
    # Check embedding dimensions
    if embeddings.shape[1] != 512:  # CLAP typically uses 512-dim embeddings
        return False, f"Unexpected embedding dimension: {embeddings.shape[1]}"
    
    # Check if embeddings are zero
    if np.all(embeddings == 0):
        return False, "All zero embeddings"
    
    return True, "Validation passed"

if __name__ == "__main__":
    test_clap_wrapper() 