import torch
import logging
import librosa
import numpy as np
from transformers import ClapProcessor, ClapModel
from typing import List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clap_wrapper")

class ClapWrapper:
    """
    A wrapper class for the CLAP model that provides easy-to-use methods for encoding
    text and audio into a shared embedding space.
    """
    
    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: str = None):
        """
        Initialize the CLAP wrapper.
        
        Args:
            model_name (str): Name of the CLAP model to use
            device (str): Device to run the model on ('cuda' or 'cpu'). If None, will auto-detect.
        """
        if device is None:
            # Use MPS (Metal) if available, otherwise CPU
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        if device == "cpu":
            logger.warning("Running on CPU. For faster processing, consider using GPU acceleration.")
        
        self.device = device
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(device)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Set default audio parameters
        self.sample_rate = 48000  # CLAP expects 48kHz audio
        self.max_length = 480000  # 10 seconds at 48kHz
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            # Load audio with correct sample rate
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Pad or trim to max_length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, max(0, self.max_length - len(audio))))
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {str(e)}")
            raise
            
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text descriptions into embeddings.
        
        Args:
            text (Union[str, List[str]]): Text description(s) to encode
            
        Returns:
            np.ndarray: Text embeddings
        """
        if isinstance(text, str):
            text = [text]
            
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            
            return outputs.cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise
            
    def encode_audio(self, audio_paths: Union[str, List[str]]) -> np.ndarray:
        """
        Encode audio files into embeddings.
        
        Args:
            audio_paths (Union[str, List[str]]): Path(s) to audio file(s)
            
        Returns:
            np.ndarray: Audio embeddings
        """
        try:
            if isinstance(audio_paths, str):
                audio_paths = [audio_paths]
            
            # Process audio in batches
            batch_size = 16  # Smaller batch size for MPS
            all_embeddings = []
            
            for i in range(0, len(audio_paths), batch_size):
                batch_paths = audio_paths[i:i + batch_size]
                audio_arrays = [self.load_audio(path) for path in batch_paths]
                
                inputs = self.processor(
                    audios=audio_arrays,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.get_audio_features(**inputs)
                
                all_embeddings.append(outputs.cpu().numpy())
            
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error encoding audio: {str(e)}")
            raise

    def compute_similarity(self, text_embeddings: np.ndarray, audio_embeddings: np.ndarray, 
                          temperature: float = 0.1, use_softmax: bool = True) -> np.ndarray:
        """
        Compute similarity between text and audio embeddings with optional softmax normalization
        
        Args:
            text_embeddings: Text embeddings of shape (n_texts, embedding_dim)
            audio_embeddings: Audio embeddings of shape (n_audios, embedding_dim)
            temperature: Temperature parameter for scaling (lower = more concentrated)
            use_softmax: Whether to apply softmax normalization
        
        Returns:
            Similarity matrix of shape (n_texts, n_audios)
        """
        try:
            # Normalize embeddings
            text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            audio_norm = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
            
            # Compute similarity matrix
            similarity = np.dot(text_norm, audio_norm.T)
            
            if use_softmax:
                # Apply temperature scaling
                scaled_similarity = similarity / temperature
                
                # Compute softmax
                # Subtract max for numerical stability
                max_sim = np.max(scaled_similarity, axis=1, keepdims=True)
                exp_sim = np.exp(scaled_similarity - max_sim)
                softmax_sim = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
                
                return softmax_sim
            else:
                # Return raw similarity scores
                return similarity
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
