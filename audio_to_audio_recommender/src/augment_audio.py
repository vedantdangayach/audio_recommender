# src/augment_audio.py

import numpy as np
import torch
import torchaudio
import soundfile as sf
from torch_audiomentations import (
    Compose, 
    Gain,
    PolarityInversion,
    AddColoredNoise,
    HighPassFilter,
    LowPassFilter,
    PeakNormalization
)
import os
from tqdm import tqdm
from typing import List

class AudioAugmenter:
    def __init__(self, sr: int = 22050):
        """
        Initialize audio augmentation pipeline using torch-audiomentations.
        
        Args:
            sr: Sample rate
        """
        self.sr = sr
        
        # Create augmentation pipeline
        self.augmentor = Compose([
            # Add colored noise (more musical than white noise)
            AddColoredNoise(
                min_snr_in_db=15.0,
                max_snr_in_db=30.0,
                p=0.5
            ),
            # Random gain changes
            Gain(
                min_gain_in_db=-6.0,
                max_gain_in_db=6.0,
                p=0.5
            ),
            # High-pass filter (removes low frequencies)
            HighPassFilter(
                min_cutoff_freq=20.0,
                max_cutoff_freq=2400.0,
                p=0.3
            ),
            # Low-pass filter (removes high frequencies)
            LowPassFilter(
                min_cutoff_freq=2000.0,
                max_cutoff_freq=4000.0,
                p=0.3
            ),
            # Randomly invert polarity (phase inversion)
            PolarityInversion(p=0.3),
            # Always normalize at the end
            PeakNormalization(p=1.0)
        ])

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and convert to tensor."""
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
        return waveform

    def pad_or_trim(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """Ensure audio is the correct length."""
        if audio.shape[1] > target_length:
            start = (audio.shape[1] - target_length) // 2
            return audio[:, start:start + target_length]
        elif audio.shape[1] < target_length:
            pad_length = target_length - audio.shape[1]
            return torch.nn.functional.pad(audio, (0, pad_length))
        return audio

    def augment_audio(self, audio: torch.Tensor, n_augmentations: int = 2) -> List[torch.Tensor]:
        """Generate augmented versions of input audio."""
        augmented_samples = []
        target_length = audio.shape[1]
        
        for _ in range(n_augmentations):
            # Apply augmentations
            augmented = self.augmentor(audio.unsqueeze(0), sample_rate=self.sr)
            augmented = augmented.squeeze(0)
            
            # Ensure correct length
            augmented = self.pad_or_trim(augmented, target_length)
            augmented_samples.append(augmented)
            
        return augmented_samples

def main():
    """Generate augmented versions of all audio files."""
    # Input and output directories
    input_dir = "data/raw_samples"
    output_dir = "data/augmented_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    augmenter = AudioAugmenter()
    
    # Process all audio files
    audio_files = [f for f in os.listdir(input_dir) 
                  if f.endswith(('.wav', '.mp3'))]
    
    for audio_file in tqdm(audio_files, desc="Augmenting audio files"):
        try:
            # Load audio
            audio_path = os.path.join(input_dir, audio_file)
            waveform = augmenter.load_audio(audio_path)
            
            # Generate augmentations
            augmented_samples = augmenter.augment_audio(waveform, n_augmentations=2)
            
            # Save augmented versions
            for i, aug_audio in enumerate(augmented_samples):
                out_path = os.path.join(output_dir, f"{audio_file[:-4]}_aug{i+1}.wav")
                sf.write(
                    out_path, 
                    aug_audio.numpy().T, 
                    augmenter.sr
                )
                
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()