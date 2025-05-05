import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
from models.clap_wrapper import ClapWrapper

# Set up paths
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_samples")

# Initialize CLAP wrapper (cache to avoid reloading)
@st.cache_resource
def get_clap():
    return ClapWrapper()

clap = get_clap()

st.title("Text-to-Audio Similarity Demo (CLAP)")

# Text input
text_prompt = st.text_input("Enter a text prompt describing the sound:", "punchy kick drum")

# List available audio files
audio_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".wav")]
audio_paths = [os.path.join(SAMPLES_DIR, f) for f in audio_files]

if st.button("Compute Similarity"):
    with st.spinner("Encoding and comparing..."):
        # Encode text and all audio
        text_emb = clap.encode_text([text_prompt])
        audio_emb = clap.encode_audio(audio_paths)
        similarity = clap.compute_similarity(text_emb, audio_emb)[0]

        # Get top 5 matches
        top_indices = similarity.argsort()[::-1][:5]
        st.subheader("Top 5 Most Similar Sounds")
        for idx in top_indices:
            fname = audio_files[idx]
            score = similarity[idx]
            st.write(f"**{fname}**: {score:.3f}")
            st.audio(audio_paths[idx])

st.markdown("---")
st.caption("Powered by CLAP and Streamlit")
