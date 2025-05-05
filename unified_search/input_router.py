import os
from unified_search.config import SAMPLES_DIR


def route_input(user_input, top_k=5):
    """
    Detects input type and routes to the correct engine.
    Returns: (engine_name, raw_results)
    """
    # If input looks like a file path to audio
    if isinstance(user_input, str) and user_input.lower().endswith(('.wav', '.mp3')):
        if not os.path.isfile(user_input):
            # Return an error if the file does not exist
            return "audio_to_audio", [{"error": f"File not found: {user_input}"}]
        from audio_to_audio_recommender.app.recommend_samples import AudioRecommender
        recommender = AudioRecommender()
        dataset_files = set(recommender.searcher.filenames)
        filename = os.path.basename(user_input)
        if filename in dataset_files:
            results = recommender.get_recommendations_from_dataset(filename, n_results=top_k)
        else:
            results = recommender.get_recommendations_from_file(user_input, n_results=top_k)
        results = [
            {"filename": fname, "score": float(score), "type": typ}
            for fname, score, typ in results
        ]
        return "audio_to_audio", results

    # Text-to-audio: string (not a file path)
    elif isinstance(user_input, str):
        from text_to_audio_recommender.models.clap_wrapper import ClapWrapper
        from unified_search.metadata import get_metadata
        samples_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            SAMPLES_DIR
        )
        samples_dir = os.path.abspath(samples_dir)
        audio_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(".wav")]
        audio_paths = [os.path.join(samples_dir, f) for f in audio_files]
        clap = ClapWrapper()
        text_emb = clap.encode_text([user_input])
        audio_emb = clap.encode_audio(audio_paths)
        similarity = clap.compute_similarity(text_emb, audio_emb)[0]
        top_indices = similarity.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            fname = audio_files[idx]
            score = similarity[idx]
            meta = get_metadata(fname)
            results.append({"filename": fname, "score": float(score), "type": meta.get('type', None)})
        return "text_to_audio", results

    else:
        return "error", [{"error": "Input must be a string (text or audio file path)"}]
