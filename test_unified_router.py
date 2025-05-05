from unified_search.input_router import route_input
from unified_search.unified_search import search_audio
from unified_search.config import AUDIO_DB_PATH, DEFAULT_TOP_K, METADATA_CSV_PATH, CACHE_PATH

AUDIO_DB_PATH = "/Users/vedantdangayach/audio_recommender/text_to_audio_recommender/data/raw_samples"
DEFAULT_TOP_K = 5


# Test with a text query
print("=== Text-to-Audio Test ===")
engine, results = route_input("airy pad", top_k=3)
print("Engine:", engine)
for r in results:
    print(r)

print("\n=== Audio-to-Audio Test ===")
audio_path = "/Users/vedantdangayach/audio_recommender/text_to_audio_recommender/data/raw_samples/KM_WbR_FX_025.wav"
engine, results = route_input(audio_path, top_k=3)
print("Engine:", engine)
for r in results:
    print(r)

print("=== Text-to-Audio with Type Filter ===")
results = search_audio(
    input="airy kick",
    top_k=5,
    filters={"type": "kick"}
)
print(results)

print("\n=== Audio-to-Audio with Duration Filter ===")
audio_path = "audio_to_audio_recommender/data/some_sample.wav"  # <-- update to a real file!
results = search_audio(
    input=audio_path,
    top_k=5,
    filters={"min_duration": 0.5, "max_duration": 2.0}
)
print(results)

print("\n=== Bad Input Test ===")
results = search_audio(
    input=12345,  # Invalid input type
    top_k=5
)
print(results)

print("\n=== Missing File Test ===")
results = search_audio(
    input="audio_to_audio_recommender/data/does_not_exist.wav",
    top_k=5
)
if results and isinstance(results, list) and "error" in results[0]:
    print("Error:", results[0]["error"])
else:
    print(results)
