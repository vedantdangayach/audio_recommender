import joblib

# Load the features file
features = joblib.load('data/embeddings/audio_features.joblib')

# Print the structure of the first item
print("Features file structure:")
first_key = list(features.keys())[0]
print(f"\nFirst key: {first_key}")
print(f"Type of first item: {type(features[first_key])}")
print(f"Content of first item: {features[first_key]}")