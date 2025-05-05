import os
import csv
import librosa

# Directory containing your audio files
AUDIO_DIR = "/Users/vedantdangayach/audio_recommender/audio_to_audio_recommender/data/raw_samples"  # Change as needed
OUTPUT_CSV = "metadata.csv"

# List of possible types to guess from filename
TYPES = ["kick", "snare", "pad", "fx", "hat", "perc", "clap", "bass", "lead", "vocal", "loop", "tom", "rim", "ride", "crash", "pluck", "stab", "chord", "drone", "atmo", "melody"]

def guess_type(filename):
    """Guess the type of the sample from its filename."""
    fname = filename.lower()
    for t in TYPES:
        if t in fname:
            return t
    return "unknown"

def get_duration(filepath):
    """Get duration of a wav file in seconds."""
    try:
        y, sr = librosa.load(filepath, sr=None, mono=True)
        return round(librosa.get_duration(y=y, sr=sr), 3)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    rows = []
    for fname in os.listdir(AUDIO_DIR):
        if fname.lower().endswith(".wav"):
            fpath = os.path.join(AUDIO_DIR, fname)
            duration = get_duration(fpath)
            if duration is None:
                continue
            typ = guess_type(fname)
            rows.append({"filename": fname, "type": typ, "duration": duration})

    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "type", "duration"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote metadata for {len(rows)} files to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
