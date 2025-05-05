import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, Tuple, Optional

class AudioTypeClassifier:
    def __init__(self):
        self.type_patterns = {
            'kick': r'(kick|bd_|bass\s*drum)',
            'snare': r'(snare|sd_)',
            'hihat': r'(hi[\s-]*hat|hh_|hat)',
            'clap': r'clap',
            'percussion': r'(perc|conga|bongo|tabla)',
            'cymbal': r'(cymbal|crash|ride)',
            'tom': r'tom',
            'vocal': r'(vox|vocal|acapella|voc)',
            'fx': r'(fx|effect|riser|sweep|impact)',
            'bass': r'(bass|sub|808)',
            'synth': r'(synth|lead|pad|arp)',
            'melodic': r'(melodic|melody|tonal|harmonic)',
        }
        self.label_encoder = LabelEncoder()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def guess_type_from_filename(self, filename: str) -> str:
        """Guess audio type from filename using regex patterns."""
        filename = filename.lower()
        
        for type_name, pattern in self.type_patterns.items():
            if re.search(pattern, filename):
                return type_name
        
        return "unknown"
    
    def extract_type_features(self, features: Dict) -> np.ndarray:
        """Extract relevant features for type classification."""
        # Use a subset of traditional features that are most relevant for type detection
        trad_features = features['traditional_features']
        
        feature_vector = np.concatenate([
            [trad_features['spectral_centroid']],
            [trad_features['spectral_bandwidth']],
            [trad_features['spectral_rolloff']],
            [trad_features['zero_crossing_rate']],
            trad_features['mfcc_mean'],
            trad_features['mfcc_var']
        ])
        
        return feature_vector
    
    def train(self, features_dict: Dict, test_size: float = 0.15) -> Tuple[float, Dict]:
        """Train the type classifier on extracted features."""
        X = []  # Features
        y = []  # Type labels
        filenames = []
        
        # Prepare dataset
        for filename, features in features_dict.items():
            try:
                # Extract features for classification
                feature_vector = self.extract_type_features(features)
                
                # Get type from filename
                audio_type = self.guess_type_from_filename(filename)
                
                X.append(feature_vector)
                y.append(audio_type)
                filenames.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        X = np.array(X)
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train, y_train)
        test_score = self.classifier.score(X_test, y_test)
        
        # Get feature importances
        feature_importance = dict(zip(
            ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
             'zero_crossing_rate'] + [f'mfcc_{i}' for i in range(40)],
            self.classifier.feature_importances_
        ))
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        return test_score, feature_importance
    
    def predict_type(self, features: Dict) -> Tuple[str, float]:
        """Predict audio type and return confidence."""
        feature_vector = self.extract_type_features(features)
        
        # Get prediction and probability
        pred_encoded = self.classifier.predict([feature_vector])[0]
        pred_proba = np.max(self.classifier.predict_proba([feature_vector]))
        
        predicted_type = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        return predicted_type, pred_proba
    
    def save(self, model_path: str):
        """Save the trained classifier."""
        joblib.dump({
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'type_patterns': self.type_patterns
        }, model_path)
    
    @classmethod
    def load(cls, model_path: str) -> 'AudioTypeClassifier':
        """Load a trained classifier."""
        data = joblib.load(model_path)
        
        classifier = cls()
        classifier.classifier = data['classifier']
        classifier.label_encoder = data['label_encoder']
        classifier.type_patterns = data['type_patterns']
        
        return classifier

def main():
    """Train and save the type classifier."""
    # Load feature database
    features_path = "data/embeddings_combined_v2/combined_features.joblib"
    features_dict = joblib.load(features_path)
    
    # Initialize and train classifier
    classifier = AudioTypeClassifier()
    test_score, feature_importance = classifier.train(features_dict)
    
    # Save trained classifier
    os.makedirs("models", exist_ok=True)
    classifier.save("models/type_classifier.joblib")
    
    # Print top important features
    print("\nTop important features:")
    for feature, importance in sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]:
        print(f"{feature}: {importance:.3f}")

if __name__ == "__main__":
    main()
