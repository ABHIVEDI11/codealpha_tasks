import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def generate_synthetic_audio_data(n_samples=1000):
    """Generate synthetic audio data for emotion recognition"""
    np.random.seed(42)
    
    emotions = ['happy', 'sad', 'angry', 'neutral', 'fearful', 'surprised']
    sample_rate = 22050
    duration = 1  # 1 second
    
    data = []
    labels = []
    
    for i in range(n_samples):
        # Generate random audio signal
        emotion = np.random.choice(emotions)
        
        # Generate simple audio signal
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if emotion == 'happy':
            audio = 0.3 * np.sin(2 * np.pi * 400 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)
        elif emotion == 'sad':
            audio = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)
        elif emotion == 'angry':
            audio = 0.4 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
        elif emotion == 'fearful':
            audio = 0.25 * np.sin(2 * np.pi * 250 * t) + 0.15 * np.sin(2 * np.pi * 350 * t)
        elif emotion == 'surprised':
            audio = 0.35 * np.sin(2 * np.pi * 450 * t) + 0.25 * np.sin(2 * np.pi * 700 * t)
        else:  # neutral
            audio = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 500 * t)
        
        # Add some noise
        audio += np.random.normal(0, 0.01, len(audio))
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        data.append(audio)
        labels.append(emotion)
    
    return data, labels

def extract_simple_features(audio_data, sample_rate=22050):
    """Extract simple features from audio data"""
    features = []
    
    for audio in audio_data:
        # Extract basic statistical features
        feature_vector = [
            np.mean(audio),           # Mean amplitude
            np.std(audio),            # Standard deviation
            np.max(audio),            # Maximum amplitude
            np.min(audio),            # Minimum amplitude
            np.median(audio),         # Median amplitude
            np.percentile(audio, 25), # 25th percentile
            np.percentile(audio, 75), # 75th percentile
            np.var(audio),            # Variance
            np.sum(np.abs(audio)),    # Sum of absolute values
            len(audio[audio > 0.5]),  # Number of high amplitude samples
            len(audio[audio < -0.5]), # Number of low amplitude samples
            np.sum(audio**2),         # Energy
            np.sqrt(np.mean(audio**2)) # RMS
        ]
        features.append(feature_vector)
    
    return np.array(features)

def load_audio_data():
    """Load audio data from files"""
    try:
        audio_dir = 'audio_samples'
        if not os.path.exists(audio_dir):
            print("Audio samples directory not found, generating synthetic data...")
            return generate_synthetic_audio_data()
        
        emotions = ['happy', 'sad', 'angry', 'neutral', 'fearful', 'surprised']
        data = []
        labels = []
        
        for emotion in emotions:
            audio_file = os.path.join(audio_dir, f'{emotion}_sample.npy')
            if os.path.exists(audio_file):
                audio = np.load(audio_file)
                # Create multiple samples by adding noise
                for i in range(100):  # 100 samples per emotion
                    noisy_audio = audio + np.random.normal(0, 0.01, len(audio))
                    data.append(noisy_audio)
                    labels.append(emotion)
        
        if len(data) > 0:
            print(f"Audio data loaded: {len(data)} samples")
            return data, labels
        else:
            print("No audio files found, generating synthetic data...")
            return generate_synthetic_audio_data()
            
    except Exception as e:
        print(f"Error loading audio data: {e}")
        print("Generating synthetic data instead...")
        return generate_synthetic_audio_data()

def create_emotion_models():
    """Create scikit-learn models for emotion recognition"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    return models

def create_visualizations(y_test, y_pred, class_names):
    """Create confusion matrix plot"""
    plt.figure(figsize=(10, 8))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Emotion Recognition Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('emotion_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_emotion_model():
    """Train emotion recognition model"""
    print("Loading audio data...")
    audio_data, labels = load_audio_data()
    
    print("Extracting features...")
    features = extract_simple_features(audio_data)
    
    print("Preparing data...")
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training models...")
    models = create_emotion_models()
    best_model = None
    best_score = 0
    best_model_name = ""
    best_predictions = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        score = model.score(X_test_scaled, y_test)
        
        print(f"{name} Accuracy: {score:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name
            best_predictions = y_pred
    
    # Save best model and preprocessing data
    print(f"\nSaving best model: {best_model_name} (Accuracy: {best_score:.4f})")
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_params': {'sample_rate': 22050}
    }
    
    with open('emotion_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'emotion_model.pkl'")
    
    # Create visualizations
    create_visualizations(y_test, best_predictions, label_encoder.classes_)

if __name__ == "__main__":
    train_emotion_model()
