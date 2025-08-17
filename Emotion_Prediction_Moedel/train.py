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
    """Generate synthetic audio data for emotion recognition using more realistic patterns"""
    np.random.seed(42)
    
    emotions = ['happy', 'sad', 'angry', 'neutral', 'fearful', 'surprised']
    sample_rate = 22050
    duration = 3  # 3 seconds
    
    data = []
    labels = []
    
    for i in range(n_samples):
        # Generate random audio signal
        emotion = np.random.choice(emotions)
        
        # Create more realistic frequency patterns for different emotions
        if emotion == 'happy':
            # Higher frequencies, more energetic, varied pitch
            base_freq = np.random.uniform(200, 400)
            freqs = [base_freq + np.random.uniform(-50, 100) for _ in range(8)]
            amplitudes = np.random.uniform(0.2, 0.5, 8)
        elif emotion == 'sad':
            # Lower frequencies, slower, monotone
            base_freq = np.random.uniform(100, 250)
            freqs = [base_freq + np.random.uniform(-20, 30) for _ in range(6)]
            amplitudes = np.random.uniform(0.1, 0.3, 6)
        elif emotion == 'angry':
            # High amplitude, harsh frequencies, rapid changes
            base_freq = np.random.uniform(300, 600)
            freqs = [base_freq + np.random.uniform(-100, 200) for _ in range(10)]
            amplitudes = np.random.uniform(0.3, 0.7, 10)
        elif emotion == 'fearful':
            # Variable frequencies, trembling, irregular
            base_freq = np.random.uniform(150, 350)
            freqs = [base_freq + np.random.uniform(-80, 120) for _ in range(7)]
            amplitudes = np.random.uniform(0.15, 0.4, 7)
        elif emotion == 'surprised':
            # Sudden high frequencies, sharp transitions
            base_freq = np.random.uniform(400, 800)
            freqs = [base_freq + np.random.uniform(-150, 300) for _ in range(9)]
            amplitudes = np.random.uniform(0.25, 0.6, 9)
        else:  # neutral
            # Balanced frequencies, steady
            base_freq = np.random.uniform(200, 400)
            freqs = [base_freq + np.random.uniform(-30, 50) for _ in range(7)]
            amplitudes = np.random.uniform(0.15, 0.35, 7)
        
        # Generate audio signal with more realistic characteristics
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Add emotion-specific characteristics
        for j, (freq, amp) in enumerate(zip(freqs, amplitudes)):
            # Add some variation over time
            time_variation = 1 + 0.1 * np.sin(2 * np.pi * 2 * t + j)
            audio += amp * time_variation * np.sin(2 * np.pi * freq * t)
        
        # Add emotion-specific noise patterns
        if emotion == 'angry':
            # Add harsh noise
            noise = np.random.normal(0, 0.05, len(audio))
            audio += noise
        elif emotion == 'fearful':
            # Add trembling effect
            tremble = 0.02 * np.sin(2 * np.pi * 8 * t) * np.random.normal(0, 1, len(audio))
            audio += tremble
        elif emotion == 'surprised':
            # Add sudden amplitude changes
            sudden_changes = np.random.choice([0.5, 1.5], len(audio), p=[0.9, 0.1])
            audio *= sudden_changes
        
        # Add general background noise
        audio += np.random.normal(0, 0.01, len(audio))
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        data.append(audio)
        labels.append(emotion)
    
    return data, labels

def extract_mfcc_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract MFCC features from audio data"""
    features = []
    
    for audio in audio_data:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        target_length = 100
        if mfccs.shape[1] < target_length:
            # Pad with zeros
            padding = np.zeros((n_mfcc, target_length - mfccs.shape[1]))
            mfccs = np.hstack([mfccs, padding])
        else:
            # Truncate
            mfccs = mfccs[:, :target_length]
        
        features.append(mfccs)
    
    return np.array(features)

def create_emotion_models():
    """Create scikit-learn models for emotion recognition"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    return models

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

def train_emotion_model():
    """Train emotion recognition model"""
    print("Loading audio data...")
    audio_data, labels = load_audio_data()
    
    print("Extracting MFCC features...")
    features = extract_mfcc_features(audio_data)
    
    # Reshape features for CNN (add channel dimension)
    features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, categorical_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Create and train model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = len(label_encoder.classes_)
    
    model = create_cnn_model(input_shape, num_classes)
    
    print("Training CNN model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    # Save model and preprocessing objects
    model.save('emotion_model.h5')
    
    # Save label encoder and preprocessing info
    preprocessing_data = {
        'label_encoder': label_encoder,
        'sample_rate': 22050,
        'n_mfcc': 13,
        'target_length': 100
    }
    
    with open('preprocessing.pkl', 'wb') as f:
        pickle.dump(preprocessing_data, f)
    
    print("Model saved as 'emotion_model.h5'")
    print("Preprocessing data saved as 'preprocessing.pkl'")
    
    # Create visualizations
    create_visualizations(history, y_test_classes, y_pred_classes, 
                         label_encoder.classes_)
    
    return model, history

def create_visualizations(history, y_test, y_pred, class_names):
    """Create training history and confusion matrix plots"""
    plt.figure(figsize=(15, 5))
    
    # Training history
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Training loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Confusion matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('results/emotion_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    train_emotion_model()
