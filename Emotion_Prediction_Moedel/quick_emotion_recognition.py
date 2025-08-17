"""
Quick Emotion Recognition from Speech

Fast version of emotion recognition system optimized for speed and testing.
Uses synthetic data and scikit-learn models for quick demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Audio Processing
import librosa

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    print("Using TensorFlow")
except ImportError:
    print("TensorFlow not available - using scikit-learn")
    TENSORFLOW_AVAILABLE = False

class FastEmotionRecognition:
    """
    Fast Emotion Recognition System
    
    Optimized for quick training and testing with synthetic data.
    Provides both deep learning and traditional ML approaches.
    """
    
    def __init__(self):
        """Initialize the fast emotion recognition system"""
        self.emotion_mapping = {
            'happy': 0, 'sad': 1, 'angry': 2, 'neutral': 3
        }
        self.reverse_mapping = {v: k for k, v in self.emotion_mapping.items()}
    
    def create_fast_dataset(self, samples_per_emotion=25):
        """
        Create a small synthetic dataset for fast testing
        
        Args:
            samples_per_emotion (int): Number of samples per emotion
            
        Returns:
            tuple: (audio_data, labels) - Audio samples and emotion labels
        """
        print("Creating fast dataset...")
        
        audio_data = []
        labels = []
        emotions = ['happy', 'sad', 'angry', 'neutral']
        sample_rate = 22050
        duration = 2  # Shorter duration for speed
        
        for emotion in emotions:
            print(f"   Creating {samples_per_emotion} {emotion} samples...")
            
            for i in range(samples_per_emotion):
                # Different frequency patterns for different emotions
                if emotion == 'happy':
                    freq = 450 + np.random.randint(-50, 50)
                    amplitude = 0.5 + np.random.uniform(-0.1, 0.1)
                elif emotion == 'sad':
                    freq = 200 + np.random.randint(-30, 30)
                    amplitude = 0.3 + np.random.uniform(-0.05, 0.05)
                elif emotion == 'angry':
                    freq = 400 + np.random.randint(-40, 40)
                    amplitude = 0.7 + np.random.uniform(-0.1, 0.1)
                else:  # neutral
                    freq = 300 + np.random.randint(-25, 25)
                    amplitude = 0.4 + np.random.uniform(-0.05, 0.05)
                
                # Generate audio signal
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                audio = np.sin(2 * np.pi * freq * t) * amplitude
                
                # Add noise and modulation for realism
                audio += np.random.normal(0, 0.05, len(audio))
                mod_freq = 5 + np.random.uniform(0, 10)
                audio *= (1 + 0.1 * np.sin(2 * np.pi * mod_freq * t))
                
                audio_data.append(audio)
                labels.append(emotion)
        
        print(f"Created {len(audio_data)} samples")
        return audio_data, labels
    
    def extract_fast_mfcc(self, audio, sample_rate=22050):
        """
        Extract MFCC features optimized for speed
        
        Uses fewer coefficients and shorter time frames for faster processing.
        
        Args:
            audio (np.array): Audio signal
            sample_rate (int): Sample rate
            
        Returns:
            np.array: MFCC features (20 coefficients x 87 time frames)
        """
        try:
            # Ensure mono audio
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Extract MFCC with fewer coefficients for speed
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sample_rate, 
                n_mfcc=20,  # Reduced from 40 to 20
                hop_length=512,
                n_fft=2048
            )
            
            # Pad or truncate to smaller size
            mfcc_padded = np.zeros((20, 87))  # Reduced time frames
            mfcc_padded[:, :min(mfcc.shape[1], 87)] = mfcc[:, :87]
            
            return mfcc_padded
            
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return None
    
    def prepare_fast_dataset(self, audio_data, labels):
        """
        Prepare dataset for fast training
        
        Args:
            audio_data (list): List of audio samples
            labels (list): List of emotion labels
            
        Returns:
            tuple: (X_mfcc, y_encoded) - Features and encoded labels
        """
        print("Preparing dataset...")
        
        X_mfcc = []
        y_encoded = []
        
        for i, (audio, label) in enumerate(zip(audio_data, labels)):
            if i % 20 == 0:
                print(f"   Processing {i+1}/{len(audio_data)}")
            
            mfcc = self.extract_fast_mfcc(audio)
            if mfcc is None:
                continue
            
            X_mfcc.append(mfcc)
            y_encoded.append(self.emotion_mapping[label])
        
        X_mfcc = np.array(X_mfcc)
        y_encoded = np.array(y_encoded)
        
        print(f"Dataset ready: {X_mfcc.shape}")
        return X_mfcc, y_encoded
    
    def build_fast_cnn(self, input_shape, num_classes):
        """
        Build a lightweight CNN for fast training
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of emotion classes
            
        Returns:
            keras.Model: Compiled CNN model
        """
        print("Building fast CNN model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Simplified convolutional layers
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_fast_lstm(self, input_shape, num_classes):
        """
        Build a lightweight LSTM for fast training
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of emotion classes
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        print("Building fast LSTM model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Single LSTM layer for speed
            layers.LSTM(64, return_sequences=False, dropout=0.2),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_fast_model(self, model, X_train, y_train, X_val, y_val, model_name, epochs=10):
        """
        Train model with fast settings
        
        Args:
            model: Keras model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_name (str): Model name
            epochs (int): Number of epochs
            
        Returns:
            keras.callbacks.History: Training history
        """
        print(f"Training {model_name} model...")
        
        # Simple callbacks for fast training
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,  # Smaller batch size for speed
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def evaluate_fast_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            model_name (str): Model name
            
        Returns:
            float: Accuracy score
        """
        print(f"Evaluating {model_name}...")
        
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   {model_name} Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def train_sklearn_models(self, X_train, y_train, X_test, y_test):
        """
        Train scikit-learn models as fallback
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            tuple: (rf_acc, svm_acc) - Accuracy scores
        """
        print("Training scikit-learn models...")
        
        # Reshape data for scikit-learn (flatten MFCC features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(X_train_flat, y_train)
        rf_pred = rf_model.predict(X_test_flat)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        # SVM
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_model.fit(X_train_flat, y_train)
        svm_pred = svm_model.predict(X_test_flat)
        svm_acc = accuracy_score(y_test, svm_pred)
        
        print(f"   Random Forest Accuracy: {rf_acc:.4f}")
        print(f"   SVM Accuracy: {svm_acc:.4f}")
        
        # Save models and results
        try:
            import joblib
            import os
            from datetime import datetime
            
            # Create results directories if they don't exist
            os.makedirs('results/models', exist_ok=True)
            os.makedirs('results/metrics', exist_ok=True)
            os.makedirs('results/logs', exist_ok=True)
            
            # Save models
            joblib.dump(rf_model, 'results/models/random_forest_model.pkl')
            joblib.dump(svm_model, 'results/models/svm_model.pkl')
            print("   Models saved to results/models/")
            
            # Save metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'results/metrics/quick_system_results_{timestamp}.txt', 'w') as f:
                f.write("Quick Emotion Recognition System Results\n")
                f.write("=" * 40 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Random Forest Accuracy: {rf_acc:.4f}\n")
                f.write(f"SVM Accuracy: {svm_acc:.4f}\n")
                f.write(f"Best Model: {'Random Forest' if rf_acc > svm_acc else 'SVM'}\n")
                f.write(f"Dataset Size: {X_train.shape[0] + X_test.shape[0]} samples\n")
                f.write(f"Training Samples: {X_train.shape[0]}\n")
                f.write(f"Test Samples: {X_test.shape[0]}\n")
            
            # Save execution log
            with open(f'results/logs/quick_system_log_{timestamp}.txt', 'w') as f:
                f.write("Quick System Execution Log\n")
                f.write("=" * 30 + "\n")
                f.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: Completed Successfully\n")
                f.write(f"Models Trained: Random Forest, SVM\n")
                f.write(f"Best Accuracy: {max(rf_acc, svm_acc):.4f}\n")
            
            print("   Results saved to results/metrics/ and results/logs/")
            
        except Exception as e:
            print(f"   Could not save results: {e}")
        
        return rf_acc, svm_acc

def main():
    """
    Main function - Fast execution
    
    This function runs the quick emotion recognition system
    optimized for speed and demonstration purposes.
    """
    print("Fast Emotion Recognition from Speech")
    print("=" * 50)
    print("TASK 2: Emotion Recognition from Speech")
    print("FAST VERSION - Optimized for speed")
    print("=" * 50)

    # Initialize system
    system = FastEmotionRecognition()
    
    # Create dataset
    print("Creating dataset...")
    audio_data, labels = system.create_fast_dataset(samples_per_emotion=25)
    
    # Prepare dataset
    print("Preparing dataset...")
    X_mfcc, y_encoded = system.prepare_fast_dataset(audio_data, labels)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_mfcc, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Dataset split:")
    print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    num_classes = len(np.unique(y_train))

    if TENSORFLOW_AVAILABLE:
        # Deep learning models
        print("Using deep learning models...")
        
        # Prepare data for different architectures
        X_train_cnn = X_train.reshape(X_train.shape[0], 20, 87, 1)
        X_val_cnn = X_val.reshape(X_val.shape[0], 20, 87, 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], 20, 87, 1)
        X_train_lstm = X_train.reshape(X_train.shape[0], 87, 20)
        X_val_lstm = X_val.reshape(X_val.shape[0], 87, 20)
        X_test_lstm = X_test.reshape(X_test.shape[0], 87, 20)

        # Train CNN
        cnn_model = system.build_fast_cnn(input_shape=(20, 87, 1), num_classes=num_classes)
        cnn_history = system.train_fast_model(cnn_model, X_train_cnn, y_train, X_val_cnn, y_val, "CNN", epochs=10)

        # Train LSTM
        lstm_model = system.build_fast_lstm(input_shape=(87, 20), num_classes=num_classes)
        lstm_history = system.train_fast_model(lstm_model, X_train_lstm, y_train, X_val_lstm, y_val, "LSTM", epochs=10)

        # Evaluate models
        system.evaluate_fast_model(cnn_model, X_test_cnn, y_test, "CNN")
        system.evaluate_fast_model(lstm_model, X_test_lstm, y_test, "LSTM")
    else:
        # Scikit-learn fallback
        print("Using scikit-learn models...")
        system.train_sklearn_models(X_train, y_train, X_test, y_test)

    print("Task 2 Complete: Fast Emotion Recognition")
    print("   • MFCC features extracted")
    print("   • Models trained")
    print("   • Fast execution achieved")

if __name__ == "__main__":
    main()
