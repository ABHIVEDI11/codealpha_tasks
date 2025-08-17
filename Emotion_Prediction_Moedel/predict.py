import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import librosa.display

def load_emotion_model():
    """Load the trained emotion recognition model and preprocessing data"""
    try:
        model = load_model('emotion_model.h5')
        with open('preprocessing.pkl', 'rb') as f:
            preprocessing_data = pickle.load(f)
        return model, preprocessing_data
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please run train.py first. {e}")
        return None, None

def extract_mfcc_from_audio(audio_path, sample_rate=22050, n_mfcc=13, target_length=100):
    """Extract MFCC features from an audio file"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        if mfccs.shape[1] < target_length:
            # Pad with zeros
            padding = np.zeros((n_mfcc, target_length - mfccs.shape[1]))
            mfccs = np.hstack([mfccs, padding])
        else:
            # Truncate
            mfccs = mfccs[:, :target_length]
        
        # Reshape for model input (add batch and channel dimensions)
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        
        return mfccs, audio, sr
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None, None, None

def predict_emotion_from_audio(audio_path):
    """Predict emotion from an audio file"""
    model, preprocessing_data = load_emotion_model()
    if model is None:
        return None
    
    # Extract features
    features, audio, sr = extract_mfcc_from_audio(
        audio_path, 
        sample_rate=preprocessing_data['sample_rate'],
        n_mfcc=preprocessing_data['n_mfcc'],
        target_length=preprocessing_data['target_length']
    )
    
    if features is None:
        return None
    
    # Make prediction
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions[0])
    predicted_emotion = preprocessing_data['label_encoder'].inverse_transform([predicted_class])[0]
    
    # Get probabilities for all emotions
    emotion_probabilities = {}
    for i, emotion in enumerate(preprocessing_data['label_encoder'].classes_):
        emotion_probabilities[emotion] = float(predictions[0][i])
    
    return {
        'predicted_emotion': predicted_emotion,
        'confidence': float(np.max(predictions[0])),
        'probabilities': emotion_probabilities,
        'audio': audio,
        'sample_rate': sr
    }

def predict_emotion_from_array(audio_array, sample_rate=22050):
    """Predict emotion from audio array"""
    model, preprocessing_data = load_emotion_model()
    if model is None:
        return None
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=preprocessing_data['n_mfcc'])
    
    # Pad or truncate to fixed length
    target_length = preprocessing_data['target_length']
    if mfccs.shape[1] < target_length:
        padding = np.zeros((preprocessing_data['n_mfcc'], target_length - mfccs.shape[1]))
        mfccs = np.hstack([mfccs, padding])
    else:
        mfccs = mfccs[:, :target_length]
    
    # Reshape for model input
    mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
    
    # Make prediction
    predictions = model.predict(mfccs)
    predicted_class = np.argmax(predictions[0])
    predicted_emotion = preprocessing_data['label_encoder'].inverse_transform([predicted_class])[0]
    
    # Get probabilities
    emotion_probabilities = {}
    for i, emotion in enumerate(preprocessing_data['label_encoder'].classes_):
        emotion_probabilities[emotion] = float(predictions[0][i])
    
    return {
        'predicted_emotion': predicted_emotion,
        'confidence': float(np.max(predictions[0])),
        'probabilities': emotion_probabilities
    }

def visualize_audio_and_prediction(audio_path, result):
    """Create visualization of audio waveform and prediction results"""
    if result is None or result['audio'] is None:
        return
    
    plt.figure(figsize=(15, 10))
    
    # Audio waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(result['audio'], sr=result['sample_rate'])
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # MFCC features
    plt.subplot(3, 1, 2)
    mfccs = librosa.feature.mfcc(y=result['audio'], sr=result['sample_rate'], n_mfcc=13)
    librosa.display.specshow(mfccs, sr=result['sample_rate'], x_axis='time')
    plt.title('MFCC Features')
    plt.colorbar(format='%+2.0f dB')
    
    # Emotion probabilities
    plt.subplot(3, 1, 3)
    emotions = list(result['probabilities'].keys())
    probabilities = list(result['probabilities'].values())
    
    bars = plt.bar(emotions, probabilities, color='skyblue')
    plt.title('Emotion Probabilities')
    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # Highlight the predicted emotion
    predicted_idx = emotions.index(result['predicted_emotion'])
    bars[predicted_idx].set_color('red')
    
    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/emotion_prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def interactive_prediction():
    """Interactive prediction interface"""
    print("=== Emotion Recognition from Speech ===")
    print("Enter the path to an audio file (WAV, MP3, etc.):")
    
    audio_path = input("Audio file path: ").strip()
    
    if not audio_path:
        print("No file path provided.")
        return
    
    print(f"\nProcessing audio file: {audio_path}")
    result = predict_emotion_from_audio(audio_path)
    
    if result:
        print("\n=== Prediction Results ===")
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll Emotion Probabilities:")
        for emotion, prob in result['probabilities'].items():
            print(f"  {emotion}: {prob:.3f}")
        
        # Create visualization
        print("\nCreating visualization...")
        visualize_audio_and_prediction(audio_path, result)
    else:
        print("Failed to process audio file. Please check the file path and format.")

def batch_prediction(audio_files):
    """Process multiple audio files"""
    results = []
    
    for audio_path in audio_files:
        print(f"Processing: {audio_path}")
        result = predict_emotion_from_audio(audio_path)
        
        if result:
            results.append({
                'file': audio_path,
                'predicted_emotion': result['predicted_emotion'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            })
        else:
            results.append({
                'file': audio_path,
                'predicted_emotion': 'ERROR',
                'confidence': 0.0,
                'probabilities': {}
            })
    
    return results

if __name__ == "__main__":
    print("Emotion Recognition from Speech")
    print("1. Interactive prediction (single file)")
    print("2. Batch prediction (multiple files)")
    
    choice = input("Choose option (1 or 2): ")
    
    if choice == "1":
        interactive_prediction()
    elif choice == "2":
        print("Enter audio file paths (one per line, press Enter twice when done):")
        files = []
        while True:
            file_path = input().strip()
            if not file_path:
                break
            files.append(file_path)
        
        if files:
            results = batch_prediction(files)
            
            # Save results
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv('results/batch_emotion_predictions.csv', index=False)
            print(f"\nBatch results saved to 'results/batch_emotion_predictions.csv'")
            
            # Print summary
            print("\n=== Batch Prediction Summary ===")
            for result in results:
                print(f"{result['file']}: {result['predicted_emotion']} ({result['confidence']:.2%})")
        else:
            print("No files provided.")
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
