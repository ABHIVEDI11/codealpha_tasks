# Emotion Recognition from Speech

A deep learning model to recognize emotions from speech audio using MFCC features and CNN.

## Features

- **CNN-based** emotion recognition model
- **MFCC feature extraction** from audio
- **6 emotion classes**: happy, sad, angry, neutral, fearful, surprised
- **Synthetic data generation** for demonstration
- **Interactive and batch prediction** capabilities
- **Audio visualization** with waveform and MFCC features

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py
```

This will:
- Generate synthetic audio data for 6 emotions
- Extract MFCC features from audio
- Train a CNN model for emotion classification
- Save the model as `emotion_model.h5`
- Create visualization plots in the `results/` directory

### Making Predictions

```bash
python predict.py
```

Choose from two options:
1. **Interactive prediction**: Process a single audio file
2. **Batch prediction**: Process multiple audio files

### Supported Audio Formats

- WAV, MP3, FLAC, OGG, and other formats supported by librosa
- Recommended: 16kHz or 22kHz sample rate
- Duration: 3+ seconds for best results

## Model Architecture

- **Input**: MFCC features (13 coefficients × 100 time steps)
- **CNN Layers**: 3 convolutional layers with max pooling
- **Dense Layers**: 128 units with dropout
- **Output**: 6 emotion classes with softmax activation

## Output

- `emotion_model.h5`: Trained Keras model
- `preprocessing.pkl`: Label encoder and preprocessing parameters
- `results/emotion_model_results.png`: Training history and confusion matrix
- `results/emotion_prediction_visualization.png`: Audio analysis visualization
- `results/batch_emotion_predictions.csv`: Batch prediction results

## Emotion Classes

1. **Happy**: High frequencies, energetic patterns
2. **Sad**: Lower frequencies, slower patterns
3. **Angry**: High amplitude, harsh frequencies
4. **Neutral**: Balanced frequency patterns
5. **Fearful**: Variable frequencies, trembling patterns
6. **Surprised**: Sudden high frequencies

## Performance

The model provides:
- Training and validation accuracy curves
- Confusion matrix for emotion classification
- Individual emotion probabilities
- Audio waveform and MFCC feature visualization
