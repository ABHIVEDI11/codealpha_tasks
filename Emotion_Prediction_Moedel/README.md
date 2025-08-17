# Emotion Recognition from Speech

A deep learning-based system for recognizing human emotions from speech audio using MFCC features and CNN/LSTM models.

## Objective

Recognize human emotions (happy, sad, angry, neutral) from speech audio using deep learning and speech signal processing techniques.

## Key Features

- **MFCC Feature Extraction**: Mel-Frequency Cepstral Coefficients for audio analysis
- **Deep Learning Models**: CNN and LSTM architectures for emotion classification
- **Online Dataset Loading**: Support for RAVDESS dataset via Deep Lake
- **Synthetic Data Generation**: Fallback synthetic audio for testing
- **Fast Processing**: Optimized for quick training and inference

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Main System (Complete Pipeline)
```bash
python main_emotion_recognition.py
```

#### Quick System (Fast Training)
```bash
python quick_emotion_recognition.py
```

## Results

All model outputs, training results, and performance metrics are automatically saved to the `results/` folder:

- **Models**: Trained model files (`.h5`, `.pkl`)
- **Plots**: Training history and performance visualizations
- **Metrics**: Detailed performance reports and accuracy scores
- **Logs**: Execution history and debugging information

### View Results
```bash
# Check saved models
ls results/models/

# View performance metrics
cat results/metrics/*.txt

# Check execution logs
ls results/logs/
```

## Project Structure

```
Emotion/
├── main_emotion_recognition.py    # Complete emotion recognition system
├── quick_emotion_recognition.py   # Fast emotion recognition system
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore rules
└── results/                      # Model outputs and results
    ├── models/                   # Trained model files
    ├── plots/                    # Training visualizations
    ├── metrics/                  # Performance reports
    ├── logs/                     # Execution logs
    └── README.md                 # Results documentation
```

## Technical Details

### Models
- **CNN**: Convolutional Neural Network for spectral feature analysis
- **LSTM**: Long Short-Term Memory for temporal sequence modeling

### Features
- **MFCC**: 40 coefficients extracted from audio signals
- **Audio Processing**: 22050 Hz sample rate, 3-second duration
- **Data Augmentation**: Synthetic audio generation for testing

### Datasets
- **RAVDESS**: Real-world emotional speech dataset (via Deep Lake)
- **Synthetic**: Generated audio samples for quick testing

## Performance

The system achieves:
- Fast training with early stopping
- Real-time emotion prediction
- Support for both TensorFlow and scikit-learn backends

## Requirements

- Python 3.7+
- TensorFlow 2.x (optional, falls back to scikit-learn)
- librosa for audio processing
- scikit-learn for machine learning
- matplotlib for visualization

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!
