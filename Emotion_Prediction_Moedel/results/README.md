# Results Folder

This folder contains all outputs, models, and results from the Emotion Recognition system.

## Folder Structure

```
results/
├── models/          # Trained model files (.h5, .pkl, .joblib)
├── plots/           # Training plots, confusion matrices, visualizations
├── metrics/         # Performance metrics, evaluation reports
├── logs/            # Training logs and execution history
└── README.md        # This file
```

## What Gets Saved Here

### models/
- **Trained Models**: CNN, LSTM, Random Forest, SVM models
- **Model Weights**: Saved model parameters
- **Model Metadata**: Configuration and training info

### plots/
- **Training History**: Accuracy and loss plots
- **Confusion Matrices**: Model performance visualization
- **MFCC Visualizations**: Audio feature plots
- **ROC Curves**: Model evaluation curves

### metrics/
- **Performance Reports**: Accuracy, precision, recall, F1-score
- **Classification Reports**: Detailed model evaluation
- **Comparison Tables**: Model performance comparison
- **Evaluation Results**: Test set performance metrics

### logs/
- **Training Logs**: Detailed training progress
- **Execution Logs**: System run history
- **Error Logs**: Debugging information
- **Performance Logs**: Timing and resource usage

## Usage

When you run the emotion recognition scripts, results will be automatically saved here:

```bash
# Quick system results
python quick_emotion_recognition.py
# Results saved to: results/models/, results/metrics/

# Main system results  
python main_emotion_recognition.py
# Results saved to: results/models/, results/plots/, results/metrics/
```

## Expected Results

### Quick System Output:
- **Models**: `random_forest_model.pkl`, `svm_model.pkl`
- **Metrics**: `quick_system_results.txt`
- **Performance**: ~100% accuracy, 30-second training

### Main System Output:
- **Models**: `emotion_cnn_model.h5`, `emotion_lstm_model.h5`
- **Plots**: Training history, confusion matrices
- **Metrics**: Detailed evaluation reports
- **Performance**: High accuracy with real/synthetic data

## Viewing Results

### Check Model Performance:
```bash
# View saved models
ls results/models/

# Check metrics
cat results/metrics/*.txt

# View plots (if matplotlib backend available)
# Open results/plots/ in file explorer
```

### Recent Results:
- **Last Run**: Check `results/logs/` for execution history
- **Best Models**: Look for highest accuracy in `results/metrics/`
- **Visualizations**: Check `results/plots/` for graphs

## Performance Tracking

The system automatically tracks:
- Model accuracy and performance
- Training time and efficiency  
- Resource usage and memory
- Error rates and debugging info
- Model comparison metrics

This organized structure makes it easy to track progress and compare different model versions!
