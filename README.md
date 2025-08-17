# Machine Learning Projects Collection

A collection of three minimalistic but functional machine learning projects for different domains.

## Projects Overview

### 1. Credit Scoring Model
**Path**: `Credit_Scoring_Model/`
- **Objective**: Predict creditworthiness from financial data
- **Algorithms**: Logistic Regression, Random Forest
- **Features**: Income, debt, payment history, credit utilization, age, employment length
- **Output**: `credit_model.pkl`

### 2. Emotion Recognition from Speech
**Path**: `Emotion_Prediction_Moedel/`
- **Objective**: Recognize emotions from speech audio
- **Algorithm**: CNN with MFCC features
- **Emotions**: Happy, sad, angry, neutral, fearful, surprised
- **Output**: `emotion_model.h5`

### 3. Disease Prediction Model
**Path**: `Disease_Prediction_Model/`
- **Objective**: Predict diseases from medical data
- **Algorithms**: Logistic Regression, Random Forest, SVM
- **Diseases**: Heart Disease, Diabetes, Breast Cancer
- **Output**: `*_model.pkl` files

## Quick Start

### Prerequisites
- Python 3.7+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ABHIVEDI11/codealpha_tasks.git
cd codealpha_tasks
```

2. Install dependencies for each project:
```bash
# Credit Scoring
cd Credit_Scoring_Model
pip install -r requirements.txt

# Emotion Recognition
cd ../Emotion_Prediction_Moedel
pip install -r requirements.txt

# Disease Prediction
cd ../Disease_Prediction_Model
pip install -r requirements.txt
```

### Running the Projects

#### Credit Scoring Model
```bash
cd Credit_Scoring_Model
python train.py    # Train the model
python predict.py  # Make predictions
```

#### Emotion Recognition
```bash
cd Emotion_Prediction_Moedel
python train.py    # Train the model
python predict.py  # Make predictions
```

#### Disease Prediction
```bash
cd Disease_Prediction_Model
python train.py    # Train the model
python predict.py  # Make predictions
```

## Project Structure

```
CODE_BASICS/
├── Credit_Scoring_Model/
│   ├── train.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── README.md
│   └── result/
├── Emotion_Prediction_Moedel/
│   ├── train.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── README.md
│   └── results/
└── Disease_Prediction_Model/
    ├── train.py
    ├── predict.py
    ├── requirements.txt
    ├── README.md
    ├── breast_cancer.csv
    └── runs/
```

## Features

### Common Features Across All Projects
- ✅ **Minimalistic but functional** implementation
- ✅ **Easy to run** with just `python train.py`
- ✅ **Comprehensive evaluation** metrics
- ✅ **Interactive and batch prediction** capabilities
- ✅ **Visualization** of results
- ✅ **Model persistence** (saved models)
- ✅ **Synthetic data generation** for demonstration

### Credit Scoring Model
- ✅ Logistic Regression and Random Forest
- ✅ ROC-AUC, Precision, Recall, F1-score evaluation
- ✅ Feature importance analysis
- ✅ Interactive prediction interface

### Emotion Recognition
- ✅ CNN-based deep learning model
- ✅ MFCC feature extraction
- ✅ 6 emotion classes
- ✅ Audio visualization
- ✅ TensorFlow/Keras implementation

### Disease Prediction
- ✅ Multiple diseases (Heart, Diabetes, Breast Cancer)
- ✅ Multiple algorithms (LR, RF, SVM)
- ✅ Comprehensive medical feature sets
- ✅ Batch processing capabilities

## Model Performance

All models include:
- **Training and validation metrics**
- **ROC curves and AUC scores**
- **Confusion matrices**
- **Feature importance analysis**
- **Classification reports**

## Output Files

### Credit Scoring
- `credit_model.pkl` - Trained model
- `result/credit_model_results.png` - Visualizations
- `credit_predictions.csv` - Batch results

### Emotion Recognition
- `emotion_model.h5` - Trained Keras model
- `preprocessing.pkl` - Preprocessing parameters
- `results/emotion_model_results.png` - Training plots
- `results/emotion_prediction_visualization.png` - Audio analysis

### Disease Prediction
- `heart_model.pkl`, `diabetes_model.pkl`, `breast_cancer_model.pkl` - Models
- `disease_models_info.pkl` - Combined info
- `runs/*/disease_model_results.png` - Analysis plots
- `*_predictions.csv` - Batch results

## Usage Examples

### Credit Scoring
```python
# Interactive prediction
python predict.py
# Enter: Income=50000, Debt=15000, Payment_History=85, etc.
```

### Emotion Recognition
```python
# Interactive prediction
python predict.py
# Enter: path/to/audio_file.wav
```

### Disease Prediction
```python
# Interactive prediction
python predict.py
# Choose: 1=Heart, 2=Diabetes, 3=Breast Cancer
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Disclaimer

⚠️ **Medical Disclaimer**: The disease prediction models use synthetic data and should NOT be used for actual medical diagnosis. Always consult healthcare professionals for medical decisions.

⚠️ **Credit Scoring Disclaimer**: The credit scoring model is for demonstration purposes only and should not be used for actual credit decisions.

## Contact

For questions or issues, please open an issue on GitHub.
