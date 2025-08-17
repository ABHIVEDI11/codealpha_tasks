# 🎥 Quick Video Recording Guide

## **Essential Commands for Your Video**

### **1. Setup & Testing (2 min)**
```bash
pip install -r requirements.txt
python test_pipeline.py
```

### **2. Train Heart Disease Model (3 min)**
```bash
python disease_prediction_pipeline.py --data heart.csv --target target --outdir runs/heart --smote --calibrate
```

### **3. Train Diabetes Model (2 min)**
```bash
python disease_prediction_pipeline.py --data diabetes.csv --target Outcome --outdir runs/diabetes --smote
```

### **4. Create Sample Patients (30 sec)**
```bash
python create_sample_patients.py
```

### **5. Video Demonstration (4 min)**
```bash
python video_demo_predictions.py
```

### **6. Individual Predictions (2 min)**
```bash
python predict_new_patients.py
```

---

## **🎯 Key Results to Highlight**

### **Model Performance:**
- **Heart Disease:** 100% accuracy (Random Forest)
- **Diabetes:** 74% accuracy (SVM)
- **Professional Metrics:** ROC AUC, F1-Score

### **Patient Predictions:**
- **8 Heart Disease Patients:** 3 High Risk, 2 Medium, 3 Low Risk
- **8 Diabetes Patients:** 3 High Risk, 2 Medium, 3 Low Risk
- **Probability Scores:** Available for heart disease model

### **Professional Features:**
- Clean, formatted output
- Color-coded predictions (🟢/🔴)
- Risk level classifications
- Multiple ML algorithms

---

## **📱 Video Platforms**

### **YouTube:**
- Title: "Disease Prediction from Medical Data using Machine Learning"
- Description: Include GitHub link and key features
- Tags: #MachineLearning #Healthcare #Python #DataScience

### **LinkedIn:**
- Professional tone
- Focus on healthcare applications
- Include GitHub repository link

### **GitHub:**
- Add video link to README
- Update project description
- Include video in project showcase

---

## **🎬 Recording Tips**

1. **Start with:** "Welcome to my Disease Prediction Model demonstration"
2. **Explain each step:** What each command does
3. **Show results:** Highlight the accuracy and predictions
4. **End with:** "This demonstrates the potential of ML in healthcare"

**Total Video Time: 12-15 minutes**
**Perfect for:** Portfolio, job applications, GitHub showcase

**Good luck! 🚀**
