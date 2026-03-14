# 🫀 CVDStack: AI-Driven Cardiovascular Disease Prediction with Stacking Generative AI

**Live App:** [cvdstack.streamlit.app](https://cvdstack.streamlit.app)  
**Repository:** [GitHub – HowardHNguyen/cvdstack](https://github.com/HowardHNguyen/cvdstack/tree/main)  
**Creator:** Dr. Howard Nguyen, PhD — *Data Science & AI, Healthcare Analytics*

## 💡 Overview

**CVDStack** is a full-stack **Generative AI + Machine Learning platform** that predicts cardiovascular disease risk with medical-grade accuracy.  
Built as part of my doctoral dissertation — *“Advancing Heart Failure Prediction: A Comparative Study of Traditional Machine Learning, Neural Networks, and Stacking Generative AI Models”* — this project demonstrates how **stacked ensemble AI** can outperform both classical ML and single deep-learning models in real-world healthcare data.

The model integrates **Generative Adversarial Networks (GANs)**, **Random Forest (RF)**, **Gradient Boosting (GBM)**, and **Convolutional Neural Networks (CNNs)** into a unified architecture, improving minority-class representation and predictive reliability.

## 🧠 Motivation

Heart disease remains the **leading global cause of death**, often driven by silent risk factors hidden in routine clinical data.  
Traditional models capture linear relationships but miss subtle, nonlinear interactions.  
**CVDStack** bridges this gap by generating balanced synthetic data (via GANs) and stacking multiple AI learners to achieve deeper, more interpretable insights.

## ⚙️ Features

- 🧬 **Stacking Generative AI Model:** Combines GAN + RF + GBM + CNN to enhance learning from imbalanced datasets.  
- 📊 **Predictive Dashboard:** Real-time ROC AUC, accuracy, and precision/recall summaries.  
- 🩺 **Explainable AI:** SHAP-based feature importance to identify clinical drivers (e.g., BMI, sysBP, totChol).  
- 🧩 **Data Balancing:** SMOTE and CTGAN integration for minority-class synthesis.  
- 💾 **Scalable Pipeline:** Handles datasets from 303 to 400K records seamlessly.  
- 🌐 **Deployment:** Live Streamlit app connected to GitHub root-level Python files (no sub-dirs).

## 🧩 Methodology

1. **Data Engineering**  
   - Pre-processed demographic & clinical variables (sex, age, BP, cholesterol, glucose, BMI, smoking).  
   - Handled class imbalance using SMOTE and GAN-based data augmentation.

2. **Model Training**  
   - Trained Random Forest, GBM, XGBoost, CNN, and Generative AI models individually.  
   - Integrated them via a Stacking Classifier meta-learner (Logistic Regression).  
   - Calibrated predictions using Isotonic Regression to ensure probability reliability.

3. **Evaluation & Validation**  
   - Cross-validation AUC ≈ **0.99**, accuracy ≈ **0.97** on 400K records.  
   - Comparative experiments vs. Logistic Regression, RF, GBM, and CNN confirmed superiority.  
   - Avoided overfitting via permutation testing and hold-out validation.

4. **Deployment & Interpretation**  
   - Streamlit UI built with Plotly and matplotlib for visual diagnostics.  
   - Displays model predictions, feature importance, and interactive threshold adjustment.

## 📈 Results Snapshot

| Model | Accuracy | ROC AUC | Key Finding |
|:------|:---------:|:-------:|-------------|
| Logistic Regression | 0.59 – 0.68 | 0.64 – 0.73 | Limited by non-linear interactions |
| Random Forest | 0.70 – 0.81 | 0.63 – 0.90 | Strong but sensitive to imbalance |
| GBM / XGBoost | 0.72 – 0.86 | 0.62 – 0.97 | High variance without balancing |
| CNN / RNN | 0.80 – 0.95 | 0.80 – 0.99 | Excellent pattern learning |
| **Stacking Gen AI (ours)** | **0.97 +** | **0.99 +** | Best overall generalization |

## 🧰 Tech Stack

| Layer | Tools / Libraries |
|-------|-------------------|
| Language | Python 3.12 + Google Colab (T4 GPU) |
| ML / AI | scikit-learn · XGBoost · LightGBM · TensorFlow/Keras · PyTorch · CTGAN · SMOTE |
| Visualization | Plotly · matplotlib · Streamlit UI |
| Deployment | Streamlit Cloud + GitHub (main branch root files) |
| Environment Control | `scikit-learn==1.6.1` · `joblib==1.4.2` · `lightgbm==4.5.0` · `xgboost==2.1.1` |


## 📊 Live Dashboard Highlights

### 🧩 Model Summary
Displays performance metrics for each algorithm — including **AUC**, **Accuracy**, **Precision**, and **Recall** — enabling direct comparison of model reliability.

### 🔍 Feature Importance
Visualizes which clinical variables (such as **BMI**, **sysBP**, **glucose**, and **totChol**) contribute most to prediction outcomes, providing explainability for clinicians and researchers.

### ⚖️ Threshold Tuning
Allows adjustment of the decision threshold to fine-tune the balance between **Sensitivity (Recall)** and **Specificity (Precision)** for different clinical priorities.

### 📂 Data Upload
Supports drag-and-drop of new CSV files, allowing external researchers or clinicians to test their own patient data and instantly visualize predictions.

### 💬 Interpretability Layer
Includes SHAP explainers and calibration metrics to translate raw model outputs into transparent, actionable insights.

## 🧬 Research Significance

> This project demonstrates that **Stacking Generative AI** can achieve medical-grade predictive accuracy and enhanced fairness for underrepresented patient groups.

It contributes to the next generation of **AI Health Diagnostics** and aligns with the vision of **AICardioHealth Inc.**, a startup dedicated to advancing AI-driven cardiovascular prevention.

By integrating Generative AI with ensemble learning, **CVDStack** represents a paradigm shift in how healthcare systems can predict, explain, and prevent heart failure through data science.

## 🏁 Summary

**CVDStack = Generative AI + Stacked Learning + Explainable Healthcare.**  
It moves beyond prediction to personalized intervention — turning clinical data into life-saving insights.

> *Predict early. Explain clearly. Act precisely.*

## 🧾 License

**Copyright © 2025 Howard Nguyen**  
*(MaxAIS · AICardioHealth)*

Permission is hereby granted **with explicit written approval from Howard Nguyen** to use, copy, modify, and distribute this software and its associated documentation files.  
Unauthorized reproduction, redistribution, or modification without prior approval is strictly prohibited.

For commercial use, research collaboration, or licensing inquiries, please contact **info@howardnguyen.com**.

## 💬 Contact

📧 **Email:** info@howardnguyen.com  
🌐 **Website:** [www.maxais.com](https://www.maxais.com)  
🔗 **LinkedIn:** [Howard H. Nguyen](https://www.linkedin.com/in/howardhnguyen/)

## ⭐ Acknowledgments

Special thanks to **Harrisburg University** faculty and research mentors for academic guidance.  
This project also draws inspiration from global healthcare AI initiatives advancing cardiovascular prediction and early intervention.
