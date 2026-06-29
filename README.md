# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

A production-style **Credit Card Fraud Detection** system built using Machine Learning to identify fraudulent transactions in highly imbalanced financial datasets.

This project follows a modular architecture with automated preprocessing, exploratory data analysis (EDA), class imbalance handling using **SMOTE**, model training, performance evaluation, threshold optimization, and visualization generation.

---

## 📌 Project Highlights

- 📊 Automated Exploratory Data Analysis (EDA)
- ⚖️ Handles extreme class imbalance using **SMOTE**
- 🌲 Random Forest Classifier
- ⚡ XGBoost Classifier
- 📈 ROC Curve & Precision-Recall Curve
- 🎯 Threshold Optimization for Business Decisions
- 📉 Confusion Matrix & Model Comparison
- 📝 Professional Logging System
- 📂 Modular Project Structure
- 💾 Automatic Model Saving
- 📊 Automatic Visualization Saving

---

# 📂 Project Structure

```
Syntecxhub_Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── logs/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── sampling.py
│   ├── models.py
│   ├── evaluation.py
│   ├── threshold_analysis.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 Features

### 📊 Exploratory Data Analysis

- Class Distribution Analysis
- Transaction Amount Distribution
- Transaction Time Analysis
- Correlation Heatmap
- Top Feature Boxplots

---

### ⚙ Data Preprocessing

- Missing Value Checking
- Feature Scaling
- Train-Test Split
- Data Cleaning

---

### ⚖ Imbalanced Data Handling

The dataset is highly imbalanced.

Implemented:

- SMOTE (Synthetic Minority Oversampling Technique)

---

### 🤖 Machine Learning Models

- Random Forest Classifier
- XGBoost Classifier

---

### 📈 Model Evaluation

Performance metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Average Precision Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

### 🎯 Threshold Analysis

Instead of relying only on the default threshold (0.5), this project analyzes different probability thresholds to determine the best trade-off between:

- Precision
- Recall
- F1 Score

This is particularly useful for real-world fraud detection systems.

---

# 📊 Generated Outputs

The pipeline automatically generates:

- Class Distribution Plot
- Transaction Amount Distribution
- Time Distribution
- Correlation Heatmap
- Feature Boxplots
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Model Comparison
- Threshold Trade-off Graph

All outputs are stored inside the **outputs/** directory.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)
- Matplotlib
- Seaborn
- Joblib
- Loguru

---

# 📥 Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

**Download the dataset:**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded file inside:

```
data/
└── creditcard.csv
```

> **Note:** The dataset is excluded from this repository because it exceeds GitHub's file size limit.

---

# ▶ Installation

Clone the repository

```bash
git clone https://github.com/ganesh9076/Syntecxhub_Credit_card_fraud_detection_task3.git
```

Move into the project

```bash
cd Syntecxhub_Credit_card_fraud_detection_task3
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

# 📌 Workflow

```
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
Exploratory Data Analysis
    │
    ▼
SMOTE Sampling
    │
    ▼
Model Training
    │
    ├── Random Forest
    └── XGBoost
          │
          ▼
Model Evaluation
          │
          ▼
Threshold Optimization
          │
          ▼
Visualization & Model Saving
```

---

# 📈 Future Improvements

- Hyperparameter Optimization
- Explainable AI using SHAP
- Streamlit Web Application
- Real-Time Fraud Detection API
- Docker Deployment
- CI/CD Pipeline

---

# 👨‍💻 Author

**Ganesh Palav**

B.Tech Computer Science & Engineering

Passionate about Machine Learning, Artificial Intelligence, Data Science, and Software Development.

---

# 🤝 Acknowledgements

This project was developed as part of the **SyntecxHub Machine Learning Internship**, focusing on real-world fraud detection using modern machine learning techniques.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
