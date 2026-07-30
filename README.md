# 🚗 Indian Pre-Owned Car Price Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-FF6600?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Machine Learning Capstone Project** for predicting used car prices in the Indian market using regression models and an interactive Streamlit application.

---

## 📌 Project Overview

**Indian Pre-Owned Car Price Prediction** is a machine learning project designed to estimate the selling price of used cars in India based on vehicle, location, ownership, engine, and usage-related features.

The project covers the complete machine learning lifecycle, including Exploratory Data Analysis, missing value handling, feature engineering, preprocessing pipelines, model training, hyperparameter tuning, model evaluation, model serialization, and deployment through a Streamlit web application.

This project is suitable for portfolio presentation because it demonstrates practical machine learning skills across data analysis, modeling, evaluation, and user-facing deployment.

---

## ✨ Features

- Complete Exploratory Data Analysis
- Missing value handling
- Feature engineering
- Data preprocessing using `Pipeline` and `ColumnTransformer`
- Regression model training
- Hyperparameter tuning using `RandomizedSearchCV`
- Cross-validation
- Model comparison using RMSE, MAE, and R² Score
- Feature importance visualization
- Model serialization using Joblib
- Streamlit web application
- Single car price prediction
- Batch CSV prediction
- Interactive dashboard
- Download prediction CSV

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Model Saving | Joblib |
| Web Application | Streamlit |
| Notebook Environment | Jupyter Notebook |

---

## 🔁 Machine Learning Workflow

```text
Raw Dataset
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Preprocessing Pipeline
    ↓
Model Training
    ↓
Model Comparison
    ↓
Hyperparameter Tuning
    ↓
Model Evaluation
    ↓
Model Serialization
    ↓
Streamlit Deployment
```

### Workflow Summary

| Step | Description |
|---|---|
| Data Cleaning | Handles missing values, inconsistent formats, and noisy records. |
| EDA | Studies feature distributions, relationships, outliers, and price trends. |
| Feature Engineering | Creates useful derived features for better prediction performance. |
| Preprocessing | Uses Scikit-learn pipelines for numerical and categorical transformations. |
| Model Training | Trains multiple regression algorithms for comparison. |
| Evaluation | Compares models using RMSE, MAE, and R² Score. |
| Deployment | Serves predictions through an interactive Streamlit application. |

---

## 🤖 Models Implemented

| Model | Type |
|---|---|
| Linear Regression | Baseline regression model |
| Random Forest Regressor | Ensemble bagging model |
| Gradient Boosting Regressor | Boosting-based regression model |
| XGBoost Regressor | Optimized gradient boosting model |
| LightGBM Regressor | Fast gradient boosting framework |
| CatBoost Regressor | Gradient boosting model for categorical features |

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE | Measures the square root of average squared prediction error. Lower is better. |
| MAE | Measures the average absolute difference between actual and predicted prices. Lower is better. |
| R² Score | Measures how well the model explains variance in car prices. Higher is better. |

---

## 🧾 Dataset Information

The project uses car listing datasets provided as CSV files.

| File | Purpose |
|---|---|
| `Cap_Training_Data_2025.csv` | Training dataset used for EDA, preprocessing, and model training |
| `Cap_Test_Data_2025.csv` | Test dataset used for final prediction generation |
| `Cap_Sample_Submission_2025.csv` | Sample submission format |

### Example Feature Groups

- Car maker and model
- Location
- Distance driven
- Owner type
- Manufacturing year
- Engine displacement
- Engine power
- Body type
- Transmission type
- Fuel type
- Door count
- Seat count
- Audit rating

---

## 🧹 Preprocessing Steps

- Removed or handled missing values
- Cleaned inconsistent column names
- Processed numerical and categorical columns separately
- Applied imputation where required
- Encoded categorical variables
- Scaled numerical features where applicable
- Built reusable preprocessing workflow using `Pipeline` and `ColumnTransformer`
- Generated derived age-related features for model prediction

---

## 📁 Project Structure

```text
Indian-Car-Price-Prediction/
├── app/
│   └── app.py
├── data/
│   ├── Cap_Sample_Submission_2025.csv
│   ├── Cap_Test_Data_2025.csv
│   └── Cap_Training_Data_2025.csv
├── models/
│   ├── best_model.pkl
│   ├── feature_names.pkl
│   └── preprocessor.pkl
├── notebooks/
│   ├── Car_Price_Prediction.ipynb
│   └── catboost_info/
├── outputs/
│   └── submission.csv
├── .venv/
├── requirement.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Somyaranjan-Jena/Indian-Car-Price-Prediction.git
cd Indian-Car-Price-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 4. Install Requirements

```bash
pip install -r requirement.txt
```

---

## 📦 Required Libraries

```text
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
streamlit
plotly
joblib
matplotlib
seaborn
```

---

## 📓 How to Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook:

```text
notebooks/Car_Price_Prediction.ipynb
```

Run all cells to perform EDA, preprocessing, model training, evaluation, and model saving.

---

## 🌐 How to Run the Streamlit App

Launch the Streamlit application:

```bash
streamlit run app/app.py
```

The application includes:

- Single car price prediction
- Batch CSV prediction
- Interactive dashboard
- Downloadable prediction output

---

## 🖼️ Screenshots

### Home Page

![Home Page](images/home.png)

### Dashboard

![Dashboard](images/dashboard.png)

### Batch Prediction

![Batch Prediction](images/batch_prediction.png)

---

## 📈 Model Performance

The models were evaluated using RMSE, MAE, and R² Score.

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 1.1033 | 0.9272 | 0.5602 |
| Random Forest Regressor | 1.1207 | 0.9133 | 0.5462 |
| Gradient Boosting Regressor | 1.0612 | 0.9183 | 0.5931 |
| XGBoost Regressor | 1.1671 | 0.9960 | 0.5079 |
| LightGBM Regressor | 1.1895 | 0.9764 | 0.4888 |
| CatBoost Regressor | 1.1271 | 0.9457 | 0.5410 |

<details>
<summary>📌 Notes on Performance</summary>

Based on R² Score and RMSE, the Gradient Boosting Regressor achieved the strongest performance among the evaluated models.

</details>

---

## 🚀 Future Improvements

- Add live deployment on Streamlit Community Cloud
- Improve UI styling and user experience
- Add model explainability using SHAP
- Add automated data validation for uploaded CSV files
- Store predictions and user inputs in a database
- Add more advanced feature engineering
- Build an API endpoint for predictions
- Add CI/CD workflow for testing and deployment

---

## 🙏 Acknowledgements

- Python open-source ecosystem
- Scikit-learn documentation
- XGBoost, LightGBM, and CatBoost communities
- Streamlit for simple and fast ML app deployment
- Dataset providers and project mentors

---

## 👨‍💻 Author

| Name | GitHub |
|---|---|
| Somyaranjan Jena | [Somyaranjan-Jena](https://github.com/Somyaranjan-Jena) |

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project with proper attribution.
