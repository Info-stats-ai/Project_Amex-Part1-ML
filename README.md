
# 🏦 American Express Default Prediction

## 📌 Project Overview
This project tackles the **American Express Default Prediction Challenge** on Kaggle. The goal is to build a machine learning model that predicts whether a customer will default on their payments within 120 days after their latest statement.  

Credit default prediction is critical for **financial institutions** to manage risk and improve lending decisions.  

---

## 📂 Dataset
- **Source:** [Kaggle - Amex Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction/data)  
- **Features:**  
  - **Delinquency (D\_*)** – payment delays  
  - **Spend (S\_*)** – spending patterns  
  - **Payment (P\_*)** – repayment behavior  
  - **Balance (B\_*)** – account balances  
  - **Risk (R\_*)** – risk factors  
- **Target Variable:**  
  - `1` → Customer defaulted  
  - `0` → Customer paid on time  

---

## 🔑 Steps Involved
1. **Data Cleaning**  
   - Duplicate removal, error handling, validation  
2. **Preprocessing**  
   - Handling missing values  
   - Format checks  
   - Dimensionality reduction (PCA, T-SNE)  
   - Feature importance & selection  
3. **Exploratory Data Analysis (EDA)**  
   - Statistical summaries  
   - Visual insights with Matplotlib, Seaborn, Plotly  
4. **Feature Engineering**  
   - Aggregations (mean, sum, std)  
   - Time-based lag & rolling features  
   - One-hot / target encoding for categorical features  
5. **Imbalanced Data Handling**  
   - SMOTE (oversampling)  
   - Random undersampling  
   - Class weight adjustments  
   - Ensemble methods (EasyEnsemble, Balanced RF)  
6. **Model Training**  
   - Logistic Regression, Random Forest, XGBoost, LightGBM  
   - Neural Networks (MLP, LSTM for time-series behavior)  
7. **Hyperparameter Tuning**  
   - GridSearchCV / RandomizedSearchCV  
   - Optuna for Bayesian optimization  

---

## ⚙️ Tech Stack
- **Languages:** Python (NumPy, Pandas)  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **ML/DL Libraries:** Scikit-learn, XGBoost, LightGBM, PyTorch/TensorFlow  
- **Imbalance Handling:** imbalanced-learn (SMOTE, Ensemble methods)  
- **Dimensionality Reduction:** PCA, t-SNE  

---

## 📊 Evaluation Metrics
- **Primary Metric (Kaggle):** Amex metric (custom evaluation function)  
- **Other Metrics:** AUC-ROC, Precision, Recall, F1-score, Confusion Matrix  

---

## 🚀 Results
- Built multiple models with hyperparameter tuning  
- XGBoost/LightGBM showed the most promise on imbalanced data  
- Applied feature selection & aggregation for performance boost  

---

## 🏗️ Future Work
- Deploy the model via **Streamlit / FastAPI** for real-time predictions  
- Integrate **explainability (SHAP, LIME)** to interpret predictions  
- Enhance feature engineering using **time-series modeling** (LSTM, Transformers)  

---

## 📌 How to Run
```bash
# Clone the repository
git clone https://github.com/your-username/amex-default-prediction.git
cd amex-default-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Project_AMEX.ipynb

