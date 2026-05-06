# Data Preprocessing & Feature Engineering

This repository contains a  data preprocessing and feature engineering pipeline applied to three classic machine learning datasets: **Student Performance**, **Titanic**, and **House Prices**.

##  Overview
The project automates the transition from raw data to model-ready features through a systematic six-step workflow:
1. **Data Cleaning**: Duplicate removal and structural verification.
2. **Missing Value Handling**: Statistical imputation (Median/Mode).
3. **Categorical Encoding**: Label and One-Hot encoding based on feature cardinality.
4. **Feature Scaling**: Standardization using `StandardScaler`.
5. **Feature Engineering**: Creation of new predictive indicators (e.g., `TotalSF`, `FamilySize`).
6. **Feature Selection**: Automated selection using Random Forest importance.

##  Datasets Covered
- **Student Performance**: Predict student final grades based on social and academic factors.
- **Titanic**: Predict survival probability using passenger demographics.
- **House Prices**: Regression analysis to predict sales prices using residential features.

##  Installation & Usage
### Prerequisites
- Python 3.8+
- Pandas, NumPy, Scikit-Learn

### Setup
```bash
git clone https://github.com/your-username/ML-Preprocessing-Feature-Engineering.git
cd ML-Preprocessing-Feature-Engineering
pip install pandas numpy scikit-learn
