import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import os

def process_student_performance(input_path, output_path):
    print("Processing Student Performance Dataset...")
    # Load with semicolon delimiter as UCI datasets often use it
    df = pd.read_csv(input_path, sep=';')
    
    # 1. Clean data (check for duplicates)
    df.drop_duplicates(inplace=True)
    
    # 2. Handle missing values (UCI dataset usually clean, but let's be sure)
    # No missing values expected here, but filling anyway for robustness
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    # 3. Feature Engineering
    df['total_grade'] = df['G1'] + df['G2'] + df['G3']
    df['grade_improvement'] = df['G3'] - df['G1']
    
    # 4. Encode categorical variables
    # For binary columns, use LabelEncoder; for others, use OneHot
    le = LabelEncoder()
    for col in ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
        df[col] = le.fit_transform(df[col])
    
    # OneHot for multi-category
    df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)
    
    # 5. Scale numerical features
    scaler = StandardScaler()
    scale_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'total_grade', 'grade_improvement']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    # 6. Feature Selection (using Random Forest for G3 prediction)
    X = df.drop(['G3'], axis=1)
    y = df['G3'].astype(int) # Simplification for selection
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), max_features=10)
    selector.fit(X, y)
    important_features = X.columns[selector.get_support()]
    print(f"Important features for Student Performance: {list(important_features)}")
    
    df.to_csv(output_path, index=False)
    return list(important_features)

def process_titanic(input_path, output_path):
    print("Processing Titanic Dataset...")
    df = pd.read_csv(input_path)
    
    # 1. Clean data
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    # 2. Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 3. Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 4. Encode categorical variables
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    # 5. Scale numerical features
    scaler = StandardScaler()
    df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(df[['Age', 'Fare', 'FamilySize']])
    
    # 6. Feature Selection
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=5)
    selector.fit(X, y)
    important_features = X.columns[selector.get_support()]
    print(f"Important features for Titanic: {list(important_features)}")
    
    df.to_csv(output_path, index=False)
    return list(important_features)

def process_house_prices(input_path, output_path):
    print("Processing House Prices Dataset...")
    df = pd.read_csv(input_path)
    
    # 1. Clean data (Drop columns with many missing values)
    missing_pct = df.isnull().sum() / len(df)
    drop_cols = missing_pct[missing_pct > 0.4].index
    df.drop(drop_cols, axis=1, inplace=True)
    df.drop('Id', axis=1, inplace=True)
    
    # 2. Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    # 3. Feature Engineering
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # 4. Encode categorical variables
    # Using Label Encoding for simplicity in this script (many categories)
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # 5. Scale numerical features
    scaler = StandardScaler()
    # Scale all except target SalePrice
    scale_cols = [c for c in df.columns if c != 'SalePrice']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    # 6. Feature Selection
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), max_features=10)
    selector.fit(X, y)
    important_features = X.columns[selector.get_support()]
    print(f"Important features for House Prices: {list(important_features)}")
    
    df.to_csv(output_path, index=False)
    return list(important_features)

if __name__ == "__main__":
    results = {}
    if not os.path.exists('cleaned_datasets'):
        os.makedirs('cleaned_datasets')
        
    results['Student Performance'] = process_student_performance('datasets/student/student-mat.csv', 'cleaned_datasets/student_cleaned.csv')
    results['Titanic'] = process_titanic('datasets/titanic.csv', 'cleaned_datasets/titanic_cleaned.csv')
    results['House Prices'] = process_house_prices('datasets/house_prices.csv', 'cleaned_datasets/house_prices_cleaned.csv')
    
    print("\nPre-processing complete!")
    for ds, features in results.items():
        print(f"{ds} Top Features: {features}")
