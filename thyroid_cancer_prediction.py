import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, 
                            confusion_matrix, roc_auc_score, roc_curve,
                            precision_recall_curve, average_precision_score)
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tabulate import tabulate
from warnings import filterwarnings
filterwarnings('ignore')

# Constants
REQUIRED_COLUMNS = ['Age', 'Recurred']
OPTIONAL_COLUMNS = [
    'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiotherapy',
    'Thyroid Function', 'Physical Examination', 'Adenopathy',
    'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response'
]

def print_table(title, data, headers=None, tablefmt="grid"):
    """Print data in a formatted table"""
    print(f"\n{title}:")
    if headers:
        print(tabulate(data, headers=headers, tablefmt=tablefmt, stralign="left", numalign="center"))
    else:
        print(tabulate(data, tablefmt=tablefmt, stralign="left", numalign="center"))

def clean_column_names(df):
    """Standardize column names and fix common misspellings"""
    df.columns = df.columns.str.strip()
    column_mapping = {
        'Hx Radiothreapy': 'Hx Radiotherapy',
        'ThyroidFunction': 'Thyroid Function',
        'PhysicalExam': 'Physical Examination',
        'recurred': 'Recurred',
        'recurrence': 'Recurred'
    }
    return df.rename(columns=column_mapping)

def convert_target_variable(df):
    """Convert target variable to binary (0/1) with robust handling"""
    if 'Recurred' not in df.columns:
        return df
    
    # First convert to string and clean
    df['Recurred'] = df['Recurred'].astype(str).str.strip().str.lower()
    
    # Map all possible variations to binary values
    true_values = ['1', 'yes', 'true', 't', 'y', 'recurred']
    false_values = ['0', 'no', 'false', 'f', 'n', 'non-recurred']
    
    # Create mapping dictionary
    mapping = {**{val: 1 for val in true_values},
               **{val: 0 for val in false_values}}
    
    # Apply mapping
    df['Recurred'] = df['Recurred'].map(mapping)
    
    # Check for unconverted values
    if df['Recurred'].isnull().any():
        invalid = df[df['Recurred'].isnull()]['Recurred'].unique()
        raise ValueError(f"Invalid values in 'Recurred': {invalid}. Should be 0/1 or Yes/No")
    
    return df

def clean_data_values(df):
    """Clean problematic data values in the dataframe"""
    for col in df.columns:
        if df[col].dtype == object and col != 'Recurred':
            # Fix concatenated Yes/No strings
            df[col] = df[col].astype(str).str.replace(r'No+', 'No', regex=True)
            df[col] = df[col].str.replace(r'Yes+', 'Yes', regex=True)
            
            # Standardize categorical values
            df[col] = df[col].str.strip().replace({
                'yes': 'Yes', 'y': 'Yes',
                'no': 'No', 'n': 'No',
                'male': 'Male', 'm': 'Male',
                'female': 'Female', 'f': 'Female'
            })
    return df

def load_data(file_path='dataset.csv'):
    """Load and validate dataset with robust error handling"""
    try:
        # Validate file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file '{file_path}' not found")
        
        # Load data with specific dtype to prevent auto-conversion
        dtype_dict = {col: str for col in OPTIONAL_COLUMNS}
        dtype_dict.update({'Age': float})
        
        try:
            df = pd.read_csv(file_path, dtype=dtype_dict)
        except pd.errors.ParserError:
            raise ValueError("Malformed CSV file - please check file format")
        
        # Clean data
        df = clean_column_names(df)
        df = convert_target_variable(df)
        df = clean_data_values(df)
        
        # Check for required columns
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
            
        # Check for at least 3 optional clinical features
        available_optional = [col for col in OPTIONAL_COLUMNS if col in df.columns]
        if len(available_optional) < 3:
            raise ValueError(f"Insufficient clinical features. Found only: {available_optional}")
        
        # Convert 'Age' to numeric
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        if df['Age'].isnull().any():
            raise ValueError("Invalid values found in 'Age' column")
            
        # Verify target variable
        if not all(df['Recurred'].isin([0, 1])):
            invalid_values = df[~df['Recurred'].isin([0, 1])]['Recurred'].unique()
            raise ValueError(f"Invalid values in 'Recurred': {invalid_values}. Should be 0/1 or Yes/No")
            
        print("\nData loaded successfully with columns:")
        print_table("Dataset Columns", [[col] for col in df.columns.tolist()], ["Column Name"])
        print(f"\nRecurrence rate: {df['Recurred'].mean():.2%}")
        
        return df
    
    except Exception as e:
        print(f"\nERROR LOADING DATA: {str(e)}")
        print("\nPlease ensure:")
        print("1. The dataset file exists in the current directory")
        print("2. Contains these required columns:", ", ".join(REQUIRED_COLUMNS))
        print("3. 'Recurred' contains only 0/1 or Yes/No values")
        print("4. 'Age' contains only numeric values")
        raise

def analyze_data(df):
    """Perform comprehensive data analysis with formatted tables"""
    print("\n=== Data Quality Report ===")
    
    # Missing values analysis
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print_table("Missing Values", 
                   [[col, count] for col, count in missing[missing > 0].items()],
                   ["Column", "Missing Count"])
    else:
        print("\nNo missing values found")
    
    # Data type verification
    print_table("Data Types", 
               [[col, dtype] for col, dtype in df.dtypes.items()],
               ["Column", "Data Type"])
    
    # Value distribution for categorical columns
    print("\nCategorical Value Distributions:")
    cat_cols = [col for col in df.columns if df[col].dtype == object]
    for col in cat_cols:
        value_counts = df[col].value_counts(dropna=False)
        print_table(f"\n{col} Distribution",
                   [[val, count] for val, count in value_counts.items()],
                   ["Value", "Count"])
    
    # Numerical stats
    print_table("\nNumerical Statistics", 
               df[['Age']].describe().round(2).reset_index().values.tolist(),
               ["Statistic", "Age"])
    
    return df

def build_preprocessor(df):
    """Create dynamic preprocessing pipeline with column validation"""
    # Identify available features
    num_features = ['Age'] if 'Age' in df.columns else []
    cat_features = [col for col in OPTIONAL_COLUMNS if col in df.columns]
    
    print_table("Features Being Used",
               [["Numerical:"] + num_features,
                ["Categorical:"] + cat_features])
    
    # Validate categorical values
    for col in cat_features:
        unique_vals = df[col].unique()
        if len(unique_vals) > 20:
            print(f"Warning: Column '{col}' has {len(unique_vals)} unique values")
    
    # Create transformers
    transformers = []
    
    if num_features:
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_transformer, num_features))
    
    if cat_features:
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, cat_features))
    
    if not transformers:
        raise ValueError("No valid features found for preprocessing")
    
    return ColumnTransformer(transformers=transformers)

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """Complete model training and evaluation workflow with formatted output"""
    # Preprocess data
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        raise
    
    # Get feature names
    feature_names = []
    if hasattr(preprocessor, 'named_transformers_'):
        if 'num' in preprocessor.named_transformers_:
            feature_names.extend(['Age'])
        if 'cat' in preprocessor.named_transformers_:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            if hasattr(preprocessor.named_transformers_['cat'], 'feature_names_in_'):
                cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
            else:
                cat_features = [col for col in OPTIONAL_COLUMNS if col in X_train.columns]
            
            if hasattr(cat_encoder, 'get_feature_names_out'):
                feature_names.extend(cat_encoder.get_feature_names_out(cat_features))
            else:
                feature_names.extend([f"{col}_{val}" for col, vals in zip(cat_features, cat_encoder.categories_) 
                                    for val in vals])
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_processed, y_train)
    
    # Evaluate model
    print("\n=== Model Evaluation ===")
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Metrics
    metrics_data = [
        ["Accuracy", f"{accuracy_score(y_test, y_pred):.4f}"],
        ["AUC-ROC", f"{roc_auc_score(y_test, y_proba):.4f}"],
        ["Average Precision", f"{average_precision_score(y_test, y_proba):.4f}"]
    ]
    print_table("Model Performance Metrics", metrics_data, ["Metric", "Value"])
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['No Recurrence', 'Recurrence'], output_dict=True)
    report_data = [
        [k, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1-score']:.2f}", v['support']]
        for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']
    ]
    print_table("Classification Report", report_data, 
               ["Class", "Precision", "Recall", "F1-Score", "Support"])
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes'])
    plt.title('Confusion Matrix\n(Actual vs Predicted Recurrence)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax2.plot(recall, precision, label=f'AP = {average_precision_score(y_test, y_proba):.2f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', 
                   data=importance.head(15),
                   palette='viridis')
        plt.title('Top 15 Important Features')
        plt.tight_layout()
        plt.show()
    
    return model

def save_artifacts(model, preprocessor):
    """Save model artifacts with versioning"""
    try:
        version = 1
        while os.path.exists(f'thyroid_cancer_model_v{version}.pkl'):
            version += 1
            
        model_path = f'thyroid_cancer_model_v{version}.pkl'
        preprocessor_path = f'preprocessor_v{version}.pkl'
        
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        
        print_table("Model Artifacts Saved",
                   [[model_path], [preprocessor_path]],
                   ["File Path"])
    except Exception as e:
        print(f"\nError saving artifacts: {str(e)}")

def main():
    try:
        # Load and analyze data
        print("Loading and processing data...")
        df = load_data()
        df = analyze_data(df)
        
        # Build preprocessing pipeline
        preprocessor = build_preprocessor(df)
        
        # Split data
        X = df.drop('Recurred', axis=1)
        y = df['Recurred']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train and evaluate
        model = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
        
        # Save artifacts
        save_artifacts(model, preprocessor)
        
        # Sample prediction
        sample_idx = np.random.randint(0, len(X_test))
        sample = X_test.iloc[sample_idx:sample_idx+1]
        sample_preprocessed = preprocessor.transform(sample)
        sample_pred = model.predict(sample_preprocessed)[0]
        sample_proba = model.predict_proba(sample_preprocessed)[0]
        
        print("\n=== Sample Prediction ===")
        print_table("Patient Features", 
                   [[k, v] for k, v in sample.to_dict('records')[0].items()],
                   ["Feature", "Value"])
        
        pred_data = [
            ["Prediction", 'RECURRENCE' if sample_pred == 1 else 'NO RECURRENCE'],
            ["Confidence", f"{max(sample_proba):.1%}"],
            ["Probability (No Recurrence)", f"{sample_proba[0]:.3f}"],
            ["Probability (Recurrence)", f"{sample_proba[1]:.3f}"]
        ]
        print_table("Prediction Results", pred_data, ["Metric", "Value"])
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Execution terminated")

if __name__ == "__main__":
    main()