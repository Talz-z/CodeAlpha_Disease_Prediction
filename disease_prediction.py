import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target


X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier()
}

results = []
trained_models = {}  

for name, model in models.items():
    
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_test_scaled)[:,1]
    preds = model.predict(X_test_scaled)
    
   
    trained_models[name] = model
    
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

results_df = pd.DataFrame(results)
print("Model Performance Comparison:")
print(results_df.round(3))


np.random.seed(None) 
min_vals = X_train.min()
max_vals = X_train.max()
new_patient = np.random.uniform(low=min_vals, high=max_vals).reshape(1, -1)
new_patient_df = pd.DataFrame(new_patient, columns=cancer.feature_names)


new_patient_scaled = scaler.transform(new_patient_df)


best_model = trained_models["XGBoost"]


prediction = best_model.predict(new_patient_scaled)

prob_malignant = best_model.predict_proba(new_patient_scaled)[0][0]



prediction_label = 'Malignant (Cancer)' if prediction[0] == 0 else 'Benign (Healthy)'


print("\nGenerated New Patient Features:")
print(new_patient_df.round(3))
print(f"\nPrediction: {prediction_label}")
print(f"Probability of malignancy: {prob_malignant:.1%}")