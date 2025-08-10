# CodeAlpha_Disease_Prediction

## Key Features:
- Comprehensive model comparison across 4 algorithms
- Standardized feature scaling for optimal performance
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Synthetic patient generation for model testing
- Benign/Malignant prediction with probability scores
- Stratified sampling for balanced class distribution

## How It Works:
1. Loads breast cancer dataset with 30 diagnostic features
2. Splits data into training/test sets (80/20) with stratification
3. Scales features using StandardScaler for normalization
4. Trains and evaluates 4 classification models:
   a) Logistic Regression
   b) Random Forest
   c) Support Vector Machine
   d) XGBoost
5. Generates synthetic patient data within observed feature ranges
6. Predicts diagnosis using best-performing model (XGBoost)
7. Outputs malignancy probability with clinical interpretation

## Code Structure:
- Uses scikit-learn and XGBoost for modeling
- Implements StandardScaler for feature normalization
- Maintains state with:
  - 'trained_models' dictionary storing fitted classifiers
  - 'results' list collecting evaluation metrics
  - 'scaler' object for consistent data transformation
- Core components:
  - Model training loop with metrics calculation
  - Synthetic patient generator (uniform distribution within feature ranges)
  - Prediction interpreter (0 = Malignant, 1 = Benign)
- Key variables:
  - 'results_df': Model performance comparison table
  - 'new_patient_scaled': Generated test case
  - 'prob_malignant': Malignancy probability (class 0)
- Workflow stages:
  1. Data loading and preparation
  2. Feature scaling
  3. Model training and evaluation
  4. New patient simulation
  5. Clinical prediction output
