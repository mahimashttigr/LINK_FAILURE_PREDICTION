# Optical Network Failure Prediction Module

This project implements a machine learning pipeline to predict failures in an optical network based on operational telemetry data. The solution is designed for imbalanced classification tasks, where failure events are rare.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Project Structure](#project-structure)  
3. [How to Run the Project](#how-to-run-the-project)  
4. [Implementation Overview](#implementation-overview)  

---

## Prerequisites

- Python 3.7+  
- Jupyter Notebook or Google Colab environment  

### Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```
## Project Structure
The project is divided into three separate scripts:

#### data_processor.py
Uploads and preprocesses the CSV data (synthetic_network_data.csv)
Performs basic feature engineering (X1 and X2)
Handles missing values and prepares data for modeling
Performs  feature engineering 
Splits data into train/test sets 
#### model_trainer.py
Trains four classifiers:
Random Forest
Logistic Regression
RBF SVM
Quadratic Discriminant Analysis (QDA)
Saves trained models and scaler.pkl in a models/ directory
#### model_tester.py
Evaluates each model with classification_report (precision, recall, F1-score)
Displays 2D decision boundary plots using X1 and X2

##  How to Run the Project
#### Step 1: Data Upload
Place the CSV data file (synthetic_network_data.csv) in the same directory as the scripts.
Alternatively, in Colab, upload the file using:

python 
```bash
from google.colab import files
uploaded = files.upload()
```
#### Step 2: Data Processing
Run data_processor.py:

```bash
python data_processor.py
````
Loads and inspects data
Drops rows with NaN values
Engineers features X1 and X2

#### Step 3: Model Training
Run model_trainer.py:

````bash
python model_trainer.py
````
#### Step 4: Model Evaluation
Run model_tester.py:

````bash
python model_tester.py
````
## Implementation Overview
Feature Engineering:
X1 = amplifier_target_db - amplifier_gain_db
X2 = span_loss_target_db - span_loss_db
Include signal_db and osnr_db for modeling
Model Selection: Random Forest, Logistic Regression, RBF SVM, and QDA
Imbalance Handling: class_weight='balanced' used for Random Forest, Logistic Regression, and RBF SVM
Data Preparation: Features standardized using StandardScaler. Stratified train-test split ensures class balance
Training and Saving: All models trained on the 4-feature dataset and saved using joblib
Evaluation: Test set evaluation includes precision, recall, F1-score, and decision boundary visualization (2D using X1/X2)
