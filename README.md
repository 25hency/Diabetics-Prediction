# Diabetics Prediction

This project predicts whether a person is diabetic or not based on various health parameters using machine learning models.

## Dataset
The dataset used is `diabetes.csv`, which contains the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (1 for diabetic, 0 for non-diabetic)

## Features
- Data preprocessing: Handling missing values and standard scaling.
- Machine learning models:
  - Random Forest Classifier
  - Decision Tree Classifier
- Model evaluation: Accuracy score, confusion matrix, and classification report.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Diabetics-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Diabetics-Prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script to train the models and make predictions:
```bash
python Diabetics_prediction.py
```

## Output
- Accuracy scores for Random Forest and Decision Tree models.
- Confusion matrix and classification report for the Decision Tree model.
- Example prediction for a specific patient.

