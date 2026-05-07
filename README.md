# Hospital Readmission Risk Predictor

A machine learning project that predicts 30-day hospital readmission risk for diabetic patients using 10 years of clinical data from 130 US hospitals. The project includes data cleaning, feature engineering, three classification models, hyperparameter tuning, SHAP Analysis and an interactive Streamlit app.

## Dataset

Source: UCI Machine Learning Repository

Link: https://archive.ics.uci.edu/dataset/296/diabetes+130-us-hospitals-for-years-1999-2008

The dataset contains over 100,000 hospital records from diabetic patients across 130 US hospitals spanning 1999 to 2008. The target variable is whether a patient was readmitted within 30 days of discharge.

## Repository Structure

```
Hospital-Readmission-Analytics/
├── app.py                   # Streamlit web application                                                                                       
├── data.ipynb               # Data cleaning, data exploration, and feature engineering
├── database.ipynb           # Data Storage
├── feature_columns.pkl      # Feature column names for model input
├── final_data.csv           # Cleaned data used for building models, evaluation, SHAP analysis, and web application
├── final_data.db            # Database with the final cleaned data
├── gb_model.pkl             # Trained Gradient Boosting model
├── icd_code.csv             # ICD-9 diagnosis code mappings
├── icd_code.json            # Raw ICD-9 code data
├── model.ipynb              # Data modeling, evaluation, hyperparameter tuning, SHAP analysis
├── original_data.csv        # Original dataset from UCI Machine Learning Repository
├── README.md                # Project documentation
├── requirements.txt         # Project documentation
└── visualization.ipynb      # Data visualization
```

## Setup and Installation

All required libraries are listed in requirements.txt.

## Running the Jupyter Notebooks

Run the notebooks in the following order, since each one depends on the output of the previous:

1. data.ipynb
2. database.ipynb
3. visualization.ipynb
4. model.ipynb

## Running the Streamlit App

Make sure gb_model.pkl and feature_columns.pkl are in the folder.

Set up a virtual environment:

```python
# windows
python -m venv venv
venv\Scripts\activate
```

```python
# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Then run:

```
streamlit run app.py
```

The app will open automatically in your browser. Enter patient details and click Predict Readmission Risk to generate a prediction and SHAP explanation.