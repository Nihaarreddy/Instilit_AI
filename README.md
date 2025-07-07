##Employee Salary Prediction Pipeline
---
Overview
This project delivers a robust machine learning pipeline to predict employee salaries using job and company features.
It covers data ingestion, preprocessing, outlier handling, model selection, explainability, drift detection, and experiment tracking.
---
![Prediction Ui](https://github.com/user-attachments/assets/0cf5e97d-65a9-4d3e-a42d-7a4bead14f2d)

---
##Features
Data Ingestion: Fetches data from PostgreSQL via SQLAlchemy.

Preprocessing: Handles missing values, outliers, and categorical encoding.

Modeling: Supports Linear Regression, Ridge, Random Forest, and XGBoost with hyperparameter tuning.

Evaluation: Calculates MAE, RMSE, R², and MAPE.

Experiment Tracking: Logs parameters, metrics, and artifacts with MLflow.

Explainability: Generates SHAP plots for feature importance.

Drift Detection: Uses Evidently to monitor and log data drift.

Deployment Ready: Exports a scikit-learn pipeline for inference.
---
##Project Structure
text
```.
├── data_ingestion.py
├── main.py
├── requirements.txt
├── README.md
├── saved_models/
├── drift_reports/
├── shap_outputs/
└── ...
```

![Screenshot 2025-07-07 014756](https://github.com/user-attachments/assets/b389e78f-cb3c-41d3-9a61-0f4ac132e999)

Setup
Clone the repository:

```bash
git clone https://github.com/Nihaarreddy/Instilit_AI.git
cd Instilit_AI
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Configure your database:
![Screenshot 2025-07-06 015356](https://github.com/user-attachments/assets/a030a7f8-b656-4128-8854-0097a17ea3d5)



Edit the connection details in main.py or your configuration file.

(Optional) Start MLflow Tracking Server:

```bash
mlflow ui
```
Access MLflow UI at http://localhost:5000.

Usage
Run the pipeline:

```bash
python main.py
```
Check MLflow UI:
View experiments, runs, and model artifacts at http://localhost:5000.

Model Inference Example:

```python
import joblib
pipeline = joblib.load('saved_models/final_pipeline_<model_name>_new_pipeline.pkl')
predictions = pipeline.predict(X_new)  # X_new is a DataFrame with the same columns as training
```
MLflow Tracking

All model parameters, metrics, and artifacts (including SHAP plots and drift reports) are logged to MLflow.
![MLFlow with Experiments](https://github.com/user-attachments/assets/e0bf2751-1c31-4308-89eb-90f09a3a14dc)


Models are registered and versioned in the MLflow Model Registry.


Model Explainability
SHAP summary plots are generated for each model and logged as artifacts.

These plots help visualize feature importance and model decisions.

Data Drift Detection
Drift reports are generated using Evidently and logged to MLflow.

Reports compare train/validation/test splits to monitor data distribution changes.

Customization
Add/remove models: Edit the models dictionary in main.py.

Change features: Edit the feature selection section.

Tune hyperparameters: Update the params in the models dictionary.




License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
scikit-learn

XGBoost

MLflow

Evidently

SHAP

Contact
For questions or suggestions, please open an issue
```

Happy Modeling! 🚀

Tip: Update the repo URL, contact info, and any specific details to match your project setup!


