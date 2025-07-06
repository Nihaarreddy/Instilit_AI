from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Initialize Flask app
app = Flask(__name__)

# Load the trained full pipeline
pipeline = joblib.load('saved_models/final_pipeline_RandomForest.pkl')

# Database connection
DB_URI = 'postgresql://postgres:Nihaar6@localhost:5432/postgres'
engine = create_engine(DB_URI)
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None

    if request.method == 'POST':
        try:
            input_data = {
                'job_title': request.form.get('job_title', '').lower().strip(),
                'experience_level': request.form.get('experience_level', ''),
                'employment_type': request.form.get('employment_type', ''),
                'company_size': request.form.get('company_size', ''),
                'company_location': request.form.get('company_location', ''),
                'remote_ratio': float(request.form.get('remote_ratio', 0)),
                'years_experience': float(request.form.get('years_experience', 0)),
                'stock_options': float(request.form.get('stock_options', 0)),
                'conversion_rate': float(request.form.get('conversion_rate', 0)),
                'total_salary': float(request.form.get('total_salary', 0)),
                'bonus': float(request.form.get('bonus', 0)),
                'salary_currency': request.form.get('salary_currency', ''),
                'salary_in_usd': float(request.form.get('salary_in_usd', 0)),
                'base_salary': float(request.form.get('base_salary', 0)),
                'currency': request.form.get('currency', '')
            }
            input_df = pd.DataFrame([input_data])

            # Ensure columns are in the right order (if needed)
            expected_columns = [
                'job_title', 'experience_level', 'employment_type', 'company_size', 'company_location',
                'remote_ratio', 'years_experience',
                'stock_options', 'conversion_rate', 'total_salary', 'bonus',
                'salary_currency', 'salary_in_usd', 'base_salary', 'currency'
            ]
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0 if col not in ['job_title', 'experience_level', 'employment_type', 'company_size', 'company_location', 'salary_currency', 'currency'] else ''
            input_df = input_df[expected_columns]

            # --- Invert log1p transformation ---
            predicted_log_salary = pipeline.predict(input_df)[0]
            predicted_salary = np.expm1(predicted_log_salary)

            converted_salary = predicted_salary * input_data['conversion_rate']
            currency_name = input_data['currency'].upper()
            prediction_text = f"Predicted Total Salary: {currency_name}{converted_salary:,.2f}"

            # Save to database
            input_data['adjusted_total_usd'] = converted_salary
            pd.DataFrame([input_data]).to_sql('employee_salaries', engine, if_exists='append', index=False)
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('predict.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True, port = 8000)
