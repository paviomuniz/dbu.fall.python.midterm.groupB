# This is a comment from Pávio Muniz


# hello


# comment from Chloe

#       - Apply data science concepts—data cleaning, visualization, modeling,
#         and evaluation—to gain insights and showcase Python proficiency.
#
# ------------------------------------------------------------------------------
# 2. PROJECT TASKS IN DETAIL
# ------------------------------------------------------------------------------
#
# 2.1 Acquire, Clean, and Preprocess Data
#
#   (a) Data Acquisition
#       - Identify your data source: file-based (CSV, JSON), database, API, etc.
#       - Document how you obtained it. For example, if from an API, show the request.
#
#   (b) Data Cleaning
#       - Tasks: Handle missing values, remove duplicates, correct invalid entries.
#       - Python Tools: pandas methods (isnull, dropna, fillna, etc.).
#       - Tips: Always justify your decisions, e.g., why dropping vs. imputing missing values.
#
#   (c) Data Preprocessing
#       - Requirement: Use at least 2 preprocessing techniques 
#         (scaling, encoding, feature engineering, etc.).
#       - Tips: Ensure numeric vs. categorical variables are appropriately transformed.
#
# ------------------------------------------------------------------------------
# 2.2 Perform Exploratory Data Analysis (EDA) and Visualize Key Insights
#
#   (a) Exploratory Data Analysis
#       - Compute basic stats (mean, median, std, etc.).
#       - Identify correlations, outliers, or data imbalances.
#       - Use pandas describe(), info(), corr() for an overview.
#
#   (b) Data Visualization
#       - Requirement: At least 3 different visualization techniques (histogram, 
#         scatter plot, box plot, heatmap, etc.).
#       - Tips: Use clear labels, titles, and legends. Let visuals drive your EDA narrative.
#
# ------------------------------------------------------------------------------
# 2.3 Build and Evaluate a Machine Learning Model
#
#   (a) Model Building
#       - Requirement: At least 2 different ML algorithms 
#         (e.g., Logistic Regression, Random Forest, Linear Regression, etc.).
#       - Tips: Match the algorithm type to your target variable 
#         (classification vs. regression).
#
#   (b) Model Evaluation
#       - Requirement: At least 2 different evaluation metrics 
#         (accuracy, precision/recall, F1, RMSE, MAE, etc.).
#       - Tips: Present numeric results and interpret them in plain English. 
#         Consider basic hyperparameter tuning.
#
# ------------------------------------------------------------------------------
# 3. DELIVERABLES
# ------------------------------------------------------------------------------
#
#   3.1 Code
#       - A well-commented Python script or Jupyter Notebook with:
#         * Data acquisition, cleaning, preprocessing
#         * EDA and visualizations
#         * Model building, training, and evaluation
#       - Ensure reproducibility. Include data or instructions to access it.
#
#   3.2 Report (Due in 3 Weeks)
#       - Structure:
#         1) Introduction to the Dataset
#         2) Data Cleaning & Preprocessing Steps
#         3) EDA & Key Insights
#         4) Model Building & Evaluation
#         5) Conclusion
#         6) References (if any)
#
# ------------------------------------------------------------------------------
# 4. TEAM COLLABORATION AND SUBMISSION TIPS
# ------------------------------------------------------------------------------
#
#   (a) Group Roles
#       - Decide early who focuses on which aspect: data cleaning, modeling, etc.
#       - Use Git or a similar VCS to merge changes and maintain a single codebase.
#
#   (b) Progress Milestones
#       - Week 1: Finalize dataset, do initial cleaning and EDA.
#       - Week 2: Refine preprocessing, build and evaluate at least one model.
#       - Week 3: Complete second model, finalize visualizations, write report.
#
#   (c) Version Control
#       - Commit frequently, use branches for different tasks, review each other's code.
#
#   (d) Polish and Professionalism
#       - Keep code readable and well-structured (clear variable names, function docstrings).
#       - Proofread your report, ensure visualizations are well-labeled.
#
# ------------------------------------------------------------------------------
# 5. PUTTING IT ALL TOGETHER
# ------------------------------------------------------------------------------
#
# By following this guide, your group will:
#   - Acquire data from a new source and thoroughly clean it.
#   - Preprocess it (e.g., scaling, encoding, feature engineering) as needed.
#   - Conduct an informative EDA with multiple visualizations.
#   - Train at least two machine learning models, evaluate them with multiple metrics.
#   - Compile findings in a concise, well-organized final report.
#
# Good luck with your data exploration and modeling!
# ------------------------------------------------------------------------------

import os

# import matplotlib

# matplotlib.use("Agg")  # Use the Agg backend for non-GUI rendering
# import datetime
# import os

# import matplotlib.pyplot as plt
# import requests
# from sklearn.linear_model import LinearRegression
import pickle
from flask import Flask, render_template, request, jsonify
# render_template_string
import nbformat
from nbconvert import HTMLExporter
# import os
import requests

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("main.html")

@app.route("/EDA")
def mod_eda():

    notebook_path = 'eda\\MidTerm.ipynb'
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Convert to HTML
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook_content)

    return render_template('notebook_viewer.html', notebook_html=body)


@app.route("/predictions", methods=["GET", "POST"])
def mod_predictions():
    model_path = os.path.join('model', 'linear_regression_model.pkl')

    if request.method == "POST":
        try:
            # New feature list (order must match model training):
            # ['Age', 'Glucose', 'Blood Pressure', 'Oxygen Saturation', 'Cholesterol',
            #  'Triglycerides', 'HbA1c', 'Smoking', 'Alcohol', 'Physical Activity',
            #  'Family History', 'Stress Level', 'Sleep Hours', 'BMI_BP_Interaction',
            #  'Age_Stress_Interaction', 'Glucose_per_BMI', 'condition_arthritis',
            #  'condition_asthma', 'condition_cancer', 'condition_diabetes',
            #  'condition_healthy', 'condition_hypertension', 'condition_obesity']

            age = float(request.form.get('Age', 0))
            glucose = float(request.form.get('Glucose', 0))
            bp = float(request.form.get('Blood Pressure', 0))
            ox = float(request.form.get('Oxygen Saturation', 0))
            chol = float(request.form.get('Cholesterol', 0))
            trig = float(request.form.get('Triglycerides', 0))
            hba1c = float(request.form.get('HbA1c', 0))
            smoking = float(request.form.get('Smoking', 0))
            alcohol = float(request.form.get('Alcohol', 0))
            activity = float(request.form.get('Physical Activity', 0))
            fh = float(request.form.get('Family History', 0))
            stress = float(request.form.get('Stress Level', 0))
            sleep = float(request.form.get('Sleep Hours', 0))
            bmi_bp = float(request.form.get('BMI_BP_Interaction', 0))
            age_stress = float(request.form.get('Age_Stress_Interaction', 0))
            gluc_per_bmi = float(request.form.get('Glucose_per_BMI', 0))
            cond_arthritis = float(request.form.get('condition_arthritis', 0))
            cond_asthma = float(request.form.get('condition_asthma', 0))
            cond_cancer = float(request.form.get('condition_cancer', 0))
            cond_diabetes = float(request.form.get('condition_diabetes', 0))
            cond_healthy = float(request.form.get('condition_healthy', 0))
            cond_hyper = float(request.form.get('condition_hypertension', 0))
            cond_obesity = float(request.form.get('condition_obesity', 0))

            X = [[age, glucose, bp, ox, chol, trig, hba1c, smoking, alcohol,
                  activity, fh, stress, sleep, bmi_bp, age_stress, gluc_per_bmi,
                  cond_arthritis, cond_asthma, cond_cancer, cond_diabetes,
                  cond_healthy, cond_hyper, cond_obesity]]

            if not os.path.exists(model_path):
                return render_template('predictions.html', error=f"Model file not found: {model_path}")

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            prediction = model.predict(X)
            try:
                value = float(prediction[0])
            except Exception:
                value = float(prediction)

            return render_template('predictions.html', result=value)

        except Exception as e:
            return render_template('predictions.html', error=str(e))

    return render_template('predictions.html')


@app.route("/flask_api", methods=["GET", "POST"])
def mod_api_predictions():
    
    if request.method == "POST":
        try:
            # New feature list (order must match model training):
            age = float(request.form.get('Age', 0))
            glucose = float(request.form.get('Glucose', 0))
            bp = float(request.form.get('Blood Pressure', 0))
            ox = float(request.form.get('Oxygen Saturation', 0))
            chol = float(request.form.get('Cholesterol', 0))
            trig = float(request.form.get('Triglycerides', 0))
            hba1c = float(request.form.get('HbA1c', 0))
            smoking = float(request.form.get('Smoking', 0))
            alcohol = float(request.form.get('Alcohol', 0))
            activity = float(request.form.get('Physical Activity', 0))
            fh = float(request.form.get('Family History', 0))
            stress = float(request.form.get('Stress Level', 0))
            sleep = float(request.form.get('Sleep Hours', 0))
            bmi_bp = float(request.form.get('BMI_BP_Interaction', 0))
            age_stress = float(request.form.get('Age_Stress_Interaction', 0))
            gluc_per_bmi = float(request.form.get('Glucose_per_BMI', 0))
            cond_arthritis = float(request.form.get('condition_arthritis', 0))
            cond_asthma = float(request.form.get('condition_asthma', 0))
            cond_cancer = float(request.form.get('condition_cancer', 0))
            cond_diabetes = float(request.form.get('condition_diabetes', 0))
            cond_healthy = float(request.form.get('condition_healthy', 0))
            cond_hyper = float(request.form.get('condition_hypertension', 0))
            cond_obesity = float(request.form.get('condition_obesity', 0))

            X = {           
                "Age": age,
                "Glucose": glucose,
                "Blood Pressure": bp,
                "Oxygen Saturation": ox,
                "Cholesterol": chol,
                "Triglycerides": trig,
                "HbA1c": hba1c,
                "Smoking": smoking,
                "Alcohol": alcohol,
                "Physical Activity": activity,
                "Family History": fh,
                "Stress Level": stress,
                "Sleep Hours": sleep,
                "BMI_BP_Interaction": bmi_bp,
                "Age_Stress_Interaction": age_stress,
                "Glucose_per_BMI": gluc_per_bmi,
                "condition_arthritis": cond_arthritis,
                "condition_asthma": cond_asthma,
                "condition_cancer": cond_cancer,
                "condition_diabetes": cond_diabetes,
                "condition_healthy": cond_healthy,
                "condition_hypertension": cond_hyper,
                "condition_obesity": cond_obesity
            }
            response = requests.post(
                request.url_root + "api/predict",
                headers={"Content-Type": "application/json"},
                json=X,
                timeout=30
            )

            return jsonify(response.json())

        except Exception as e:
            return render_template('api.html', error=str(e))

    return render_template('api.html')

    

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint for predictions. Accepts POST with JSON data containing the 23 features.
    Returns JSON with prediction or error message.
    
    Example request:
    {
        "Age": 45,
        "Glucose": 120,
        "Blood Pressure": 80,
        ...
    }

    Returns:
    {
        "prediction": 123.45
    }
    or
    {
        "error": "Missing required field: Age"
    }
    """
    model_path = os.path.join('model', 'linear_regression_model.pkl')

    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Required fields in exact order
        required_fields = [
            'Age', 'Glucose', 'Blood Pressure', 'Oxygen Saturation', 'Cholesterol',
            'Triglycerides', 'HbA1c', 'Smoking', 'Alcohol', 'Physical Activity',
            'Family History', 'Stress Level', 'Sleep Hours', 'BMI_BP_Interaction',
            'Age_Stress_Interaction', 'Glucose_per_BMI', 'condition_arthritis',
            'condition_asthma', 'condition_cancer', 'condition_diabetes',
            'condition_healthy', 'condition_hypertension', 'condition_obesity'
        ]

        # Check for missing fields
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        # Convert all values to float in the exact order needed
        try:
            values = [float(data[field]) for field in required_fields]
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid value format: {str(e)}"}), 400

        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {model_path}"}), 500

        # Load model and predict
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict([values])
        try:
            value = float(prediction[0])
        except Exception:
            value = float(prediction)

        return jsonify({"prediction": value})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not os.path.exists("static"):
        os.mkdir("static")
    app.run(debug=True)