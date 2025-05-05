import os
import logging
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Import the ML model
from heart_disease_model import HeartDiseaseModel

# Initialize the model
heart_model = HeartDiseaseModel()

@app.route("/")
def index():
    """Render the home page with dataset information and model overview."""
    # Get dataset summary and visualizations
    dataset_info = heart_model.get_dataset_info()
    
    # Get model performance metrics
    model_metrics = heart_model.get_model_metrics()
    
    # Generate visualizations
    age_plot = heart_model.plot_age_distribution()
    gender_plot = heart_model.plot_gender_distribution()
    target_plot = heart_model.plot_target_distribution()
    correlation_plot = heart_model.plot_correlation_matrix()
    feature_importance = heart_model.plot_feature_importance()
    
    return render_template(
        "index.html", 
        dataset_info=dataset_info,
        model_metrics=model_metrics,
        age_plot=age_plot,
        gender_plot=gender_plot,
        target_plot=target_plot,
        correlation_plot=correlation_plot,
        feature_importance=feature_importance
    )

@app.route("/predict", methods=["GET"])
def predict_form():
    """Render the prediction form."""
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Process the form data and make a prediction."""
    try:
        # Extract form data
        age = int(request.form.get("age"))
        sex = int(request.form.get("sex"))
        cp = int(request.form.get("cp"))
        trestbps = int(request.form.get("trestbps"))
        chol = int(request.form.get("chol"))
        fbs = int(request.form.get("fbs"))
        restecg = int(request.form.get("restecg"))
        thalach = int(request.form.get("thalach"))
        exang = int(request.form.get("exang"))
        oldpeak = float(request.form.get("oldpeak"))
        slope = int(request.form.get("slope"))
        ca = int(request.form.get("ca"))
        thal = int(request.form.get("thal"))
        
        # Create input data dictionary
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Make prediction
        prediction_results = heart_model.predict(input_data)
        
        # Store results in session
        session['prediction_results'] = prediction_results
        session['input_data'] = input_data
        
        return redirect(url_for('results'))
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        flash(f"Error making prediction: {str(e)}", "danger")
        return redirect(url_for('predict_form'))

@app.route("/results")
def results():
    """Display the prediction results."""
    prediction_results = session.get('prediction_results')
    input_data = session.get('input_data')
    
    if not prediction_results or not input_data:
        flash("No prediction data available. Please submit the form first.", "warning")
        return redirect(url_for('predict_form'))
    
    # Get feature explanations based on input data
    feature_explanations = heart_model.get_feature_explanations(input_data)
    
    # Get individual feature contributions (for SHAP values visualization)
    feature_contributions = heart_model.get_feature_contributions(input_data)
    
    return render_template(
        "results.html",
        prediction=prediction_results,
        input_data=input_data,
        feature_explanations=feature_explanations,
        feature_contributions=feature_contributions
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
