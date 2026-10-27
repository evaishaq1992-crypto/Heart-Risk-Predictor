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
from models import db, User, Prediction

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database with a fallback to SQLite if DATABASE_URL is not set
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///database.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database with the app
db.init_app(app)

# Create all database tables
with app.app_context():
    db.create_all()

# Import the ML model
from heart_disease_model import HeartDiseaseModel

# Initialize the model
heart_model = HeartDiseaseModel()

@app.route("/")
def index():
    """Render the home page with dataset information and model overview."""
    dataset_info = heart_model.get_dataset_info()
    model_metrics = heart_model.get_model_metrics()
    
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
        
        prediction_results = heart_model.predict(input_data)
        feature_contributions = heart_model.get_feature_contributions(input_data)
        
        try:
            new_prediction = Prediction(
                probability=prediction_results['probability'],
                prediction_result=prediction_results['prediction'] == 1,
                risk_level=prediction_results['risk_level'],
                model_name=prediction_results['model_name'],
                age=age,
                sex=sex,
                cp=cp,
                trestbps=trestbps,
                chol=chol,
                fbs=fbs,
                restecg=restecg,
                thalach=thalach,
                exang=exang,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal,
                model_predictions=prediction_results['model_predictions'],
                feature_contributions=feature_contributions
            )
            
            db.session.add(new_prediction)
            db.session.commit()
            logger.info(f"Prediction saved to database with ID: {new_prediction.id}")
            session['prediction_id'] = new_prediction.id
            
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            db.session.rollback()
        
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
    prediction_id = session.get('prediction_id')
    
    if not prediction_results or not input_data:
        flash("No prediction data available. Please submit the form first.", "warning")
        return redirect(url_for('predict_form'))
    
    feature_explanations = heart_model.get_feature_explanations(input_data)
    feature_contributions = heart_model.get_feature_contributions(input_data)
    
    return render_template(
        "results.html",
        prediction=prediction_results,
        input_data=input_data,
        feature_explanations=feature_explanations,
        feature_contributions=feature_contributions,
        prediction_id=prediction_id
    )

@app.route("/history")
def prediction_history():
    """Display prediction history from the database."""
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", predictions=predictions)

@app.route("/history/<int:prediction_id>")
def prediction_detail(prediction_id):
    """Display detailed results for a specific prediction."""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    input_data = {
        'age': prediction.age,
        'sex': prediction.sex,
        'cp': prediction.cp,
        'trestbps': prediction.trestbps,
        'chol': prediction.chol,
        'fbs': prediction.fbs,
        'restecg': prediction.restecg,
        'thalach': prediction.thalach,
        'exang': prediction.exang,
        'oldpeak': prediction.oldpeak,
        'slope': prediction.slope,
        'ca': prediction.ca,
        'thal': prediction.thal
    }
    
    feature_explanations = heart_model.get_feature_explanations(input_data)
    
    prediction_results = {
        'probability': prediction.probability,
        'prediction': 1 if prediction.prediction_result else 0,
        'risk_level': prediction.risk_level,
        'model_name': prediction.model_name,
        'model_predictions': prediction.model_predictions
    }
    
    return render_template(
        "results.html",
        prediction=prediction_results,
        input_data=input_data,
        feature_explanations=feature_explanations,
        feature_contributions=prediction.feature_contributions,
        prediction_id=prediction.id,
        from_history=True
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)