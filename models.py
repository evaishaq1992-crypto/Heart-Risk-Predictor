from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from flask_sqlalchemy import SQLAlchemy
import datetime
import os

# Create database instance
db = SQLAlchemy()

# Define models for the heart disease detection system
class User(db.Model):
    """User model to store user information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.name}>"


class Prediction(db.Model):
    """Model to store prediction history"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    probability = Column(Float, nullable=False)
    prediction_result = Column(Boolean, nullable=False)
    risk_level = Column(String(20), nullable=False)
    model_name = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Store input clinical parameters
    age = Column(Integer, nullable=False)
    sex = Column(Integer, nullable=False)
    cp = Column(Integer, nullable=False)  # chest pain type
    trestbps = Column(Integer, nullable=False)  # resting blood pressure
    chol = Column(Integer, nullable=False)  # cholesterol
    fbs = Column(Integer, nullable=False)  # fasting blood sugar
    restecg = Column(Integer, nullable=False)  # resting ECG
    thalach = Column(Integer, nullable=False)  # max heart rate
    exang = Column(Integer, nullable=False)  # exercise induced angina
    oldpeak = Column(Float, nullable=False)  # ST depression
    slope = Column(Integer, nullable=False)  # slope of peak exercise ST segment
    ca = Column(Integer, nullable=False)  # number of major vessels colored
    thal = Column(Integer, nullable=False)  # thalassemia
    
    # Store all model predictions
    model_predictions = Column(JSON, nullable=True)
    
    # Store feature contributions
    feature_contributions = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction {self.id} - Result: {self.prediction_result}>"
