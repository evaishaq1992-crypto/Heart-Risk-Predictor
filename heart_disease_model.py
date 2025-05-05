import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.pipeline import Pipeline
import warnings

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class HeartDiseaseModel:
    def __init__(self):
        """Initialize the Heart Disease model."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # Load or train the model
        try:
            self._load_model()
        except (FileNotFoundError, EOFError):
            logger.info("No saved model found. Training new model...")
            self._load_data()
            self._preprocess_data()
            self._train_models()
            self._save_model()
    
    def _load_data(self):
        """Load the UCI Heart Disease dataset."""
        try:
            # First try to load from the static directory
            self.data = pd.read_csv('static/data/heart.csv')
            logger.info("Loaded dataset from local file")
        except FileNotFoundError:
            # If local file not found, download from UCI repository
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            self.data = pd.read_csv(url, names=column_names)
            logger.info("Downloaded dataset from UCI repository")
            
            # Save dataset for future use
            os.makedirs('static/data', exist_ok=True)
            self.data.to_csv('static/data/heart.csv', index=False)
        
        # Check if data was loaded successfully
        if self.data is None or len(self.data) == 0:
            raise ValueError("Failed to load dataset")
        
        # Replace '?' with NaN and convert to appropriate types
        self.data = self.data.replace('?', np.nan)
        for col in self.data.columns:
            if col in ['age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype('Int64')
            else:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Convert target to binary for heart disease prediction (0 = no disease, 1 = disease)
        # In the UCI dataset, 0 = no disease, 1-4 = different stages of heart disease
        self.data['target'] = self.data['target'].apply(lambda x: 0 if x == 0 else 1)
        
        # Handle missing values (if any)
        if self.data.isnull().sum().sum() > 0:
            # For numerical columns, fill with median
            for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
                self.data[col] = self.data[col].fillna(self.data[col].median())
            
            # For categorical columns, fill with mode
            for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        logger.info(f"Dataset loaded with {len(self.data)} records and {len(self.data.columns)} features")
    
    def _preprocess_data(self):
        """Preprocess the data for training."""
        # Split features and target
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Save feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Data preprocessed: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
    
    def _train_models(self):
        """Train multiple machine learning models and select the best one."""
        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        
        # Train and evaluate each model
        best_auc = 0
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_prob = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Evaluate model
            acc = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_prob)
            cm = confusion_matrix(self.y_test, y_pred)
            
            logger.info(f"{name} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            
            # Store model and metrics
            self.models[name] = {
                'model': model,
                'accuracy': acc,
                'auc': auc,
                'confusion_matrix': cm,
            }
            
            # Update best model if current one is better
            if auc > best_auc:
                best_auc = auc
                self.best_model = model
                self.best_model_name = name
        
        logger.info(f"Best model: {self.best_model_name} with AUC: {best_auc:.4f}")
    
    def _save_model(self):
        """Save the trained model and scaler to disk."""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'models': self.models
        }
        
        os.makedirs('static/models', exist_ok=True)
        with open('static/models/heart_disease_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to disk: {self.best_model_name}")
    
    def _load_model(self):
        """Load the trained model and scaler from disk."""
        with open('static/models/heart_disease_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.models = model_data['models']
        
        logger.info(f"Model loaded from disk: {self.best_model_name}")
    
    def predict(self, input_data):
        """Make a prediction on the given input data."""
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure the input features match the expected features
        missing_features = set(self.feature_names) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Reorder columns to match training data
        input_df = input_df[self.feature_names]
        
        # Scale input data
        input_scaled = self.scaler.transform(input_df)
        
        # Make prediction
        prediction_proba = self.best_model.predict_proba(input_scaled)[0, 1]
        prediction_class = 1 if prediction_proba >= 0.5 else 0
        
        # Get probabilities from all models for comparison
        model_predictions = {}
        for name, model_data in self.models.items():
            model = model_data['model']
            prob = model.predict_proba(input_scaled)[0, 1]
            model_predictions[name] = prob
        
        return {
            'probability': float(prediction_proba),
            'prediction': int(prediction_class),
            'risk_level': self._get_risk_level(prediction_proba),
            'model_name': self.best_model_name,
            'model_predictions': model_predictions
        }
    
    def _get_risk_level(self, probability):
        """Convert probability to a risk level category."""
        if probability < 0.2:
            return "Low Risk"
        elif probability < 0.4:
            return "Moderate Risk"
        elif probability < 0.6:
            return "Medium Risk"
        elif probability < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def get_dataset_info(self):
        """Get summary information about the dataset."""
        if self.data is None:
            self._load_data()
        
        # Basic dataset statistics
        info = {
            'num_records': len(self.data),
            'num_features': len(self.data.columns) - 1,  # Exclude target
            'target_distribution': self.data['target'].value_counts().to_dict(),
            'numeric_features': [col for col in self.data.columns if col != 'target'],
            'categorical_features': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        }
        
        return info
    
    def get_model_metrics(self):
        """Get the performance metrics for all models."""
        return {name: {
            'accuracy': data['accuracy'],
            'auc': data['auc'],
            'confusion_matrix': data['confusion_matrix'].tolist()
        } for name, data in self.models.items()}
    
    def get_feature_explanations(self, input_data):
        """Get explanations for how each feature affects the prediction."""
        # Define feature explanations based on medical knowledge
        explanations = {
            'age': {
                'description': 'Age in years',
                'impact': 'Higher age generally increases risk of heart disease'
            },
            'sex': {
                'description': 'Sex (1 = male, 0 = female)',
                'impact': 'Males typically have higher risk than females'
            },
            'cp': {
                'description': 'Chest pain type (0-3)',
                'impact': 'Types: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic'
            },
            'trestbps': {
                'description': 'Resting blood pressure (mm Hg)',
                'impact': 'Higher blood pressure is associated with increased risk'
            },
            'chol': {
                'description': 'Serum cholesterol (mg/dl)',
                'impact': 'Higher cholesterol is associated with increased risk'
            },
            'fbs': {
                'description': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
                'impact': 'Higher blood sugar can indicate diabetes, a risk factor'
            },
            'restecg': {
                'description': 'Resting electrocardiographic results (0-2)',
                'impact': 'Values: 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy'
            },
            'thalach': {
                'description': 'Maximum heart rate achieved',
                'impact': 'Lower maximum heart rate may indicate reduced cardiac function'
            },
            'exang': {
                'description': 'Exercise induced angina (1 = yes, 0 = no)',
                'impact': 'Exercise-induced chest pain suggests coronary artery disease'
            },
            'oldpeak': {
                'description': 'ST depression induced by exercise relative to rest',
                'impact': 'Greater depression indicates more severe ischemia'
            },
            'slope': {
                'description': 'Slope of the peak exercise ST segment (0-2)',
                'impact': 'Values: 0 = upsloping, 1 = flat, 2 = downsloping. Downsloping indicates abnormality'
            },
            'ca': {
                'description': 'Number of major vessels colored by fluoroscopy (0-3)',
                'impact': 'More affected vessels indicate more widespread disease'
            },
            'thal': {
                'description': 'Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)',
                'impact': 'Reversible defect indicates ischemia; fixed defect indicates prior heart attack'
            }
        }
        
        # Add user's specific values to explanations
        for feature, value in input_data.items():
            if feature in explanations:
                explanations[feature]['value'] = value
                
                # Add risk assessment for specific values
                if feature == 'age':
                    if value > 60:
                        explanations[feature]['assessment'] = 'High risk factor'
                    elif value > 45:
                        explanations[feature]['assessment'] = 'Moderate risk factor'
                    else:
                        explanations[feature]['assessment'] = 'Lower risk factor'
                
                elif feature == 'sex':
                    if value == 1:
                        explanations[feature]['assessment'] = 'Higher risk (male)'
                    else:
                        explanations[feature]['assessment'] = 'Lower risk (female)'
                
                elif feature == 'trestbps':
                    if value >= 140:
                        explanations[feature]['assessment'] = 'High risk (hypertension)'
                    elif value >= 120:
                        explanations[feature]['assessment'] = 'Moderate risk (pre-hypertension)'
                    else:
                        explanations[feature]['assessment'] = 'Normal range'
                
                elif feature == 'chol':
                    if value >= 240:
                        explanations[feature]['assessment'] = 'High risk (high cholesterol)'
                    elif value >= 200:
                        explanations[feature]['assessment'] = 'Moderate risk (borderline cholesterol)'
                    else:
                        explanations[feature]['assessment'] = 'Desirable range'
                
                # Add more feature-specific assessments as needed
        
        return explanations
    
    def get_feature_contributions(self, input_data):
        """Calculate the contribution of each feature to the prediction."""
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[self.feature_names]
        
        # Scale input data
        input_scaled = self.scaler.transform(input_df)
        
        # Calculate feature contributions using permutation approach
        contributions = {}
        baseline_prob = self.best_model.predict_proba(input_scaled)[0, 1]
        
        for i, feature in enumerate(self.feature_names):
            # Create a copy of the input data
            modified_input = input_scaled.copy()
            
            # Change the feature value (set to median)
            modified_input[0, i] = 0  # Use 0 for scaled data (approximately median)
            
            # Get new prediction
            modified_prob = self.best_model.predict_proba(modified_input)[0, 1]
            
            # Calculate difference in probability
            contribution = baseline_prob - modified_prob
            contributions[feature] = float(contribution)
        
        # Normalize contributions
        total_abs_contribution = sum(abs(c) for c in contributions.values())
        if total_abs_contribution > 0:
            normalized_contributions = {f: (c / total_abs_contribution) for f, c in contributions.items()}
        else:
            normalized_contributions = {f: 0 for f in contributions.keys()}
        
        # Sort by absolute contribution
        sorted_contributions = dict(sorted(
            normalized_contributions.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        ))
        
        return sorted_contributions
    
    # Visualization methods
    def _create_figure(self, plot_func, *args, **kwargs):
        """Helper method to create a figure and convert it to base64 for embedding in HTML."""
        plt.figure(figsize=(10, 6))
        plot_func(*args, **kwargs)
        plt.tight_layout()
        
        # Save figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Encode the figure as base64 string
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{img_str}'
    
    def plot_age_distribution(self):
        """Plot age distribution by heart disease status."""
        if self.data is None:
            self._load_data()
        
        def plot():
            sns.histplot(data=self.data, x='age', hue='target', multiple='dodge', bins=10)
            plt.title('Age Distribution by Heart Disease Status')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.legend(['No Disease', 'Disease'])
        
        return self._create_figure(plot)
    
    def plot_gender_distribution(self):
        """Plot gender distribution by heart disease status."""
        if self.data is None:
            self._load_data()
        
        def plot():
            gender_disease = pd.crosstab(self.data['sex'], self.data['target'])
            gender_disease.index = ['Female', 'Male']
            gender_disease.columns = ['No Disease', 'Disease']
            gender_disease.plot(kind='bar', stacked=True)
            plt.title('Gender Distribution by Heart Disease Status')
            plt.xlabel('Gender')
            plt.ylabel('Count')
        
        return self._create_figure(plot)
    
    def plot_target_distribution(self):
        """Plot the distribution of the target variable."""
        if self.data is None:
            self._load_data()
        
        def plot():
            target_counts = self.data['target'].value_counts()
            plt.pie(target_counts, labels=['No Disease', 'Disease'], autopct='%1.1f%%', colors=['#5DA5DA', '#FAA43A'])
            plt.title('Heart Disease Distribution')
            plt.axis('equal')
        
        return self._create_figure(plot)
    
    def plot_correlation_matrix(self):
        """Plot the correlation matrix of features."""
        if self.data is None:
            self._load_data()
        
        def plot():
            correlation_matrix = self.data.corr()
            mask = np.triu(correlation_matrix)
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix of Features')
        
        return self._create_figure(plot)
    
    def plot_feature_importance(self):
        """Plot feature importance for the best model."""
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        def plot():
            if self.best_model_name == 'Logistic Regression':
                # For logistic regression, use coefficients as importance
                importance = np.abs(self.best_model.coef_[0])
                feature_importance = pd.DataFrame({'Feature': self.feature_names, 'Importance': importance})
            
            elif self.best_model_name in ['Random Forest', 'XGBoost']:
                # For tree-based models, use feature_importances_
                importance = self.best_model.feature_importances_
                feature_importance = pd.DataFrame({'Feature': self.feature_names, 'Importance': importance})
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Plot
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title(f'Feature Importance ({self.best_model_name})')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
        
        return self._create_figure(plot)
