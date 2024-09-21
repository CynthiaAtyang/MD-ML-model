import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['Gender', 'DiseaseSubtype', 'GeneticMarker', 'ExerciseType'])
    
    # Split features and target
    X = data.drop(['DateOfBirth', 'OutcomeStrength', 'OutcomeFunction'], axis=1)
    y = data[['OutcomeStrength', 'OutcomeFunction']]
    
    return X, y

# Step 2: Model Creation
def create_model():
    return RandomForestRegressor(n_estimators=100, random_state=42)

# Step 3: Model Training and Evaluation
def train_and_evaluate_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = create_model()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2

# Step 4: Prediction Function
def predict_exercise_response(model, scaler, patient_data):
    # Preprocess the patient data (assuming it's a dictionary)
    patient_df = pd.DataFrame([patient_data])
    patient_df_encoded = pd.get_dummies(patient_df)
    
    # Ensure all columns from training are present
    for col in scaler.feature_names_in_:
        if col not in patient_df_encoded.columns:
            patient_df_encoded[col] = 0
    
    # Reorder columns to match the training data
    patient_df_encoded = patient_df_encoded[scaler.feature_names_in_]
    
    # Scale the data
    patient_data_scaled = scaler.transform(patient_df_encoded)
    
    # Make prediction
    prediction = model.predict(patient_data_scaled)
    
    return prediction[0]  # Return the first (and only) prediction

# Main execution
if __name__ == "__main__":
    # Load and preprocess the data
    X, y = load_and_preprocess_data("muscular_dystrophy_exercise_data.csv")
    
    # Train and evaluate the model
    model, scaler, mse, r2 = train_and_evaluate_model(X, y)
    
    print(f"Model Performance: MSE = {mse}, R2 = {r2}")
    
    # Example prediction
    sample_patient = {
        'Age': 30,
        'Gender': 'Male',
        'DiseaseSubtype': 'Duchenne',
        'GeneticMarker': 'DMD',
        'ExerciseType': 'Aerobic',
        'ExerciseDuration': 30,
        'ExerciseFrequency': 3,
        'BaselineStrength': 50,
        'BaselineFunction': 60
    }
    
    prediction = predict_exercise_response(model, scaler, sample_patient)
    print(f"Predicted outcome for sample patient: Strength = {prediction[0]}, Function = {prediction[1]}")