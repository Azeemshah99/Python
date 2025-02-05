import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load data (same as before, but inside a function for Streamlit):
@st.cache_data  # Cache the data loading to improve performance
def load_data():
    import os
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

df = load_data()  # Load the data

# 2. Separate features and target (same as before):
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 3. Split data (same as before):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model (same as before, but inside a function and cached):
@st.cache_resource # Cache the model training
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Streamlit app:
st.title("California Housing Price Prediction")

# Display dataset (optional):
if st.checkbox("Show Dataset"):
    st.write(df)

# Input for prediction:
st.subheader("Make a Prediction")
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f"Enter value for {feature}")

# Make prediction:
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[X.columns] # Ensure correct column order

    # Handle missing input values (important):
    if input_df.isnull().any().any():
        st.warning("Please fill in all feature values. Missing values are handled by imputing with mean.")
        input_df = input_df.fillna(X_train.mean())  # Impute missing values
    
    predictions = model.predict(input_df)
    st.write(f"Predicted Median House Value: ${predictions[0]:.2f}")

# Model evaluation:
st.subheader("Model Evaluation (on Test Set)")
predictions_test = model.predict(X_test)
mse = mean_squared_error(y_test, predictions_test)
st.write(f"Mean Squared Error (on test set): {mse}")

# About the model:
st.subheader("About the Model")
st.write("This app uses a Linear Regression model to predict median house values in California based on various features.  The model is trained on the California housing dataset from scikit-learn.")

st.write("Note: This is a simplified example. Real-world applications would involve more complex data processing, feature engineering, model selection, and evaluation techniques.")