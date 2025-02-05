import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import altair as alt
import numpy as np  # For numerical operations

# 1. Load data:
@st.cache_data
def load_data():
    import os
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

df = load_data()

# 2. Separate features and target:
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 3. Split data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model:
@st.cache_resource
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Streamlit app:
st.title("California Housing Price Prediction")

# --- Visualizations and Insights ---
st.sidebar.subheader("Data Exploration")

# Display dataset:
if st.sidebar.checkbox("Show Dataset"):
    st.dataframe(df.head(10))

# Feature Distributions (Histograms):
st.sidebar.subheader("Feature Distributions")
for feature in df.columns[:-1]:
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(feature, bin=True),
        y='count()'
    ).properties(title=feature)
    st.altair_chart(chart, use_container_width=True)

# Scatter plot with trend line:
st.sidebar.subheader("Feature vs. Target Scatter Plot")
selected_feature = st.sidebar.selectbox("Select a feature for scatter plot", df.columns[:-1])
scatter_chart = alt.Chart(df).mark_circle(size=60).encode(
    x=selected_feature,
    y='MedHouseVal',
    tooltip=[selected_feature, 'MedHouseVal']
).properties(title=f"Scatter Plot of {selected_feature} vs. Median House Value")

# Add a trend line (linear regression):
trend_line = scatter_chart.transform_regression(
    selected_feature, 'MedHouseVal'
).mark_line(color='red')

st.altair_chart(scatter_chart + trend_line, use_container_width=True)


# Correlation matrix:
st.sidebar.subheader("Correlation Matrix")
corr = df.corr()
st.write(corr)  # Display the correlation matrix

# --- Prediction ---
st.subheader("Make a Prediction")
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f"Enter value for {feature}")

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[X.columns]

    # Handle missing input values:
    if input_df.isnull().any().any():
        st.warning("Please fill in all feature values. Missing values are handled by imputing with mean.")
        input_df = input_df.fillna(X_train.mean())
    
    predictions = model.predict(input_df)
    st.write(f"Predicted Median House Value: ${predictions[0]:.2f}")

# --- Model Evaluation ---
st.subheader("Model Evaluation (on Test Set)")
predictions_test = model.predict(X_test)
mse = mean_squared_error(y_test, predictions_test)
st.write(f"Mean Squared Error (on test set): {mse}")

# R-squared:
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions_test)
st.write(f"R-squared (on test set): {r2}")

# --- Predictions vs. Actual (Test Set) ---
st.subheader("Predictions vs. Actual (Test Set)")
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_test})

# Altair interactive plot:
predictions_chart = alt.Chart(results_df.reset_index()).mark_line(point=True).encode(
    x='index',
    y=alt.Y('Actual', title='Median House Value'),
    tooltip=['index', 'Actual', 'Predicted']
).properties(title="Actual vs. Predicted Values")

st.altair_chart(predictions_chart, use_container_width=True)

# --- About the model ---
st.subheader("About the Model")
st.write("This app uses a Linear Regression model to predict median house values in California based on various features.  The model is trained on the California housing dataset from scikit-learn.")

st.write("Note: This is a simplified example. Real-world applications would involve more complex data processing, feature engineering, model selection, and evaluation techniques.")