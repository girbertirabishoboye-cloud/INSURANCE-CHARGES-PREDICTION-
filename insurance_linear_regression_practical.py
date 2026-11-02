
"""
INSURANCE CHARGES PREDICTION
Machine Learning Practical - Questions 6 to 13
Algorithm: Linear Regression
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Insurance Charges Prediction")
st.subheader("Machine Learning Practical — Linear Regression")

st.write(
    "This application analyzes medical insurance data, "
    "trains a Linear Regression model, evaluates its performance, "
    "and predicts insurance charges for a new individual."
)


# ============================================================
# QUESTION 6: LOAD AND EXPLORE THE DATASET
# ============================================================

DATA_PATH = Path("insurance.csv")

st.header("Question 6: Dataset Exploration")

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH.resolve()}. "
        "Please upload insurance.csv to your GitHub repository "
        "in the same directory as this Python file."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

df = df[required_columns].dropna().copy()

st.write("### First Five Rows")
st.dataframe(df.head(), use_container_width=True)

st.write("### Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{df.shape[0]:,}")
col2.metric("Total Features", len(df.columns) - 1)
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.write("### Summary Statistics")
st.dataframe(df.describe(), use_container_width=True)

st.write("### Missing Values")
st.dataframe(
    df.isnull().sum().rename("Missing Values"),
    use_container_width=True
)


# ============================================================
# QUESTION 7: ONE-HOT ENCODING
# ============================================================

st.header("Question 7: One-Hot Encoding")

df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

st.write("Encoded Dataset")
st.dataframe(df_encoded.head(), use_container_width=True)

st.write("Encoded Column Names")
st.write(df_encoded.columns.tolist())


# ============================================================
# QUESTION 8: SPLIT FEATURES AND TARGET (80:20)
# ============================================================

st.header("Question 8: Training and Testing Split")

X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

col1, col2 = st.columns(2)

col1.metric("Training Records", X_train.shape[0])
col2.metric("Testing Records", X_test.shape[0])

st.write("Training Features Shape:", X_train.shape)
st.write("Testing Features Shape:", X_test.shape)


# ============================================================
# QUESTION 9: BUILD AND TRAIN LINEAR REGRESSION MODEL
# ============================================================

st.header("Question 9: Model Training")

model = LinearRegression()

model.fit(X_train, y_train)

st.success("Linear Regression model trained successfully.")

st.write("### Model Coefficients")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.dataframe(coefficients, use_container_width=True)

st.write("### Model Intercept")
st.write(round(model.intercept_, 4))


# ============================================================
# QUESTION 10: DISPLAY REGRESSION EQUATION
# ============================================================

st.header("Question 10: Regression Equation")

equation = f"charges = {model.intercept_:.4f}"

for feature, coefficient in zip(X.columns, model.coef_):
    sign = "+" if coefficient >= 0 else "-"
    equation += f" {sign} {abs(coefficient):.4f}({feature})"

st.code(equation, language="text")


# ============================================================
# QUESTION 11: MODEL EVALUATION
# ============================================================

st.header("Question 11: Model Evaluation")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"${mae:,.2f}")
col2.metric("MSE", f"${mse:,.2f}")
col3.metric("RMSE", f"${rmse:,.2f}")
col4.metric("R² Score", f"{r2:.4f}")

st.write(f"R² as a percentage: {r2 * 100:.2f}%")


# ============================================================
# QUESTION 12(i): ACTUAL VS PREDICTED PLOT
# ============================================================

st.header("Question 12(i): Actual vs Predicted Plot")

fig1, ax1 = plt.subplots(figsize=(9, 6))

ax1.scatter(
    y_test,
    y_pred,
    alpha=0.65,
    label="Test observations"
)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

ax1.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle="--",
    linewidth=2,
    label="Perfect prediction"
)

ax1.set_xlabel("Actual Insurance Charges ($)")
ax1.set_ylabel("Predicted Insurance Charges ($)")
ax1.set_title("Linear Regression: Actual vs Predicted Charges")

ax1.legend()
ax1.grid(True, alpha=0.25)

fig1.tight_layout()

st.pyplot(fig1)

fig1.savefig(
    "regression_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig1)


# ============================================================
# QUESTION 12(ii): RESIDUAL PLOT
# ============================================================

st.header("Question 12(ii): Residual Plot")

residuals = y_test - y_pred

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2.scatter(
    y_pred,
    residuals,
    alpha=0.65
)

ax2.axhline(
    y=0,
    linestyle="--",
    linewidth=2
)

ax2.set_xlabel("Predicted Insurance Charges ($)")
ax2.set_ylabel("Residual (Actual - Predicted) ($)")
ax2.set_title("Residual Plot for Linear Regression")

ax2.grid(True, alpha=0.25)

fig2.tight_layout()

st.pyplot(fig2)

fig2.savefig(
    "regression_residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig2)


# ============================================================
# QUESTION 13: PREDICT CHARGES FOR A NEW INDIVIDUAL
# ============================================================

st.header("Question 13: New Individual Prediction")

st.write(
    "Predict insurance charges for the following individual:"
)

new_person = pd.DataFrame([{
    "age": 19,
    "sex": "female",
    "bmi": 27.9,
    "children": 0,
    "smoker": "yes",
    "region": "southwest"
}])

st.dataframe(new_person, use_container_width=True)

new_person_encoded = pd.get_dummies(
    new_person,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

new_person_encoded = new_person_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

predicted_charge = model.predict(new_person_encoded)[0]

st.success(
    f"Predicted Insurance Charge: ${predicted_charge:,.2f}"
)


# ============================================================
# SAVE PREDICTION RESULTS
# ============================================================

results = pd.DataFrame({
    "actual_charges": y_test.values,
    "predicted_charges": y_pred,
    "residuals": residuals.values
})

results.to_csv(
    "insurance_prediction_results.csv",
    index=False
)

st.download_button(
    label="Download Prediction Results CSV",
    data=results.to_csv(index=False),
    file_name="insurance_prediction_results.csv",
    mime="text/csv"
)

st.info(
    "Practical completed successfully. "
    "The model, evaluation metrics, charts, and new individual "
    "prediction have been generated."
)
