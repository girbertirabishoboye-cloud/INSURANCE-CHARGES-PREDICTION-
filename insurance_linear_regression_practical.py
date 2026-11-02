
"""
INSURANCE CHARGES PREDICTION
Linear Regression Practical - Questions 6 to 13
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CUSTOM APPEARANCE
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #0E1628;
    color: #E5E7EB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

h1 {
    color: #60A5FA !important;
    font-weight: 700;
}

h2 {
    color: #93C5FD !important;
}

h3 {
    color: #CBD5E1 !important;
}

div[data-testid="stMetric"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #60A5FA;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

hr {
    border-color: #334155;
}

.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("🏥 Insurance Charges Prediction")

st.markdown("""
A Machine Learning practical using **Linear Regression** to analyze
medical insurance data and predict individual insurance charges.
""")

st.divider()


# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = Path("insurance.csv")

if not DATA_PATH.exists():
    st.error(
        "The insurance.csv file was not found. "
        "Please upload it to the same GitHub folder as this Python file."
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
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

df = df[required_columns].dropna().copy()


# ============================================================
# QUESTION 6: LOAD AND DISPLAY DATASET
# ============================================================

st.header("Question 6: Dataset Loading and Exploration")

st.markdown("### First Five Rows")

st.dataframe(
    df.head(),
    use_container_width=True,
    hide_index=True
)

st.markdown("### Summary Statistics")

st.dataframe(
    df.describe(),
    use_container_width=True
)


# ============================================================
# QUESTION 7: ONE-HOT ENCODING
# ============================================================

st.divider()

st.header("Question 7: Handling Categorical Variables")

st.markdown("""
Categorical variables are converted into numerical features
using One-Hot Encoding.
""")

df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

st.dataframe(
    df_encoded.head(),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# QUESTION 8: SPLIT FEATURES AND TARGET
# ============================================================

st.divider()

st.header("Question 8: Training and Testing Data")

X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Data (80%)",
        f"{len(X_train):,} records"
    )

with col2:
    st.metric(
        "Testing Data (20%)",
        f"{len(X_test):,} records"
    )


# ============================================================
# QUESTION 9: BUILD AND TRAIN LINEAR REGRESSION
# ============================================================

st.divider()

st.header("Question 9: Linear Regression Model")

model = LinearRegression()

model.fit(X_train, y_train)

st.success("Linear Regression model trained successfully.")

st.markdown("### Model Coefficients")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.dataframe(
    coefficients,
    use_container_width=True,
    hide_index=True
)

st.markdown("### Model Intercept")

st.code(f"{model.intercept_:.4f}")


# ============================================================
# QUESTION 10: REGRESSION EQUATION
# ============================================================

st.divider()

st.header("Question 10: Linear Regression Equation")

equation = f"charges = {model.intercept_:.4f}"

for feature, coefficient in zip(X.columns, model.coef_):
    sign = "+" if coefficient >= 0 else "-"
    equation += (
        f" {sign} {abs(coefficient):.4f}({feature})"
    )

with st.expander("View the complete regression equation", expanded=True):
    st.code(equation, language="text")


# ============================================================
# QUESTION 11: MODEL EVALUATION
# ============================================================

st.divider()

st.header("Question 11: Model Evaluation")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Mean Absolute Error (MAE)",
        f"${mae:,.2f}"
    )

with col2:
    st.metric(
        "Mean Squared Error (MSE)",
        f"${mse:,.2f}"
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2 * 100:.2f}%"
    )

st.caption(
    "R² percentage represents the proportion of variation in "
    "insurance charges explained by the model. It is not classification accuracy."
)


# ============================================================
# QUESTION 12(i): REGRESSION PLOT
# ============================================================

st.divider()

st.header("Question 12(i): Regression Plot")

fig1, ax1 = plt.subplots(figsize=(9, 5))

fig1.patch.set_facecolor("#0E1628")
ax1.set_facecolor("#1E293B")

ax1.scatter(
    y_test,
    y_pred,
    color="#60A5FA",
    alpha=0.75,
    edgecolors="none",
    label="Predicted charges"
)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

ax1.plot(
    [line_min, line_max],
    [line_min, line_max],
    color="#F87171",
    linestyle="--",
    linewidth=2,
    label="Ideal prediction line"
)

ax1.set_xlabel(
    "Actual Insurance Charges ($)",
    color="white"
)

ax1.set_ylabel(
    "Predicted Insurance Charges ($)",
    color="white"
)

ax1.set_title(
    "Actual vs Predicted Insurance Charges",
    color="white"
)

ax1.tick_params(colors="white")
ax1.grid(alpha=0.15)
ax1.legend()

fig1.tight_layout()

st.pyplot(fig1)

plt.close(fig1)


# ============================================================
# QUESTION 12(ii): CONFUSION MATRIX NOTE
# ============================================================

st.markdown("### Question 12(ii): Confusion Matrix Heatmap")

st.info("""
A confusion matrix is designed for classification problems.
This practical uses Linear Regression to predict continuous
insurance charges, so a confusion matrix is not mathematically
appropriate. The regression plot above is used to evaluate
predicted charges against actual charges.
""")


# ============================================================
# QUESTION 13: PREDICT INSURANCE CHARGES
# ============================================================

st.divider()

st.header("Question 13: Insurance Charges Prediction")

st.markdown("### Individual Information")

new_person = pd.DataFrame([{
    "age": 19,
    "sex": "female",
    "bmi": 27.9,
    "children": 0,
    "smoker": "yes",
    "region": "southwest"
}])

st.dataframe(
    new_person,
    use_container_width=True,
    hide_index=True
)

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

predicted_charge = model.predict(
    new_person_encoded
)[0]

st.markdown("### Predicted Insurance Charge")

st.metric(
    label="Estimated Insurance Charges",
    value=f"${predicted_charge:,.2f}"
)

st.caption(
    "Prediction for a 19-year-old female, BMI 27.9, "
    "with 0 children, smoker=yes, region=southwest."
)

st.divider()

st.markdown("""
<div style="text-align:center; color:#94A3B8; padding:10px;">
    Insurance Charges Prediction | Linear Regression Practical
</div>
""", unsafe_allow_html=True)
